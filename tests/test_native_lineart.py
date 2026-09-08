import asyncio
import copy
import importlib
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_diffusion_drawing_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "diffusion_drawing")]
sys.modules[PACKAGE] = package
native_lineart = importlib.import_module(f"{PACKAGE}.native_lineart")


class Layer:
    def __init__(self, kind="paintlayer", label=0, children=()):
        self.kind, self.label = kind, label
        self.children = list(children)
        self.visible = True

    def childNodes(self):
        return self.children

    def type(self):
        return self.kind

    def colorLabel(self):
        return self.label

    def setVisible(self, visible):
        self.visible = visible


class CaptureTests(unittest.TestCase):
    def document(self):
        root = Layer("grouplayer", children=[
            Layer(label=1, children=[Layer("transparencymask")]),
            Layer("grouplayer", children=[Layer(label=2)]),
            Layer("filtermask"),
        ])
        snapshot = Mock()
        snapshot.rootNode.return_value = copy.deepcopy(root)
        snapshot.rootNode.return_value.projectionPixelData = Mock(return_value=bytes(16 * 16 * 4))
        snapshot.width.return_value = snapshot.height.return_value = 16
        snapshot.setColorSpace.return_value = True
        document = Mock()
        document.clone.return_value = snapshot
        document.rootNode.return_value = root
        return document, snapshot

    def test_capture_preserves_original_layers_and_included_masks(self):
        document, snapshot = self.document()
        result = native_lineart.capture_bgra(document, 1)
        self.assertEqual(len(result), 16 * 16 * 4)
        snapshot.setColorSpace.assert_called_once_with("RGBA", "U8", native_lineart.SRGB_PROFILE)
        layers = snapshot.rootNode().children
        self.assertTrue(layers[0].visible)
        self.assertTrue(layers[0].children[0].visible)
        self.assertFalse(layers[1].children[0].visible)
        self.assertTrue(layers[2].visible)
        self.assertTrue(document.rootNode().children[1].children[0].visible)
        snapshot.close.assert_called_once()

    def test_failed_conversion_closes_snapshot(self):
        document, snapshot = self.document()
        snapshot.setColorSpace.return_value = False
        with self.assertRaisesRegex(RuntimeError, "convert"):
            native_lineart.capture_bgra(document, 1)
        snapshot.close.assert_called_once()

    def test_incomplete_buffer_closes_snapshot(self):
        document, snapshot = self.document()
        snapshot.rootNode().projectionPixelData.return_value = b""
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            native_lineart.capture_bgra(document, 1)
        snapshot.close.assert_called_once()


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_load_and_inference_run_off_ui_thread_and_reuse_model(self):
        main_thread = threading.get_ident()
        threads = []
        model = Mock(backend="wgpu", device="Test GPU (DiscreteGpu, Dx12)")
        model.infer.side_effect = lambda **inputs: (threads.append(threading.get_ident()), b"result")[1]
        factory = Mock(side_effect=lambda: (threads.append(threading.get_ident()), model)[1])
        inputs = dict(scribble=b"s", lineart=b"l", width=16, height=16, strength=0.5, seed=42, denoise_steps=2)
        worker = native_lineart.NativeLineart()
        self.assertIsNone(worker.backend_description)
        with patch.object(native_lineart.importlib, "import_module", return_value=types.SimpleNamespace(LineartModel=factory)):
            results = await asyncio.gather(worker.infer(**inputs), worker.infer(**inputs))
        self.assertEqual(results, [b"result", b"result"])
        self.assertEqual(worker.backend_description, "wgpu: Test GPU (DiscreteGpu, Dx12)")
        factory.assert_called_once()
        model.infer.assert_called_with(**inputs)
        self.assertTrue(all(thread != main_thread for thread in threads))

    async def test_missing_extension_produces_actionable_error(self):
        worker = native_lineart.NativeLineart()
        with patch.object(native_lineart.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "operating system and CPU architecture"):
                await worker.infer(scribble=b"", lineart=b"", width=16, height=16,
                                   strength=0.5, seed=0, denoise_steps=1)


def load_docker():
    krita = types.ModuleType("krita")
    for name in ["DockWidget", "Node", "Document", "Canvas"]:
        setattr(krita, name, type(name, (), {}))
    krita.Krita = Mock()
    qtcore = types.ModuleType("PyQt5.QtCore")
    qtcore.QUuid = type("QUuid", (), {})
    qtcore.QByteArray = bytes
    qtcore.pyqtSlot = lambda *args: lambda method: method
    modules = {"krita": krita, "PyQt5": types.ModuleType("PyQt5"),
               "PyQt5.QtCore": qtcore, "PyQt5.QtWidgets": types.ModuleType("PyQt5.QtWidgets")}
    with patch.dict(sys.modules, modules):
        return importlib.import_module(f"{PACKAGE}.diffusion_drawing")


class Toggle:
    """Emit toggled synchronously, as Qt does for setChecked()."""

    def __init__(self, checked, changed):
        self.checked = checked
        self.changed = changed
        self.enabled = True

    def isChecked(self):
        return self.checked

    def setChecked(self, checked):
        if checked != self.checked:
            self.checked = checked
            self.changed(checked)

    def setEnabled(self, enabled):
        self.enabled = enabled


class ApplyTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = load_docker()

    def make_docker(self):
        docker = self.ui.DiffusionDrawingDocker.__new__(self.ui.DiffusionDrawingDocker)
        document = Mock()
        document.width.return_value = document.height.return_value = 16
        document.colorModel.return_value = "RGBA"
        document.colorDepth.return_value = "U16"
        document.colorProfile.return_value = "document-profile"
        document.rootNode().uniqueId.return_value = "original-document"
        layer = Mock()
        layer.setPixelData.return_value = True
        document.nodeByUniqueID.return_value = layer
        docker.active_document = document
        docker._lineart_task = None
        docker._lineart_generation = 0
        docker.gen_lineart_button = Toggle(True, docker.gen_lineart)
        docker.document_nodes_map = {"original-document": types.SimpleNamespace(lineart="original-layer", tmp_dir="")}
        docker.disable_buttons, docker.enable_buttons, docker.log = Mock(), Mock(), Mock()
        docker.lineart_strength = Mock(value=Mock(return_value=0.5))
        docker.lineart_seed = Mock(value=Mock(return_value=42))
        docker.lineart_steps = Mock(value=Mock(return_value=1))
        docker.native_lineart = Mock(infer=AsyncMock(return_value=bytes(16 * 16 * 4)),
                                     backend_description="wgpu: Test GPU")
        docker.setup_area_none, docker.setup_area_ready, docker.setup_area_initialize = Mock(), Mock(), Mock()
        self.ui.krita.Krita.instance().documents.return_value = [document]
        self.addCleanup(lambda: docker.document_nodes_map.clear())
        return docker, document, layer

    async def test_generated_pixels_use_the_document_color_space(self):
        docker, document, layer = self.make_docker()
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            await docker.gen_lineart_inner()
        layer.setPixelData.assert_called_once()
        document.nodeByUniqueID.assert_called_once_with("original-layer")
        layer.setLocked.assert_called_with(True)
        layer.setColorSpace.assert_any_call("RGBA", "U8", native_lineart.SRGB_PROFILE)
        layer.setColorSpace.assert_called_with("RGBA", "U16", "document-profile")

    async def test_document_switch_stops_generation_and_discards_pending_result(self):
        docker, document, layer = self.make_docker()
        other_document = Mock()
        self.ui.krita.Krita.instance().activeDocument.return_value = other_document

        async def infer(**inputs):
            docker.canvasChanged(None)
            return bytes(16 * 16 * 4)

        docker.native_lineart.infer.side_effect = infer
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            task = self.start_loop(docker)
            await asyncio.wait_for(task, 2)
        layer.setPixelData.assert_not_called()
        other_document.nodeByUniqueID.assert_not_called()
        self.assertFalse(docker.gen_lineart_button.isChecked())
        self.assertIsNone(docker._lineart_task)

    async def test_closed_document_is_not_written(self):
        docker, document, layer = self.make_docker()
        async def infer(**inputs):
            self.ui.krita.Krita.instance().documents.return_value = []
            return bytes(16 * 16 * 4)

        docker.native_lineart.infer.side_effect = infer
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            task = self.start_loop(docker)
            await asyncio.wait_for(task, 2)
        layer.setPixelData.assert_not_called()
        docker.enable_buttons.assert_called_once()
        self.assertFalse(docker.gen_lineart_button.isChecked())

    async def test_resize_during_inference_is_rejected(self):
        docker, document, layer = self.make_docker()

        async def infer(**inputs):
            document.width.return_value = 32
            return bytes(16 * 16 * 4)

        docker.native_lineart.infer.side_effect = infer
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            with self.assertRaisesRegex(RuntimeError, "resized"):
                await docker.gen_lineart_inner()
        layer.setPixelData.assert_not_called()

    def start_loop(self, docker):
        docker.gen_lineart_button.setChecked(False)
        docker.gen_lineart_button.setChecked(True)
        return docker._lineart_task

    async def test_unchanged_input_is_generated_and_applied_repeatedly(self):
        docker, document, layer = self.make_docker()
        document.modified.return_value = False

        def apply(*args):
            if layer.setPixelData.call_count == 3:
                docker.gen_lineart_button.setChecked(False)
            return True

        layer.setPixelData.side_effect = apply
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)) as capture:
            task = self.start_loop(docker)
            await asyncio.wait_for(task, 2)
        self.assertEqual(docker.native_lineart.infer.await_count, 3)
        self.assertEqual(layer.setPixelData.call_count, 3)
        self.assertEqual(capture.call_count, 6)
        document.modified.assert_not_called()
        docker.disable_buttons.assert_called_once_with(keep_lineart_controls=True)
        docker.enable_buttons.assert_called_once()
        self.assertIsNone(docker._lineart_task)
        backend_logs = [call for call in docker.log.call_args_list if "Lineart backend" in str(call)]
        self.assertEqual(len(backend_logs), 1)

    async def test_off_during_inference_discards_result_and_stops(self):
        docker, document, layer = self.make_docker()
        entered, release = asyncio.Event(), asyncio.Event()

        async def infer(**inputs):
            entered.set()
            await release.wait()
            return bytes(16 * 16 * 4)

        docker.native_lineart.infer.side_effect = infer
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            task = self.start_loop(docker)
            await asyncio.wait_for(entered.wait(), 2)
            docker.gen_lineart_button.setChecked(False)
            self.assertFalse(task.done())  # Keep ownership of the native worker until it finishes.
            release.set()
            await asyncio.wait_for(task, 2)
        layer.setPixelData.assert_not_called()
        self.assertEqual(docker.native_lineart.infer.await_count, 1)
        self.assertIsNone(docker._lineart_task)
        docker.enable_buttons.assert_called_once()

    async def test_rapid_off_on_reuses_loop_and_drops_previous_run_result(self):
        docker, document, layer = self.make_docker()
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        outputs = [bytes([0, 0, 0, alpha]) * 16 * 16 for alpha in (12, 24)]
        calls, active, max_active = 0, 0, 0

        async def infer(**inputs):
            nonlocal calls, active, max_active
            index = calls
            calls += 1
            active += 1
            max_active = max(max_active, active)
            try:
                entered[index].set()
                await release[index].wait()
                return outputs[index]
            finally:
                active -= 1

        def apply(*args):
            docker.gen_lineart_button.setChecked(False)
            return True

        docker.native_lineart.infer.side_effect = infer
        layer.setPixelData.side_effect = apply
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            task = self.start_loop(docker)
            await asyncio.wait_for(entered[0].wait(), 2)
            docker.gen_lineart_button.setChecked(False)
            docker.gen_lineart_button.setChecked(True)
            self.assertIs(docker._lineart_task, task)
            release[0].set()
            await asyncio.wait_for(entered[1].wait(), 2)
            layer.setPixelData.assert_not_called()
            release[1].set()
            await asyncio.wait_for(task, 2)
        self.assertEqual(max_active, 1)
        self.assertEqual(calls, 2)
        layer.setPixelData.assert_called_once_with(outputs[1], 0, 0, 16, 16)
        docker.disable_buttons.assert_called_once_with(keep_lineart_controls=True)

    async def test_error_turns_toggle_off_and_restores_controls(self):
        docker, document, layer = self.make_docker()
        docker.native_lineart.infer.side_effect = RuntimeError("GPU failure")
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            task = self.start_loop(docker)
            await asyncio.wait_for(task, 2)
        self.assertFalse(docker.gen_lineart_button.isChecked())
        self.assertIsNone(docker._lineart_task)
        self.assertEqual(docker.native_lineart.infer.await_count, 1)
        docker.enable_buttons.assert_called_once()
        self.assertTrue(any("GPU failure" in str(call) for call in docker.log.call_args_list))

    async def test_new_document_can_start_while_previous_inference_is_finishing(self):
        docker, document, layer = self.make_docker()
        other_document, other_layer = Mock(), Mock()
        other_document.width.return_value = other_document.height.return_value = 16
        other_document.rootNode().uniqueId.return_value = "other-document"
        other_document.nodeByUniqueID.return_value = other_layer
        docker.document_nodes_map["other-document"] = types.SimpleNamespace(lineart="other-layer", tmp_dir="")
        self.ui.krita.Krita.instance().documents.return_value = [document, other_document]
        self.ui.krita.Krita.instance().activeDocument.return_value = other_document
        calls = 0

        async def infer(**inputs):
            nonlocal calls
            calls += 1
            if calls == 1:
                docker.canvasChanged(None)
                docker.gen_lineart_button.setChecked(True)
            return bytes(16 * 16 * 4)

        def apply(*args):
            docker.gen_lineart_button.setChecked(False)
            return True

        docker.native_lineart.infer.side_effect = infer
        other_layer.setPixelData.side_effect = apply
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            task = self.start_loop(docker)
            await asyncio.wait_for(task, 2)
        self.assertEqual(calls, 2)
        layer.setPixelData.assert_not_called()
        other_layer.setPixelData.assert_called_once()
        other_document.nodeByUniqueID.assert_called_once_with("other-layer")

    def test_continuous_generation_leaves_stop_and_options_enabled(self):
        docker, document, layer = self.make_docker()
        for name in ("initialize_button", "lineart_options", "gen_detail_button",
                     "lineart_transfer_toggle", "shadow_transfer_toggle", "light_transfer_toggle"):
            setattr(docker, name, Mock())
        self.ui.DiffusionDrawingDocker.disable_buttons(docker, keep_lineart_controls=True)
        self.assertTrue(docker.gen_lineart_button.enabled)
        docker.lineart_options.setEnabled.assert_called_with(True)
        docker.gen_detail_button.setEnabled.assert_called_with(False)
        docker.initialize_button.setEnabled.assert_called_with(False)


if __name__ == "__main__":
    unittest.main()
