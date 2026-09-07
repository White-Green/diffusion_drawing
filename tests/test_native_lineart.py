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
        model = Mock()
        model.infer.side_effect = lambda **inputs: (threads.append(threading.get_ident()), b"result")[1]
        factory = Mock(side_effect=lambda: (threads.append(threading.get_ident()), model)[1])
        inputs = dict(scribble=b"s", lineart=b"l", width=16, height=16, strength=0.5, seed=42, denoise_steps=2)
        worker = native_lineart.NativeLineart()
        with patch.object(native_lineart.importlib, "import_module", return_value=types.SimpleNamespace(LineartModel=factory)):
            results = await asyncio.gather(worker.infer(**inputs), worker.infer(**inputs))
        self.assertEqual(results, [b"result", b"result"])
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
        docker.document_nodes_map = {"original-document": types.SimpleNamespace(lineart="original-layer", tmp_dir="")}
        docker.disable_buttons, docker.enable_buttons, docker.log = Mock(), Mock(), Mock()
        docker.lineart_strength = Mock(value=Mock(return_value=0.5))
        docker.lineart_seed = Mock(value=Mock(return_value=42))
        docker.lineart_steps = Mock(value=Mock(return_value=1))
        docker.native_lineart = Mock(infer=AsyncMock(return_value=bytes(16 * 16 * 4)))
        self.ui.krita.Krita.instance().documents.return_value = [document]
        self.addCleanup(lambda: docker.document_nodes_map.clear())
        return docker, document, layer

    async def test_document_switch_during_inference_updates_original_document(self):
        docker, document, layer = self.make_docker()
        other_document = Mock()

        async def infer(**inputs):
            docker.active_document = other_document
            return bytes(16 * 16 * 4)

        docker.native_lineart.infer.side_effect = infer
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            await docker.gen_lineart_inner()
        layer.setPixelData.assert_called_once()
        document.nodeByUniqueID.assert_called_once_with("original-layer")
        other_document.nodeByUniqueID.assert_not_called()
        layer.setLocked.assert_called_with(True)
        layer.setColorSpace.assert_any_call("RGBA", "U8", native_lineart.SRGB_PROFILE)
        layer.setColorSpace.assert_called_with("RGBA", "U16", "document-profile")
        docker.enable_buttons.assert_called_once()

    async def test_closed_document_is_not_written(self):
        docker, document, layer = self.make_docker()
        self.ui.krita.Krita.instance().documents.return_value = []
        with patch.object(self.ui, "capture_bgra", return_value=bytes(16 * 16 * 4)):
            await docker.gen_lineart_inner()
        layer.setPixelData.assert_not_called()
        docker.enable_buttons.assert_called_once()

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
        docker.enable_buttons.assert_called_once()


if __name__ == "__main__":
    unittest.main()
