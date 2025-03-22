import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Coroutine

import krita
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .diffusion_controller import DiffusionController

EMBEDDED_EMPTY_IMAGE_FILE_NAME = "empty.png"

DIFFUSION_DRAWING_LAYER_MARKER = "[DiffusionDrawing SystemLayer]"
LINEART_LAYER_NAME = f"lineart{DIFFUSION_DRAWING_LAYER_MARKER}"
SHADOW_LAYER_NAME = f"shadow{DIFFUSION_DRAWING_LAYER_MARKER}"
LIGHT_LAYER_NAME = f"light{DIFFUSION_DRAWING_LAYER_MARKER}"

LINEART_FILE_NAME = "lineart.png"
SHADOW_FILE_NAME = "shadow.png"
LIGHT_FILE_NAME = "light.png"

SYSTEM_LAYER_DEFAULT_OPACITY = 127

SCRIBBLE_COLOR_LABEL = 1
LINEART_COLOR_LABEL = 2
BASE_COLOR_COLOR_LABEL = 3
SHADOW_COLOR_LABEL = 4
LIGHT_COLOR_LABEL = 5


def hide_layers(node: krita.Node, allow_labels: list[int]):
    if node.type() != "grouplayer" and node.colorLabel() not in allow_labels:
        node.setVisible(False)
    for n in node.childNodes():
        hide_layers(n, allow_labels)


@dataclass
class SystemLayers:
    tmp_dir: str
    lineart: QUuid | None = None
    shadow: QUuid | None = None
    light: QUuid | None = None


class DiffusionDrawingDocker(krita.DockWidget):
    def __init__(self):
        super().__init__()
        self.diffusion_controller = DiffusionController()

        self.active_document: krita.Document = None
        self.document_nodes_map: dict[QUuid, SystemLayers] = {}

        self.event_loop = asyncio.get_event_loop()
        self.async_timer = QTimer(self)
        self.async_timer.timeout.connect(self.asyncio_step)
        self.async_timer.start(10)

        self.setWindowTitle("DiffusionDrawing Control Panel")

        self.initialize_button = None

        self.main_widget = QWidget(self)
        self.setWidget(self.main_widget)
        self.main_widget.setLayout(QVBoxLayout())

        self.setup_area = QWidget(self.main_widget)
        self.setup_area.setLayout(QHBoxLayout())
        self.main_widget.layout().addWidget(self.setup_area)

        self.main_area = QWidget(self.main_widget)
        self.main_area.setLayout(QGridLayout())
        self.main_widget.layout().addWidget(self.main_area)

        # lineart
        lineart_label = QLabel("lineart")
        self.lineart_transfer_toggle = QCheckBox("Transfer")
        self.lineart_transfer_toggle.stateChanged.connect(
            lambda state: self.handle_lineart_transfer(state == Qt.Checked))
        self.gen_lineart_button = QPushButton("Gen")
        self.gen_lineart_button.clicked.connect(self.gen_lineart)

        # row=0にlineart関連を配置
        self.main_area.layout().addWidget(lineart_label, 0, 0)
        self.main_area.layout().addWidget(self.lineart_transfer_toggle, 0, 1)
        self.main_area.layout().addWidget(self.gen_lineart_button, 0, 2)

        # shadow
        shadow_label = QLabel("shadow")
        self.shadow_transfer_toggle = QCheckBox("Transfer")
        self.shadow_transfer_toggle.stateChanged.connect(lambda state: self.handle_shadow_transfer(state == Qt.Checked))
        self.gen_detail_button = QPushButton("Gen")
        self.gen_detail_button.clicked.connect(self.gen_detail_colored)

        # row=1にshadow関連を配置
        self.main_area.layout().addWidget(shadow_label, 1, 0)
        self.main_area.layout().addWidget(self.shadow_transfer_toggle, 1, 1)
        self.main_area.layout().addWidget(self.gen_detail_button, 1, 2, 2, 1)

        # light
        light_label = QLabel("light")
        self.light_transfer_toggle = QCheckBox("Transfer")
        self.light_transfer_toggle.stateChanged.connect(lambda state: self.handle_light_transfer(state == Qt.Checked))

        # row=2にlight関連を配置
        self.main_area.layout().addWidget(light_label, 2, 0)
        self.main_area.layout().addWidget(self.light_transfer_toggle, 2, 1)

        self.log_window = QTextBrowser(self.main_widget)
        self.log_window.setReadOnly(True)
        self.main_widget.layout().addWidget(self.log_window)

        self.setup_area_none()

    def __del__(self):
        for system_layers in self.document_nodes_map.values():
            os.removedirs(system_layers.tmp_dir)

    def asyncio_step(self):
        self.event_loop.call_soon(self.event_loop.stop)
        self.event_loop.run_forever()

    def clear_setup_area(self):
        for i in reversed(range(self.setup_area.layout().count())):
            widget = self.setup_area.layout().itemAt(i).widget()
            self.setup_area.layout().removeWidget(widget)

    def setup_area_none(self):
        self.clear_setup_area()
        label = QLabel("document not loaded")
        self.setup_area.layout().addWidget(label)

    def setup_area_initialize(self):
        self.clear_setup_area()
        label = QLabel("not initialized")
        self.initialize_button = QPushButton("initialize")
        self.setup_area.layout().addWidget(label)
        self.setup_area.layout().addWidget(self.initialize_button)
        self.initialize_button.clicked.connect(self.initialize_document)

    def setup_area_ready(self):
        self.clear_setup_area()
        label = QLabel("ready")
        self.initialize_button = QPushButton("reload")
        self.setup_area.layout().addWidget(label)
        self.setup_area.layout().addWidget(self.initialize_button)
        self.initialize_button.clicked.connect(self.initialize_document)

    def disable_buttons(self):
        if self.initialize_button is not None:
            self.initialize_button.setEnabled(False)

        self.gen_lineart_button.setEnabled(False)
        self.gen_detail_button.setEnabled(False)
        self.lineart_transfer_toggle.setChecked(False)
        self.shadow_transfer_toggle.setChecked(False)
        self.light_transfer_toggle.setChecked(False)
        self.lineart_transfer_toggle.setEnabled(False)
        self.shadow_transfer_toggle.setEnabled(False)
        self.light_transfer_toggle.setEnabled(False)

    def enable_buttons(self):
        if self.initialize_button is not None:
            self.initialize_button.setEnabled(True)

        self.gen_lineart_button.setEnabled(True)
        self.gen_detail_button.setEnabled(True)
        self.lineart_transfer_toggle.setEnabled(True)
        self.shadow_transfer_toggle.setEnabled(True)
        self.light_transfer_toggle.setEnabled(True)

    def export_image_filtered_color_label(self, allow_labels: list[int], alpha: bool, output_path: str):
        document = self.active_document.clone()
        hide_layers(document.rootNode(), allow_labels)
        output_info = krita.InfoObject()
        output_info.setProperties(
            {"alpha": alpha, "compression": 1, "forceSRGB": False, "indexed": False, "interlaced": False,
             "saveSRGBProfile": False, "transparencyFillcolor": [255, 255, 255]})
        document.refreshProjection()
        document.waitForDone()
        document.setBatchmode(True)

        document.exportImage(output_path, output_info)

    def apply_layer_mask_filtered_color_label(self, allow_labels: list[int]):
        def traverse(node: krita.Node):
            if node.type() == "grouplayer":
                for n in node.childNodes():
                    traverse(n)
            elif node.colorLabel() in allow_labels:
                bounds = node.bounds()
                pixels = node.projectionPixelData(bounds.left(), bounds.top(), bounds.width(), bounds.height())

                node.setPixelData(pixels, bounds.left(), bounds.top(), bounds.width(), bounds.height())
                for n in node.childNodes():
                    node.removeChildNode(n)

        traverse(self.active_document.rootNode())

    def create_transparency_mask_from_layer_filtered_color_label(self, allow_labels: list[int], base_layer: krita.Node):
        base_layer_blending_mode = base_layer.blendingMode()
        base_layer.setBlendingMode("normal")
        self.active_document.refreshProjection()
        self.active_document.waitForDone()

        width = self.active_document.width()
        height = self.active_document.height()

        base_layer_pixels = base_layer.projectionPixelData(0, 0, width, height)
        base_layer.setBlendingMode(base_layer_blending_mode)

        def traverse(node: krita.Node):
            if node.type() == "grouplayer":
                for n in node.childNodes():
                    traverse(n)
            elif node.colorLabel() in allow_labels:
                tmp_document = krita.Krita.instance().createDocument(
                    width,
                    height,
                    "Image",
                    self.active_document.colorModel(),
                    self.active_document.colorDepth(),
                    self.active_document.colorProfile(),
                    self.active_document.resolution()
                )
                foreground_layer = tmp_document.rootNode().childNodes()[0]
                tmp_document.rootNode().removeChildNode(foreground_layer)
                foreground_layer.setInheritAlpha(True)

                base_layer_clone = tmp_document.createNode("base_layer", "paintLayer")
                base_layer_clone.setPixelData(base_layer_pixels, 0, 0, width, height)

                pixels = node.projectionPixelData(0, 0, width, height)
                n = tmp_document.createNode("l", "paintLayer")
                n.setPixelData(pixels, 0, 0, width, height)

                tmp_document.rootNode().addChildNode(base_layer_clone, None)
                tmp_document.rootNode().addChildNode(n, None)
                tmp_document.rootNode().addChildNode(foreground_layer, None)
                tmp_document.refreshProjection()
                tmp_document.waitForDone()

                s = krita.Selection()
                s.select(0, 0, width, height, 255)
                binarize_filter = tmp_document.createFilterMask("binarize_filter_mask",
                                                                krita.Krita.instance().filter("levels"), s)
                n.addChildNode(binarize_filter, None)
                binarize_filter.filter().configuration().setProperties(
                    {'blackvalue': 0, 'channel_0': '0;1;1;0;1', 'channel_1': '0;1;1;0;1', 'channel_2': '0;1;1;0;1',
                     'channel_3': '0;1;1;0;1', 'channel_4': '0;1;1;0;1', 'channel_5': '0;1;1;0;1',
                     'channel_6': '0;1;1;0;1', 'channel_7': '0;1;1;0;1', 'gammavalue': 1.0,
                     'histogram_mode': 'logarithmic', 'lightness': '0;1;1;0;1', 'mode': 'channels',
                     'number_of_channels': 8, 'outblackvalue': 0, 'outwhitevalue': 255, 'whitevalue': 255})
                binarize_filter.filter().configuration().setProperty("channel_4", "0;1;10;0;1")

                s = krita.Selection()
                s.select(0, 0, width, height, 255)
                binarize_filter = tmp_document.createFilterMask("binarize_filter_mask",
                                                                krita.Krita.instance().filter("levels"), s)
                n.addChildNode(binarize_filter, None)
                binarize_filter.filter().configuration().setProperties(
                    {'blackvalue': 0, 'channel_0': '0;1;1;0;1', 'channel_1': '0;1;1;0;1', 'channel_2': '0;1;1;0;1',
                     'channel_3': '0;1;1;0;1', 'channel_4': '0;1;1;0;1', 'channel_5': '0;1;1;0;1',
                     'channel_6': '0;1;1;0;1', 'channel_7': '0;1;1;0;1', 'gammavalue': 1.0,
                     'histogram_mode': 'logarithmic', 'lightness': '0;1;1;0;1', 'mode': 'channels',
                     'number_of_channels': 8, 'outblackvalue': 0, 'outwhitevalue': 255, 'whitevalue': 255})
                binarize_filter.filter().configuration().setProperty("channel_4", "0;1;10;0;1")

                tmp_document.refreshProjection()
                tmp_document.waitForDone()

                pixels = tmp_document.rootNode().projectionPixelData(0, 0, width, height)
                base_layer_clone.setPixelData(pixels, 0, 0, width, height)
                tmp_document.rootNode().removeChildNode(n)
                tmp_document.rootNode().removeChildNode(foreground_layer)
                tmp_document.refreshProjection()
                tmp_document.waitForDone()
                tmp_document.setColorSpace("A", "U8", "")
                tmp_document.refreshProjection()
                tmp_document.waitForDone()

                pixels = tmp_document.rootNode().projectionPixelData(0, 0, width, height)

                transparency_mask = self.active_document.createTransparencyMask("mask")
                transparency_mask.setPixelData(pixels, 0, 0, width, height)
                transparency_mask.setLocked(True)
                node.addChildNode(transparency_mask, None)

        traverse(self.active_document.rootNode())
        self.active_document.refreshProjection()
        self.active_document.waitForDone()
        pass

    def handle_lineart_transfer(self, transfer: bool):
        self.apply_layer_mask_filtered_color_label([LINEART_COLOR_LABEL])

        if transfer:
            active_node = self.active_document.activeNode()
            self.create_transparency_mask_from_layer_filtered_color_label(
                [LINEART_COLOR_LABEL],
                self.active_document.nodeByUniqueID(
                    self.document_nodes_map[self.active_document.rootNode().uniqueId()].lineart))
            self.active_document.setActiveNode(active_node)

    @pyqtSlot(bool)
    def gen_lineart(self):
        self.log("gen_lineart")
        self.spawn_future(self.gen_lineart_inner())

    async def gen_lineart_inner(self):
        if self.active_document is None:
            return

        try:
            self.disable_buttons()
            lineart_output_path = os.path.join(
                self.document_nodes_map[self.active_document.rootNode().uniqueId()].tmp_dir,
                LINEART_FILE_NAME)

            with tempfile.NamedTemporaryFile(suffix=".png") as scribble_file:
                with tempfile.NamedTemporaryFile(suffix=".png") as lineart_file:
                    self.export_image_filtered_color_label([SCRIBBLE_COLOR_LABEL], False, scribble_file.name)
                    self.export_image_filtered_color_label([LINEART_COLOR_LABEL], True, lineart_file.name)

                    await self.diffusion_controller.scribble_to_line(
                        scribble_file.name, lineart_file.name, lineart_output_path)

            self.lineart_transfer_toggle.setChecked(False)
        finally:
            self.enable_buttons()

    def handle_shadow_transfer(self, transfer: bool):
        self.apply_layer_mask_filtered_color_label([SHADOW_COLOR_LABEL])

        if transfer:
            active_node = self.active_document.activeNode()
            self.create_transparency_mask_from_layer_filtered_color_label(
                [SHADOW_COLOR_LABEL],
                self.active_document.nodeByUniqueID(
                    self.document_nodes_map[self.active_document.rootNode().uniqueId()].shadow))
            self.active_document.setActiveNode(active_node)

    @pyqtSlot(bool)
    def gen_detail_colored(self):
        self.log("gen_detail_colored")
        self.spawn_future(self.gen_detail_inner())
        pass

    def handle_light_transfer(self, transfer: bool):
        self.apply_layer_mask_filtered_color_label([LIGHT_COLOR_LABEL])

        if transfer:
            active_node = self.active_document.activeNode()
            self.create_transparency_mask_from_layer_filtered_color_label(
                [LIGHT_COLOR_LABEL],
                self.active_document.nodeByUniqueID(
                    self.document_nodes_map[self.active_document.rootNode().uniqueId()].light))
            self.active_document.setActiveNode(active_node)

    async def gen_detail_inner(self):
        if self.active_document is None:
            return

        try:
            self.disable_buttons()
            shadow_output_path = os.path.join(
                self.document_nodes_map[self.active_document.rootNode().uniqueId()].tmp_dir,
                SHADOW_FILE_NAME)
            light_output_path = os.path.join(
                self.document_nodes_map[self.active_document.rootNode().uniqueId()].tmp_dir,
                LIGHT_FILE_NAME)

            with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
                with tempfile.NamedTemporaryFile(suffix=".png") as basecolor_image_file:
                    with tempfile.NamedTemporaryFile(suffix=".png") as lineart_file:
                        with tempfile.NamedTemporaryFile(suffix=".png") as basecolor_file:
                            with tempfile.NamedTemporaryFile(suffix=".png") as shadow_file:
                                with tempfile.NamedTemporaryFile(suffix=".png") as light_file:
                                    self.export_image_filtered_color_label(
                                        [LINEART_COLOR_LABEL, BASE_COLOR_COLOR_LABEL,
                                         SHADOW_COLOR_LABEL, LIGHT_COLOR_LABEL],
                                        False, image_file.name)
                                    self.export_image_filtered_color_label(
                                        [LINEART_COLOR_LABEL, BASE_COLOR_COLOR_LABEL],
                                        False, basecolor_image_file.name)
                                    self.export_image_filtered_color_label([LINEART_COLOR_LABEL], True,
                                                                           lineart_file.name)
                                    self.export_image_filtered_color_label(
                                        [BASE_COLOR_COLOR_LABEL], True, basecolor_file.name)
                                    self.export_image_filtered_color_label([SHADOW_COLOR_LABEL], True, shadow_file.name)
                                    self.export_image_filtered_color_label([LIGHT_COLOR_LABEL], True, light_file.name)

                                    await self.diffusion_controller.detail_colored(
                                        image_file.name,
                                        basecolor_image_file.name,
                                        lineart_file.name,
                                        basecolor_file.name,
                                        shadow_file.name,
                                        light_file.name,
                                        shadow_output_path,
                                        light_output_path)

            self.shadow_transfer_toggle.setChecked(False)
            self.light_transfer_toggle.setChecked(False)
        finally:
            self.enable_buttons()

    @pyqtSlot(bool)
    def initialize_document(self):
        self.log("initialize_document")
        document_id = self.active_document.rootNode().uniqueId()
        if document_id not in self.document_nodes_map:
            self.log("Open unknown document")
            empty_image = os.path.join(os.path.dirname(__file__), EMBEDDED_EMPTY_IMAGE_FILE_NAME)
            tmp_dir = tempfile.mkdtemp(prefix="diffusion_drawing_")

            shutil.copy(empty_image, os.path.join(tmp_dir, LINEART_FILE_NAME))
            shutil.copy(empty_image, os.path.join(tmp_dir, SHADOW_FILE_NAME))
            shutil.copy(empty_image, os.path.join(tmp_dir, LIGHT_FILE_NAME))

            self.document_nodes_map[document_id] = SystemLayers(tmp_dir)

        system_layers = self.document_nodes_map[document_id]
        rootNode = self.active_document.rootNode()
        for n in filter(lambda n: n.name().find(DIFFUSION_DRAWING_LAYER_MARKER) != -1, rootNode.childNodes()):
            rootNode.removeChildNode(n)

        lineart_layer = self.active_document.createFileLayer(
            LINEART_LAYER_NAME,
            os.path.join(str(system_layers.tmp_dir), LINEART_FILE_NAME),
            "ToImageSize")
        lineart_layer.setOpacity(SYSTEM_LAYER_DEFAULT_OPACITY)
        self.active_document.rootNode().addChildNode(lineart_layer, None)
        lineart_layer.setLocked(True)
        system_layers.lineart = lineart_layer.uniqueId()

        shadow_layer = self.active_document.createFileLayer(
            SHADOW_LAYER_NAME,
            os.path.join(str(system_layers.tmp_dir), SHADOW_FILE_NAME),
            "ToImageSize")
        shadow_layer.setOpacity(SYSTEM_LAYER_DEFAULT_OPACITY)
        shadow_layer.setBlendingMode("multiply")
        self.active_document.rootNode().addChildNode(shadow_layer, None)
        shadow_layer.setLocked(True)
        system_layers.shadow = shadow_layer.uniqueId()

        light_layer = self.active_document.createFileLayer(
            LIGHT_LAYER_NAME,
            os.path.join(str(system_layers.tmp_dir), LIGHT_FILE_NAME),
            "ToImageSize")
        light_layer.setOpacity(SYSTEM_LAYER_DEFAULT_OPACITY)
        light_layer.setBlendingMode("add")
        self.active_document.rootNode().addChildNode(light_layer, None)
        light_layer.setLocked(True)
        system_layers.light = light_layer.uniqueId()

        self.setup_area_ready()

    def spawn_future(self, coro: Coroutine):
        async def wrapper():
            try:
                await coro
            except Exception as e:
                self.log(e)
                import traceback
                traceback_str = traceback.format_exc()
                self.log(traceback_str)

        asyncio.ensure_future(wrapper())

    # メインで開いているドキュメントが変わったときには呼ばれるらしい
    # @override
    def canvasChanged(self, canvas: krita.Canvas) -> None:
        active_document = krita.Krita.instance().activeDocument()

        if active_document == self.active_document:
            return

        self.active_document = active_document

        if self.active_document is None:
            self.setup_area_none()
        elif self.active_document.rootNode().uniqueId() in self.document_nodes_map:
            self.setup_area_ready()
        else:
            self.setup_area_initialize()

    def log(self, message: any) -> None:
        self.log_window.append(str(message))
