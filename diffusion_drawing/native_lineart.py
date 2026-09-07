"""In-memory lineart inference; Krita objects stay on the UI thread."""

import asyncio
import importlib
import threading


SRGB_PROFILE = "sRGB-elle-V2-srgbtrc.icc"


def capture_bgra(document, color_label: int) -> bytes:
    """Read a filtered, RGBA/U8 copy without changing the user's document."""
    def hide_other_layers(node):
        for child in node.childNodes():
            if child.type() == "grouplayer":
                hide_other_layers(child)
            elif child.type().endswith("mask"):
                # Preserve masks on included layers and groups.
                continue
            elif child.colorLabel() != color_label:
                child.setVisible(False)

    snapshot = document.clone()
    try:
        snapshot.setBatchmode(True)
        hide_other_layers(snapshot.rootNode())
        if not snapshot.setColorSpace("RGBA", "U8", SRGB_PROFILE):
            raise RuntimeError("Could not convert the input image to 8-bit sRGB.")
        snapshot.refreshProjection()
        snapshot.waitForDone()
        width, height = snapshot.width(), snapshot.height()
        pixels = bytes(snapshot.rootNode().projectionPixelData(0, 0, width, height))
        if len(pixels) != width * height * 4:
            raise RuntimeError("Krita returned an incomplete image buffer.")
        return pixels
    finally:
        snapshot.close()


class NativeLineart:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    @property
    def backend_description(self) -> str | None:
        if self._model is None:
            return None
        return f"{self._model.backend}: {self._model.device}"

    async def infer(self, *, scribble: bytes, lineart: bytes, width: int, height: int,
                    strength: float, seed: int, denoise_steps: int) -> bytes:
        return await asyncio.to_thread(
            self._infer, scribble=scribble, lineart=lineart, width=width, height=height,
            strength=strength, seed=seed, denoise_steps=denoise_steps,
        )

    def _infer(self, **inputs) -> bytes:
        with self._lock:
            if self._model is None:
                try:
                    native = importlib.import_module("._native.lineartgen_native", __package__)
                except (ImportError, OSError) as error:
                    raise RuntimeError(
                        "Could not load lineart generation. Install the Diffusion Drawing ZIP "
                        "for your operating system and CPU architecture (Python 3.10 or later)."
                    ) from error
                self._model = native.LineartModel()
            return self._model.infer(**inputs)
