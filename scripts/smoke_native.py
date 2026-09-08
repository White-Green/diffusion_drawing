"""Exercise the complete packaged ONNX runtime without importing Krita or PyQt."""

import asyncio
import importlib
from pathlib import Path
import sys
import types


async def smoke(plugin: Path, provider: str = "cpu") -> None:
    package_name = "_diffusion_drawing_smoke"
    package = types.ModuleType(package_name)
    package.__path__ = [str(plugin.resolve())]
    sys.modules[package_name] = package
    integration = importlib.import_module(f"{package_name}.native_lineart")
    model = integration.NativeLineart()
    width, height = 33, 17
    scribble = bytearray(width * height * 4)
    for y in range(height):
        scribble[(y * width + width // 2) * 4 + 3] = (64, 128, 255)[y % 3]
    options = dict(
        scribble=bytes(scribble),
        lineart=bytes(width * height * 4), width=width, height=height,
        strength=0.5, seed=42, denoise_steps=2,
    )
    first = await model.infer(**options)
    print(f"Bundled model backend: {model.backend_description}", flush=True)
    assert "onnxruntime" in model.backend_description
    if provider == "directml":
        assert "DmlExecutionProvider" in model.backend_description
    assert "jax" not in sys.modules and "flax" not in sys.modules
    for dependency in ("numpy", "onnxruntime", "lineartgen_runtime"):
        assert Path(sys.modules[dependency].__file__).is_relative_to(plugin / "_native"), dependency
    repeated = await model.infer(**options)
    assert isinstance(first, bytes)
    assert len(first) == width * height * 4
    assert first == repeated, "A fixed seed must be reproducible"
    assert all(first[index:index + 3] == bytes(3) for index in range(0, len(first), 4))
    for index in range(0, len(scribble), 4):
        scribble[index:index + 3] = bytes([255, 80, 190])
    recolored = await model.infer(**(options | dict(scribble=bytes(scribble))))
    assert recolored == first, "Scribble RGB must not affect inference when alpha is unchanged"
    # Exercise recursive inference too, rather than only the smallest single-level input.
    width, height = 145, 129
    recursive = await model.infer(**(options | dict(
        scribble=bytes(width * height * 4),
        lineart=bytes(width * height * 4), width=width, height=height, denoise_steps=1,
    )))
    assert len(recursive) == width * height * 4
    for invalid in [dict(strength=float("nan")), dict(denoise_steps=0), dict(scribble=b"")]:
        try:
            await model.infer(**(options | invalid))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input was accepted: {invalid}")
    print("Bundled ONNX model: import, inference, alpha-only scribble, padding, recursion, fixed seed and validation passed")


if __name__ == "__main__":
    asyncio.run(smoke(Path(sys.argv[1]).resolve(), sys.argv[2] if len(sys.argv) > 2 else "cpu"))
