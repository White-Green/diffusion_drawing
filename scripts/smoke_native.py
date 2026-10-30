"""Exercise the complete packaged ONNX runtime without importing Krita or PyQt."""

import argparse
import asyncio
import importlib
import hashlib
import json
from pathlib import Path
import sys
import types


async def smoke(plugin: Path, provider: str = "cpu", *, allow_cpu_fallback: bool = False) -> None:
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
    assert "graph" in model.backend_description, "This preview must contain the spatial stage graph model"
    if provider == "directml":
        # CPU fallback must not hide accidentally shipping the CPU-only ORT wheel.
        available = sys.modules["onnxruntime"].get_available_providers()
        assert "DmlExecutionProvider" in available, "The bundled runtime is missing DirectML"
        if "DmlExecutionProvider" not in model.backend_description:
            assert allow_cpu_fallback, "DirectML inference was required but fell back to CPU"
            assert "CPUExecutionProvider" in model.backend_description
            print(
                "DirectML is bundled, but this host could not initialize it; "
                "testing CPU fallback. GPU execution is not validated by this run.",
                flush=True,
            )
    assert "jax" not in sys.modules and "flax" not in sys.modules
    for dependency in ("numpy", "onnxruntime", "lineartgen_runtime"):
        assert Path(sys.modules[dependency].__file__).is_relative_to(plugin / "_native"), dependency
    # Exercise every exported stage against saved JAX reference values on the
    # actual bundled Windows runtime, including DirectML when available.
    import numpy as np
    from lineartgen_runtime.stage_feedback import forward
    assets = Path(sys.modules["lineartgen_runtime"].__file__).parent / "assets"
    reference = json.loads((assets / "spatial-stage-reference.json").read_text())
    spatial = model._model._spatial
    assert spatial.metadata["representation"] == "spatial_intent_stage_graph"
    assert reference["checkpoint_sha256"] == spatial.metadata["checkpoint_sha256"]
    assert reference["onnx_sha256"] == spatial.metadata["stage"]["sha256"]
    assert spatial.metadata["training_steps"] == 48000
    assert spatial.metadata["checkpoint_sha256"] == (
        "bc8ea33a594740eab7834b776633e9f7d8eda8270033038aa71293f03ea54ab2"
    )
    arrays = assets / "spatial-stage-reference.npz"
    assert hashlib.sha256(arrays.read_bytes()).hexdigest() == reference["arrays_sha256"]
    with np.load(arrays, allow_pickle=False) as z:
        h, w = z["scribble"].shape[:2]
        for depth in reference["depths"]:
            heads = forward(spatial._get_session(), tuple(z[f"image{depth}_{i}"] for i in range(depth)))
            for level, head in enumerate(heads):
                np.testing.assert_allclose(head, z[f"head{depth}_{level}"], **reference["tolerance"])
            native = await model.infer(
                scribble=z["scribble"].tobytes(), lineart=z["lineart"].tobytes(),
                width=w, height=h, strength=reference["bgra_strength"],
                seed=reference["seed"], denoise_steps=depth,
            )
            actual = np.frombuffer(native, np.uint8).reshape(h, w, 4)
            np.testing.assert_allclose(actual, z[f"bgra{depth}"], atol=reference["bgra_atol"], rtol=0)
            protected = z["lineart"][..., 3] == 255
            np.testing.assert_array_equal(actual[protected], z[f"bgra{depth}"][protected])
    print("All recursion depths 1-6 match independent CPU JAX references", flush=True)
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
    lineart = bytearray(width * height * 4)
    for y in range(height):
        lineart[(y * width + width // 2) * 4 + 3] = 255
        lineart[(y * width + width // 2 + 1) * 4 + 3] = 128
    recursive_options = options | dict(
        scribble=bytes(width * height * 4),
        lineart=bytes(lineart), width=width, height=height, denoise_steps=3,
    )
    recursive = await model.infer(**recursive_options)
    assert len(recursive) == width * height * 4
    assert recursive == await model.infer(**recursive_options)
    # Fully opaque user lineart stays untouched, including white protection.
    for y in range(height):
        assert recursive[(y * width + width // 2) * 4 + 3] == 255
    white = bytes([255, 255, 255, 255]) * (width * height)
    protected = await model.infer(**(recursive_options | dict(lineart=white)))
    assert not any(protected), "Opaque white protection must remain transparent output"
    for depth in (1, 6):
        assert len(await model.infer(**(options | dict(denoise_steps=depth)))) == len(first)
    blank = await model.infer(**(options | dict(scribble=bytes(33 * 17 * 4))))
    assert not any(blank), "Blank inputs must not create strokes"
    for invalid in [dict(strength=float("nan")), dict(denoise_steps=0), dict(denoise_steps=7), dict(scribble=b"")]:
        try:
            await model.infer(**(options | invalid))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input was accepted: {invalid}")
    print("Bundled ONNX model: import, inference, alpha-only scribble, padding, recursion, fixed seed and validation passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", type=Path)
    parser.add_argument("provider", choices=("cpu", "directml"), nargs="?", default="cpu")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()
    asyncio.run(smoke(
        args.plugin.resolve(), args.provider, allow_cpu_fallback=args.allow_cpu_fallback,
    ))
