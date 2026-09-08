"""Bundle a host-compatible lineartgen wheel into a Krita plugin ZIP."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def write_plugin_zip(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(staging.rglob("*")):
            # Krita discovers plugins through explicit directory entries.
            # Files below diffusion_drawing/ alone are not enough.
            archive.write(path, path.relative_to(staging).as_posix())


def package(wheel_dir: Path, output: Path, provider: str = "cpu") -> None:
    wheels = sorted(wheel_dir.glob("lineartgen_runtime-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one lineartgen-runtime wheel in {wheel_dir}, found {len(wheels)}")
    if provider == "directml" and sys.platform != "win32":
        raise ValueError("Build DirectML packages on Windows using Krita's Python minor version")
    controller = ROOT / "diffusion_drawing" / "diffusion_controller" / "__init__.py"
    if not controller.is_file():
        raise ValueError("Initialize the submodules first: git submodule update --init --recursive")

    with tempfile.TemporaryDirectory(prefix="diffusion-drawing-package-") as temporary:
        staging = Path(temporary)
        plugin = staging / "diffusion_drawing"
        shutil.copytree(
            ROOT / "diffusion_drawing", plugin,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo", "_native"),
        )
        native = plugin / "_native"
        native.mkdir()
        shutil.copyfile(ROOT / "diffusion_drawing" / "_native" / "__init__.py", native / "__init__.py")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-compile", "--no-index", "--find-links", str(wheel_dir.resolve()),
            "--target", str(native), f"{wheels[0].resolve()}[{provider}]",
        ], check=True)
        (native / "runtime.json").write_text(json.dumps({
            "python": list(sys.version_info[:2]), "platform": sys.platform, "provider": provider,
        }, indent=2) + "\n")
        shutil.copyfile(ROOT / "diffusion_drawing.desktop", staging / "diffusion_drawing.desktop")
        shutil.copyfile(ROOT / "LICENSE", plugin / "LICENSE")

        # Use another process so Windows releases the ORT .pyd before cleanup.
        subprocess.run([
            sys.executable, "-B", str(ROOT / "scripts" / "smoke_native.py"), str(plugin), provider,
        ], check=True)
        write_plugin_zip(staging, output)
    print(f"Created {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "wheels")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "diffusion_drawing-windows-x64.zip")
    parser.add_argument("--provider", choices=("cpu", "directml"), default="cpu")
    args = parser.parse_args()
    package(args.wheel_dir, args.output, args.provider)
