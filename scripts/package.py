"""Bundle a host-compatible lineartgen wheel into a Krita plugin ZIP."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def package(wheel_dir: Path, output: Path) -> None:
    wheels = sorted(wheel_dir.glob("lineartgen_native-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one lineartgen-native wheel in {wheel_dir}, found {len(wheels)}")
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
            "--no-deps", "--no-compile", "--no-index", "--target", str(native), str(wheels[0].resolve()),
        ], check=True)
        shutil.copyfile(ROOT / "diffusion_drawing.desktop", staging / "diffusion_drawing.desktop")
        shutil.copyfile(ROOT / "LICENSE", plugin / "LICENSE")

        # Use another process so Windows releases the .pyd before staging cleanup.
        subprocess.run([
            sys.executable, "-B", str(ROOT / "scripts" / "smoke_native.py"), str(plugin),
        ], check=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
    print(f"Created {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "wheels")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "diffusion_drawing-windows-x64.zip")
    args = parser.parse_args()
    package(args.wheel_dir, args.output)
