"""Install a plugin ZIP with Krita's actual importer, without starting the GUI."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import types
from urllib.request import urlopen


# Krita 5.2.9, pinned by commit and content digest so upstream changes cannot
# silently change the code executed by this check. The source is GPL-3.0-or-later.
IMPORTER_URL = (
    "https://raw.githubusercontent.com/KDE/krita/"
    "ac6cae86d02ce033e81f20f70fdda10aa4fd8dfb/"
    "plugins/python/plugin_importer/plugin_importer.py"
)
IMPORTER_SHA256 = "cde7508cbce888f2516b92ada053d1f65cb096660cd4cbd845d44053af46c76f"


def check_import(archive_path: Path, importer_source: Path | None = None) -> None:
    if importer_source is None:
        with urlopen(IMPORTER_URL, timeout=60) as response:
            source = response.read()
    else:
        source = importer_source.read_bytes()
    if hashlib.sha256(source).hexdigest() != IMPORTER_SHA256:
        raise ValueError("Krita importer source does not match the pinned SHA-256")

    module = types.ModuleType("krita_plugin_importer")
    # Krita normally supplies this translation function to its Python plugins.
    module.i18n = lambda message: message
    exec(compile(source, IMPORTER_URL, "exec"), module.__dict__)

    with tempfile.TemporaryDirectory(prefix="krita-import-check-") as resources:
        importer = module.PluginImporter(str(archive_path), resources, lambda plugin: True)
        try:
            corrupt_file = importer.archive.testzip()
            if corrupt_file is not None:
                raise ValueError(f"ZIP checksum failed: {corrupt_file}")
            imported = importer.import_all()
            if [plugin["name"] for plugin in imported] != ["diffusion_drawing"]:
                raise ValueError(f"Unexpected imported plugins: {imported}")
            installed = Path(resources) / "pykrita"
            file_count = 0
            for entry in importer.archive.infolist():
                if entry.is_dir():
                    continue
                destination = installed / entry.filename
                if destination.read_bytes() != importer.archive.read(entry):
                    raise ValueError(f"Installed file differs from ZIP: {entry.filename}")
                file_count += 1
        finally:
            importer.archive.close()
    print(f"Krita 5.2.9 imported diffusion_drawing; all {file_count} files match the ZIP")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--importer-source", type=Path, help="Use a local copy of the pinned importer")
    args = parser.parse_args()
    check_import(args.archive, args.importer_source)
