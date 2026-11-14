"""Audit wheel and sdist before making public FXBEX releases.

This check catches accidental large model/compiled artifacts or local credentials.
It does not replace a manual security review.
"""
from pathlib import Path
import sys
import tarfile
import zipfile

FORBIDDEN_SUFFIXES = (
    ".fxbx", ".fxbex", ".fxb", ".runnable", ".edf", ".safetensors",
    ".pt", ".bin", ".log", ".so", ".dylib", ".dll", ".pyc",
)
FORBIDDEN_NAMES = {".env", ".pypirc", "id_rsa", "id_ed25519"}


def audit(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names), (
            f"{path}: wheel is missing Apache-2.0 LICENSE"
        )
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
        assert any(name.endswith("/LICENSE") for name in names), (
            f"{path}: sdist is missing LICENSE"
        )
    else:
        raise ValueError(f"Unexpected distribution format: {path}")

    assert any(name.endswith("/fxbex/cli.py") or name == "fxbex/cli.py" for name in names), (
        f"{path}: CLI module is missing"
    )
    assert any(name.endswith('/fxbx/__init__.py') or name == 'fxbx/__init__.py' for name in names), (
        f'{path}: public fxbx module is missing'
    )
    producer_modules = {'model_export.py', 'static_graph_contract.py', 'provenance.py', 'capture.py'}
    for entry in names:
        name = Path(entry).name
        parts = Path(entry).parts
        if 'tplmachine' in parts or 'fxb_export' in parts or ('fxbex' in parts and name in producer_modules):
            raise ValueError(f"{path}: private authoring module: {entry}")
        if name in FORBIDDEN_NAMES or name.startswith(".env.") or entry.endswith(FORBIDDEN_SUFFIXES):
            raise ValueError(f"{path}: forbidden packaged file: {entry}")

    print(f"OK: {path} ({len(names)} archive entries)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/check_dist.py dist/*")
    for item in sys.argv[1:]:
        audit(Path(item))
