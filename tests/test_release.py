"""The source export must be complete and must exclude incidental local data."""
import importlib.util
from pathlib import Path
import re
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_export", ROOT / "scripts/export.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_manifest_covers_runtime_and_export(tmp_path):
    paths = set(release.release_files())
    assert {Path(p) for p in ("LICENSE", "THIRD_PARTY_NOTICES.md", "deps.json",
                             "README.md", "requirements-server.txt")} <= paths
    for directory in ("src", "patches", "tests", ".github"):
        required = {p.relative_to(ROOT) for p in (ROOT / directory).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts}
        assert required <= paths
    output = tmp_path / "llama-ring"
    archive, count = release.export(output)
    assert count == len(paths)
    for path in paths:
        assert (output / path).read_bytes() == (ROOT / path).read_bytes()
    with tarfile.open(archive) as tar:
        assert {p.name for p in tar.getmembers()} == {f"llama-ring/{p}" for p in paths}
    with pytest.raises(FileExistsError):
        release.export(output)


def test_unlisted_data_is_not_exported(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "RELEASE_FILES.txt").write_text("public.txt\n")
    (source / "public.txt").write_text("public")
    (source / ".env").write_text("PRIVATE=do-not-export")
    (source / "conversation.json").write_text("private conversation")
    output = tmp_path / "release"
    archive, _ = release.export(output, source)
    assert {p.name for p in output.iterdir()} == {"public.txt"}
    with tarfile.open(archive) as tar:
        assert tar.getnames() == ["release/public.txt"]


def test_documentation_links_stay_inside_export():
    paths = set(release.release_files())
    for path in paths:
        if path.suffix != ".md":
            continue
        for target in re.findall(r'\]\(([^)]+)\)', (ROOT / path).read_text()):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (ROOT / path.parent / target.split("#")[0]).resolve()
            assert resolved.is_relative_to(ROOT), (path, target)
            assert resolved.relative_to(ROOT) in paths, (path, target)
