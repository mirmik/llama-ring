#!/usr/bin/env python3
"""Export only reviewed source files, independent of Git tracking/ignore state."""
import argparse
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def release_files(root=ROOT):
    paths = []
    for line in (root / "RELEASE_FILES.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path = Path(line)
        if path.is_absolute() or ".." in path.parts or str(path) != line:
            raise ValueError(f"Invalid release path: {line}")
        source = root / path
        if not source.is_file() or source.resolve() != root.resolve() / path:
            raise ValueError(f"Missing file or symlink in release: {line}")
        paths.append(path)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Empty or duplicate release manifest")
    return paths


def export(output, root=ROOT):
    paths = release_files(root)
    archive = output.with_name(output.name + ".tar.gz")
    if output.exists() or archive.exists():
        raise FileExistsError("Choose a new output path; exports are never overwritten")
    output.mkdir(parents=True)
    for path in paths:
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, target)
    with tarfile.open(archive, "w:gz") as tar:
        for path in paths:
            info = tar.gettarinfo(str(output / path), arcname=str(Path(output.name) / path))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with (output / path).open("rb") as stream:
                tar.addfile(info, stream)
    return archive, len(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive, count = export(args.output.resolve())
    print(f"Exported {count} files to {args.output} and {archive}")
