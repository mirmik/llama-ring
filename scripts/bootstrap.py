#!/usr/bin/env python3
"""Extract a pinned upstream tree into .deps, then apply the local patch."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / 'deps.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, help='Existing llama.cpp Git repository (read-only)')
parser.add_argument('--destination', type=Path, default=root / '.deps/llama.cpp',
                    help='Fresh destination directory (default: .deps/llama.cpp)')
args = parser.parse_args()
destination = args.destination.resolve()
if destination.exists():
    raise SystemExit(f'{destination} already exists; use a fresh directory to avoid overwriting changes')
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='vcortex-llama-') as tmp:
    tmp = Path(tmp)
    source = args.source
    if source is None:
        source = tmp / 'upstream'
        subprocess.run(['git', 'init', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'fetch', '--depth=1', lock['repository'], lock['revision']], check=True)
    archive = tmp / 'source.tar'
    subprocess.run(['git', '-C', str(source), 'archive', lock['revision'], '-o', str(archive)], check=True)
    destination.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter='data')
    for patch in lock['patches']:
        subprocess.run(['git', 'apply', '--check', str(root / patch)], cwd=destination, check=True)
        subprocess.run(['git', 'apply', str(root / patch)], cwd=destination, check=True)
(destination / 'vcortex-revision.json').write_text(json.dumps(lock, indent=2) + '\n')
print(destination)
