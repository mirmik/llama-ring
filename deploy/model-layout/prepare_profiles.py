#!/usr/bin/env python3
"""Run on the Nemor host once; clone credentials locally without changing characters."""
import argparse
import json
import os
from pathlib import Path


PROFILES = {
    'model-flash-ring': ('model-atom-flash-test', 'https://192.168.0.61:8090',
                         'qwen3.8-flash-next-iq4-xs', 'qwen3.8-flash-next-iq4-xs-ring'),
    'model-27b-ring': ('model-atom', 'https://192.168.0.173:8080',
                      'qwen3.8-27b-uncensored-q4-mtp', 'qwen3.8-27b-uncensored-q4-mtp-ring'),
}


def prepare(directory):
    base = Path(directory).expanduser() / 'model_profiles'
    pending = []
    for target, (source, server, native, model) in PROFILES.items():
        path = base / (target + '.json')
        if path.exists():
            existing = json.loads(path.read_text())
            if any(existing.get(k) != v for k, v in {'id': target, 'server': server, 'model': model}.items()):
                raise ValueError('Existing profile differs; refusing overwrite: ' + target)
            continue
        data = json.loads((base / (source + '.json')).read_text())
        if data.get('server') != server or data.get('model') != native:
            raise ValueError('Unexpected source profile: ' + source)
        data.update(id=target, name=model + ' · ' + server, model=model)
        pending.append((path, data))
    for path, data in pending:
        # Never replace an existing profile or follow a concurrently created symlink.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        print('Created ' + path.stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='~/.nemor')
    prepare(parser.parse_args().directory)
