#!/usr/bin/env python3
"""Run the already built stock/ring speed probes and upstream llama-bench."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def aggregate(out):
    paired = {name: json.loads((out / f'{name}.json').read_text()) for name in ('stock', 'ring')}
    result = {'paired': {}, 'comparisons': []}
    for name, data in paired.items():
        assert data['completed']
        result['paired'][name] = []
        for case in data['cases']:
            samples = case['samples']
            assert len(samples) == 3 and all(s['seconds'] > 0 for s in samples)
            speeds = [s['tokens_s'] for s in samples]
            result['paired'][name].append({
                'mode': case['mode'], 'depth': case['depth'], 'tokens': case['tokens'],
                'median_tokens_s': statistics.median(speeds),
                'min_tokens_s': min(speeds), 'max_tokens_s': max(speeds),
            })
    for stock, ring in zip(result['paired']['stock'], result['paired']['ring']):
        assert (stock['mode'], stock['depth']) == (ring['mode'], ring['depth'])
        result['comparisons'].append({
            'mode': stock['mode'], 'depth': stock['depth'],
            'ring_time_overhead_pct': 100*(stock['median_tokens_s']/ring['median_tokens_s']-1),
        })
    result['paired_generation_tokens_equal'] = all(
        [s['generated'] for s in a['samples']] == [s['generated'] for s in b['samples']]
        for a, b in zip(paired['stock']['cases'], paired['ring']['cases']) if a['mode'] == 'tg'
    )
    bench = json.loads((out / 'llama-bench.json').read_text())
    assert len(bench) == 4
    result['llama_bench'] = [{
        'n_prompt': b['n_prompt'], 'n_gen': b['n_gen'], 'depth': b['n_depth'],
        'mean_tokens_s': b['avg_ts'], 'stddev_tokens_s': b['stddev_ts'],
        'median_tokens_s': statistics.median(b['samples_ts']),
    } for b in bench]
    (out / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.aggregate_only:
        print(json.dumps(aggregate(out), indent=2))
        return
    if not args.model or not args.model.is_file():
        parser.error('--model must point to a GGUF file')
    root = Path(__file__).resolve().parents[1]
    stock = root / 'build-stock/bin'
    ring = root / 'build/bin'
    model = str(args.model.resolve())
    out.mkdir(parents=True, exist_ok=True)
    commands = [
        ('stock', [str(stock/'stock-speed'), model, str(out/'stock.json')]),
        ('ring', [str(ring/'ring-speed'), model, str(out/'ring.json')]),
        ('llama-bench', [str(stock/'llama-bench'), '-m', model, '-p', '1024', '-n', '128',
                         '-d', '0,8259', '-b', '512', '-ub', '512', '-t', '8',
                         '-ngl', '99', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16',
                         '-r', '3', '-o', 'json']),
    ]
    assert all(Path(cmd[0]).is_file() for _, cmd in commands), 'Build all three executables first'
    manifest = {'commands': commands, 'binary_sha256': {}}
    for folder in (stock, ring):
        for f in folder.iterdir():
            if f.is_file() and not f.is_symlink() and ('.so' in f.name or f.name in ('stock-speed', 'ring-speed', 'llama-bench')):
                manifest['binary_sha256'][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'commands.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env.pop('VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS', None)
    env.pop('VCORTEX_QWEN35_TEXT_SHIFT', None)
    for name, cmd in commands:
        print(f'Running {name}', flush=True)
        with (out/f'{name}.log').open('w') as err, (out/f'{name}.stdout').open('w') as stdout:
            subprocess.run(cmd, stdout=stdout, stderr=err, env=env, check=True)
        if name == 'llama-bench':
            (out/'llama-bench.json').write_text((out/'llama-bench.stdout').read_text())
    print(json.dumps(aggregate(out), indent=2))


if __name__ == '__main__':
    main()
