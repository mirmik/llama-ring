#!/usr/bin/env python3
"""Validate the fixed 24-record fixture (029/030) and aggregate timing/recall."""
import argparse
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
fixture = json.loads((args.run / 'fixture.json').read_text())
checks = json.loads((args.run / 'checks.json').read_text())
assert checks['passed']
results = {}
for path in sorted(args.run.glob('*.json')):
    if not re.fullmatch(r'(cached|fresh|none)(-\d+)?', path.stem):
        continue
    data = json.loads(path.read_text())
    assert data['completed'], path
    assert len(data['blocks']) == 24 and len(data['events']) == 16 and len(data['probes']) == 32
    assert data['active_tokens'] == 8192 and data['prefix_tokens'] <= 4096
    assert all(b['tokens'] == 1024 for b in data['blocks'])
    assert all(e['count'] == 1024 and e['evicted_start'] == 1024*i for i, e in enumerate(data['events']))
    assert all(p['eos'] for p in data['probes'])
    locations = {name: {'correct': 0, 'total': 0} for name in ('active_only', 'prefix_only', 'both')}
    for probe in data['probes']:
        code = probe['question']['expected']
        i = probe['block']
        active = any(code in b['text'] for b in fixture['blocks'][max(0, i-7):i+1])
        evicted = any(code in b['text'] for b in fixture['blocks'][:max(0, i-7)])
        assert active or evicted
        loc = 'both' if active and evicted else 'active_only' if active else 'prefix_only'
        locations[loc]['total'] += 1
        locations[loc]['correct'] += probe['correct']
    row = {'mode': data['mode'], 'elapsed_s': data['elapsed_s'], 'prefix_tokens': data['prefix_tokens'],
           'correct': sum(p['correct'] for p in data['probes']), 'questions': 32, 'locations': locations}
    row['stream_append_s'] = sum(b['append_s_including_pending_shift'] for b in data['blocks'])
    row['stream_tokens_s'] = 24576 / row['stream_append_s']
    row['prefix_append_s'] = sum(e['prefix_append_s'] for e in data['events'])
    row['evict_metadata_s'] = sum(e['evict_metadata_s'] for e in data['events'])
    row['summary_memory_append_s'] = sum(e['summary_memory_append_s'] for e in data['events'])
    row['probe_decode_s'] = sum(p['decode_s'] for p in data['probes'])
    # The first generated token comes from prefill; EOS follows the last decode.
    row['probe_decode_steps'] = sum(len(p['tokens']) - 1 for p in data['probes'])
    row['probe_decode_steps_s'] = row['probe_decode_steps'] / row['probe_decode_s']
    if data['mode'] != 'none':
        summaries = [e['summary'] for e in data['events']]
        assert all(s['eos'] and len(s['tokens']) <= 256 for s in summaries)
        row['summary_prefill_tokens'] = sum(s['prefill_tokens'] for s in summaries)
        row['summary_prefill_s'] = sum(s['prefill_s'] for s in summaries)
        row['summary_decode_s'] = sum(s['decode_s'] for s in summaries)
        row['summary_s'] = row['summary_prefill_s'] + row['summary_decode_s']
        row['summary_with_append_s'] = row['summary_s'] + row['summary_memory_append_s']
        row['summary_empty'] = sum(s['text'] == 'NONE' for s in summaries)
        seen, duplicates, unsupported = set(), [], []
        for e in data['events']:
            for code in re.findall(r'\b\d{4}\b', e['summary']['text']):
                if code in seen:
                    duplicates.append(code)
                elif code not in e['discarded_text']:
                    unsupported.append(code)
                seen.add(code)
        row.update(summary_unique_codes=len(seen), duplicate_codes=duplicates, unsupported_codes=unsupported)
    results[path.stem] = row
grouped = {}
for mode in ('cached', 'fresh', 'none'):
    rows = [x for x in results.values() if x['mode'] == mode]
    if rows:
        grouped[mode] = {'runs': len(rows)}
        for key in ('elapsed_s', 'stream_tokens_s', 'probe_decode_steps_s', 'summary_with_append_s'):
            values = [x[key] for x in rows if key in x]
            if values:
                grouped[mode][key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
out = {'runs': results, 'grouped': grouped}
(args.run / 'aggregate.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(grouped, ensure_ascii=False, indent=2))
