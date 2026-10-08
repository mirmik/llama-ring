#!/usr/bin/env python3
"""Validate and summarize experiment 032's two 24-record shadow schedules."""
import argparse
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run
    for name in ['checks-4b', 'checks-27b']:
        assert json.loads((root/f'{name}.json').read_text())['passed']
    results = {}
    for mode in ['busy', 'idle']:
        d = json.loads((root/f'{mode}.json').read_text())
        fixture = json.loads((root/f'{mode}-fixture.json').read_text())
        assert d['completed'] and len(d['blocks']) == 24 and len(d['probes']) == 32
        assert all(b['tokens'] == 1024 and b['active_tokens'] <= 8192 for b in d['blocks'])
        assert len(d['events']) == 3 and d['active_tokens'] == 24576 - 3*5735
        assert d['prefix_tokens'] <= 4096 and d['pending_phase'] != 'failed'
        assert all(p['eos'] for p in d['probes'])
        assert all(e['cut'] == 5735 and e['eos'] and len(e['summary_tokens']) <= 512 for e in d['events'])
        assert all(e['active_after'] == 2457 for e in d['events'])
        all_codes = {q['expected'] for b in fixture['blocks'] for q in b.get('questions', [])}
        summaries = []
        for i, event in enumerate(d['events']):
            # Each original fact is at the beginning of its fixed 1024-token record.
            # The cuts lie hundreds of tokens into weather padding, after that fact.
            covered_records = ((i+1)*5735 + 1023)//1024
            expected = set(re.findall(r'\b\d{4}\b', '\n'.join(b['text'] for b in fixture['blocks'][:covered_records])))
            found = re.findall(r'\b\d{4}\b', event['summary'])
            summaries.append({'generation': event['generation'], 'expected_codes': sorted(expected),
                              'found_codes': found, 'missing': sorted(expected-set(found)),
                              'unsupported': sorted(set(found)-expected),
                              'duplicates': sorted(c for c in set(found) if found.count(c) > 1)})
        row = {
            'correct': sum(p['correct'] for p in d['probes']), 'questions': len(d['probes']),
            'elapsed_s': d['elapsed_s'], 'prefix_tokens': d['prefix_tokens'],
            'commits': len(d['events']), 'ready_commits': sum(e['ready_before_capacity'] for e in d['events']),
            'foreground_wait_s': sum(e['foreground_wait_s'] for e in d['events']),
            'foreground_wait_each_s': [e['foreground_wait_s'] for e in d['events']],
            'background_compute_s': sum(e['idle_compute_s'] for e in d['events']),
            'forced_compute_s': sum(e['forced_compute_s'] for e in d['events']),
            'prefill_s': sum(e['prefill_s'] for e in d['events']),
            'summary_decode_s': sum(e['decode_s'] for e in d['events']),
            'prefix_prepare_s': sum(e['prefix_prepare_s'] for e in d['events']),
            'commit_s_excluding_lazy_rotation': sum(e['commit_s_excluding_lazy_rotation'] for e in d['events']),
            'max_quantum_s': max(e['max_quantum_s'] for e in d['events']),
            'idle_steps': sum(e['idle_steps'] for e in d['events']),
            'forced_steps': sum(e['forced_steps'] for e in d['events']),
            'scheduled_idle_s': sum(w['requested_ms']/1000 for w in d['idle_windows']),
            'max_arrival_delay_s': max((w['arrival_delay_s'] for w in d['idle_windows']), default=0),
            'foreground_s': sum(b['foreground_s'] for b in d['blocks']),
            'stream_append_s': sum(b['append_s_including_pending_shift'] for b in d['blocks']),
            'pending_phase_at_eof': d['pending_phase'],
            'final_memory_codes': sorted(set(re.findall(r'\b\d{4}\b', d['prefix_text']))),
            'summary_audit': summaries,
        }
        row['stream_tokens_s'] = 24576/row['stream_append_s']
        quanta = [s['seconds'] for e in d['events'] for s in e['steps']]
        row['median_quantum_s'] = statistics.median(quanta)
        assert set(row['final_memory_codes']) <= all_codes
        if mode == 'busy':
            assert row['idle_steps'] == 0 and row['ready_commits'] == 0
        results[mode] = row
    out = {'runs': results, 'summary_tokens_identical': all(
        a['summary_tokens'] == b['summary_tokens']
        for a,b in zip(json.loads((root/'busy.json').read_text())['events'],
                       json.loads((root/'idle.json').read_text())['events']))}
    (root/'aggregate.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
