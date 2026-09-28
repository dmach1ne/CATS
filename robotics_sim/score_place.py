"""Report zero-hold placement success from the simulator's 30 Hz traces."""
import argparse
import hashlib
import json
from pathlib import Path

def success(trace):
    return any(abs(f['height_m'] - .0505) <= .005 + 1e-9
        and f['table_contact'] and f['width_m'] >= .06 - 1e-9
        and not f['left_finger_contact'] and not f['right_finger_contact']
        and f['xy_error_m'] <= .05 + 1e-9 for f in trace)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.summary.read_text())
    if data.get('status') != 'complete' or len(data['trials']) != data['expected_num_trials']:
        raise ValueError('Incomplete evaluation; do not report partial runs as completed results')
    trials = data['trials']
    result = {'success_percent': 100 * sum(success(row['trace']) for row in trials) / len(trials),
              'trials': len(trials), 'hold_seconds': 0, 'threshold_xy_m': .05,
              'arm_smoothness': data['summary']['arm_smoothness'],
              'source_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
