"""Audit saved branch inventories, controls, interventions, signals and videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_arm(result, *, start, deadline, execute, first_actions, control=None):
    trace, decisions = result['trace'], result['decisions']
    end = result['sim_steps']
    if (result['start_step'] != start or not start < end <= deadline or
            [r['sim_step'] for r in trace] != list(range(start, end + 1)) or
            [d['sim_step'] for d in decisions] != list(range(start, end, execute)) or
            result['continuation_policy_calls'] != len(decisions) - 1):
        raise ValueError('incomplete branch trace or wrong absolute budget/cadence')
    if any(r[k] for r in trace[:-1] for k in ['success', 'terminated', 'truncated']):
        raise ValueError('branch continued after a stop')
    final = trace[-1]
    reason = 'success' if final['success'] else 'truncated' if final['truncated'] else 'terminated' if final['terminated'] else 'budget'
    if (result['success'] != final['success'] or result['stop_reason'] != reason or
            reason == 'budget' and end != deadline):
        raise ValueError('branch outcome differs from saved trace')
    for row in trace:
        if bool(row['labels']['task_success']) != row['success']:
            raise ValueError('official success differs from task signals')
        for name in ['integration_sha256', 'observation_sha256', 'policy_observation_sha256']:
            s = row[name]
            if len(s) != 64 or any(c not in '0123456789abcdef' for c in s):
                raise ValueError('invalid observation or integration hash')
    for decision in decisions:
        if decision['policy_observation_sha256'] != trace[decision['sim_step'] - start]['policy_observation_sha256']:
            raise ValueError('policy decision does not use the recorded current observation')
        if any(not np.isfinite(np.asarray(a)).all() for a in decision['action_chunk'].values()):
            raise ValueError('non-finite branch actions')
    if (set(decisions[0]['action_chunk']) != set(first_actions) or any(
            not np.array_equal(np.asarray(decisions[0]['action_chunk'][k], np.float32), first_actions[k][0])
            for k in first_actions)):
        raise ValueError('first intervention differs from saved prediction')
    if control is not None and trace != control[start:]:
        raise ValueError('reference branch differs from the full verified oracle')
    json.dumps(result, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['report', 'rollout-report', 'candidate-report', 'candidate-dir',
                 'oracle-report', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error('refusing to overwrite analysis')
    from grootN1_Robotics.candidate_outcomes import PILOT_POINTS, summarize_points
    from grootN1_Robotics.decision_capture import file_hash
    from grootN1_Robotics.tools.eval_baseline import save_report
    report, rollout, candidates, oracle = [json.loads(p.read_text()) for p in
        [args.report, args.rollout_report, args.candidate_report, args.oracle_report]]
    smoke = report['purpose'] == 'local_smoke'
    expected = [(rollout['episodes'][0]['seed'], 0)] if smoke else PILOT_POINTS
    if (report['status'] != 'complete' or report['execute'] != 8 or
            report['max_steps'] != rollout['max_steps'] or
            report['selection_rule_evaluated'] is not False or
            report['continuation'] != 'real_policy_closed_loop_with_paired_isolated_rng' or
            [(p['seed'], p['sim_step']) for p in report['points']] != expected):
        parser.error('candidate pilot is incomplete or differs from fixed protocol')
    for name, path in [('rollout', args.rollout_report), ('candidate', args.candidate_report), ('oracle', args.oracle_report)]:
        if report[name + '_report_sha256'] != file_hash(path):
            parser.error('changed reference: ' + name)
    total_frames = 0
    for point in report['points']:
        seed, start = point['seed'], point['sim_step']
        names = ['reference', 'candidate_0'] if smoke else ['reference', *[f'candidate_{i}' for i in range(4)]]
        if [a['arm'] for a in point['arms']] != names:
            parser.error('missing/duplicated intervention arm')
        original = next(e for e in rollout['episodes'] if e['seed'] == seed)
        ctrl = next(e for e in oracle['episodes'] if e['seed'] == seed)
        entry = ctrl['trace_files'][0]
        path = args.oracle_report.parent / entry['file']
        if file_hash(path) != entry['sha256']: parser.error('changed oracle trace')
        control = json.loads(path.read_text())['trace']
        predictions = next(d for d in candidates['decisions'] if (d['seed'], d['sim_step']) == (seed, start))
        for arm in point['arms']:
            name = arm['arm']
            relative = f'episode_{seed}_step_{start}/{name}/continuation.json'
            path = args.report.parent / relative
            if arm['artifact'] != relative or file_hash(path) != arm['artifact_sha256']:
                parser.error('changed branch artifact')
            result = json.loads(path.read_text())
            if name == 'reference':
                decision = next(d for d in original['decisions'] if d['sim_step'] == start)
                actions = {k: np.asarray(v, np.float32)[None] for k, v in decision['action_chunk'].items()}
                if arm['reference_exact_verified'] is not True:
                    parser.error('reference arm was not verified')
            else:
                entry = predictions['candidate_differences'][int(name.split('_')[-1])]
                if entry['actions_file'] != name + '.npz': parser.error('candidate order changed')
                prediction = args.candidate_dir / f'episode_{seed}/step_{start}' / entry['actions_file']
                if file_hash(prediction) != entry['actions_sha256']: parser.error('changed candidate')
                with np.load(prediction, allow_pickle=False) as archive:
                    actions = {k: archive[k].copy() for k in archive.files}
            verify_arm(result, start=start, deadline=report['max_steps'], execute=8,
                       first_actions=actions, control=control if name == 'reference' else None)
            if result['trace'][0] != control[start]: parser.error('changed branch starting state')
            for key in ['success', 'sim_steps', 'stop_reason', 'continuation_policy_calls']:
                if arm[key] != result[key]: parser.error('branch summary differs')
            signals = [json.loads(s) for s in path.with_name(f'episode_{seed}.jsonl').read_text().splitlines()]
            if len(signals) != len(result['trace']): parser.error('incomplete video signals')
            for signal, row in zip(signals, result['trace']):
                if (signal['sim_step'] != row['sim_step'] or signal['success'] != row['success'] or
                        signal['simulation_seconds'] != row['sim_step'] / rollout['control_freq'] or
                        any(signal[k] != v for k, v in row['labels'].items())):
                    parser.error('video signals differ from branch trace')
            import cv2
            video = cv2.VideoCapture(str(path.with_name(f'episode_{seed}.mp4')))
            try:
                if (not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(signals) or
                        abs(video.get(cv2.CAP_PROP_FPS) - rollout['control_freq']) > 1e-6):
                    parser.error('video inventory differs from branch trace')
            finally: video.release()
            metadata = json.loads(path.with_name(f'episode_{seed}.metadata.json').read_text())
            if metadata['start_sim_step'] != start or metadata['frames'] != len(signals):
                parser.error('video start offset differs')
            total_frames += len(signals)
    summary = None if smoke else summarize_points(report['points'])
    if not smoke and report['summary'] != summary: parser.error('candidate summary differs')
    save_report(args.out, {'status': 'verified', 'report_sha256': file_hash(args.report),
        'points': len(expected), 'arms': sum(len(p['arms']) for p in report['points']),
        'video_frames': total_frames, 'summary': summary, 'selection_rule_evaluated': False})
    print(f'independent candidate audit: {len(expected)} points, {total_frames} frames verified')


if __name__ == '__main__': main()
