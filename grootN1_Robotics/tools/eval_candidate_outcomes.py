"""Evaluate fixed saved interventions with real closed-loop GR00T continuations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_policy(checkpoint, rollout, device):
    from importlib.metadata import version
    import torch
    from common.device import describe
    from grootN1_Robotics.decision_capture import file_hash
    from grootN1_Robotics.policy import LocalGr00tPolicy
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.data.embodiment_tags import EmbodimentTag
    if describe(device) != rollout['accelerator']:
        raise ValueError('candidate pilot accelerator/dtype differs from capture')
    if {p.name: file_hash(p) for p in checkpoint.iterdir()
            if p.suffix in {'.json', '.safetensors'}} != rollout['checkpoint_sha256']:
        raise ValueError('candidate pilot checkpoint differs')
    for name, digest in rollout['code_sha256'].items():
        if (name.endswith('policy.py') or name.endswith('.patch') or
                name.endswith('decision_capture.py')) and file_hash(ROOT / name) != digest:
            raise ValueError(f'candidate policy source differs: {name}')
    for name, expected in rollout['packages'].items():
        if version(name) != expected:
            raise ValueError(f'candidate policy package differs: {name}')
    return Gr00tSimPolicyWrapper(LocalGr00tPolicy(EmbodimentTag(rollout['embodiment']),
        checkpoint, device=str(device.device), dtype=device.amp_dtype or torch.float32))


def candidate_actions(directory, entry, action_space, horizon):
    from grootN1_Robotics.decision_capture import file_hash
    from grootN1_Robotics.baseline import validate_chunk
    if entry['actions_file'] not in {f'candidate_{i}.npz' for i in range(4)}:
        raise ValueError('invalid candidate artifact name')
    path = directory / entry['actions_file']
    if file_hash(path) != entry['actions_sha256']:
        raise ValueError('candidate prediction hash differs')
    with np.load(path, allow_pickle=False) as archive:
        actions = {k: archive[k].copy() for k in archive.files}
    validate_chunk(actions, action_space, horizon)
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'rollout-report', 'capture-dir', 'candidate-report',
                 'candidate-dir', 'oracle-report', 'oracle-analysis', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to overwrite an existing candidate report')
    from common.device import pick_device
    from grootN1_Robotics.baseline import validate_contract
    from grootN1_Robotics.branch_replay import RecordedReplay
    from grootN1_Robotics.candidate_outcomes import PILOT_POINTS, PolicyStream, run_continuation, summarize_points
    from grootN1_Robotics.decision_capture import file_hash, load_capture, replay_decision
    from grootN1_Robotics.diagnostics import drawer_signals, EpisodeRecorder
    from grootN1_Robotics.tools.eval_baseline import create_seeded_env, save_report
    from grootN1_Robotics.tools.verify_branch_replay import validate_simulator_provenance
    from grootN1_Robotics.tools.replay_decisions import validate_capture_rollout
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from types import SimpleNamespace
    rollout = json.loads(args.rollout_report.read_text())
    oracle = json.loads(args.oracle_report.read_text())
    analysis = json.loads(args.oracle_analysis.read_text())
    candidates = json.loads(args.candidate_report.read_text())
    if (rollout['status'] != 'complete' or rollout['execute'] != 8 or
            oracle['status'] != 'complete' or analysis['status'] != 'verified' or
            analysis['report_sha256'] != file_hash(args.oracle_report) or
            oracle['rollout_report_sha256'] != file_hash(args.rollout_report) or
            candidates['status'] != 'complete' or
            candidates['rollout_report_sha256'] != file_hash(args.rollout_report)):
        parser.error('requires hash-linked completed rollout, candidate capture and verified oracle')
    device = pick_device(args.device)
    if args.smoke:
        if rollout['purpose'] != 'local_smoke' or rollout['max_steps'] != 2:
            parser.error('local gate is the existing real two-step smoke')
        points, names = [(rollout['episodes'][0]['seed'], 0)], ['reference', 'candidate_0']
    else:
        if not device.is_cuda or rollout['max_steps'] != 1440:
            parser.error('full candidate outcomes require the cloud 1,440-step control')
        points, names = PILOT_POINTS, ['reference', *[f'candidate_{i}' for i in range(4)]]
    validate_simulator_provenance(rollout)
    configs = {k: SimpleNamespace(**v) for k, v in rollout['contract'].items()}
    horizon = len(configs['action'].delta_indices)
    entries = {}
    for seed, step in points:
        episode = next(e for e in rollout['episodes'] if e['seed'] == seed)
        capture = args.capture_dir / f'episode_{seed}/step_{step}'
        manifest, observation, actions, saved = load_capture(capture)
        validate_capture_rollout(capture, manifest, observation, actions, rollout)
        cand = next(d for d in candidates['decisions'] if (d['seed'], d['sim_step']) == (seed, step))
        if (candidates['provenance'] != manifest['provenance'] or
                not cand['exact_replay_verified'] or
                len(cand['candidate_differences']) != len(names) - 1):
            parser.error('candidate inventory differs from fixed pilot')
        episode_oracle = next(e for e in oracle['episodes'] if e['seed'] == seed)
        trace_file = args.oracle_report.parent / episode_oracle['trace_files'][0]['file']
        if file_hash(trace_file) != episode_oracle['trace_files'][0]['sha256']:
            parser.error('oracle trace hash differs')
        trace = json.loads(trace_file.read_text())['trace']
        if not episode_oracle['exact_reconstruction_verified'] or trace[step]['labels'] != manifest['labels']:
            parser.error('oracle is not an exact match to captured decision')
        entries[(seed, step)] = (episode, capture, observation, actions, saved, cand, trace)
    policy = load_policy(args.checkpoint, rollout, device)
    for episode, capture, *_ in entries.values():
        replay_decision(policy, capture, device.device, candidates=0)
    sources = ['grootN1_Robotics/candidate_outcomes.py',
               'grootN1_Robotics/tools/eval_candidate_outcomes.py',
               'grootN1_Robotics/branch_replay.py']
    report = {'status': 'running', 'purpose': 'local_smoke' if args.smoke else 'candidate_outcome_pilot',
        'rollout_report_sha256': file_hash(args.rollout_report),
        'candidate_report_sha256': file_hash(args.candidate_report),
        'oracle_report_sha256': file_hash(args.oracle_report),
        'code_sha256': {p: file_hash(ROOT / p) for p in sources},
        'execute': 8, 'max_steps': rollout['max_steps'],
        'continuation': 'real_policy_closed_loop_with_paired_isolated_rng',
        'intervention': 'one_saved_chunk_first_8_actions',
        'selection_rule_evaluated': False, 'published_metric_reproduction': False,
        'points': [{'seed': seed, 'sim_step': step, 'arms': []} for seed, step in points]}
    save_report(args.out, report)
    started = time.monotonic()
    try:
        # All reference arms must pass before spending time on counterfactuals.
        for name in names:
            for point in report['points']:
                seed, step = point['seed'], point['sim_step']
                episode, capture, obs, reference, saved, cand, control = entries[(seed, step)]
                env = create_seeded_env(rollout['env'], seed)
                recorder = None
                try:
                    action_space = env.action_space
                    validate_contract(env, configs, 8)
                    if env.unwrapped.env.control_freq != rollout['control_freq']:
                        raise ValueError('candidate environment control frequency differs')
                    env = MultiStepWrapper(env, np.asarray(configs['video'].delta_indices),
                        np.asarray(configs['state'].delta_indices), n_action_steps=1, max_episode_steps=None)
                    import torch
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    cold = RecordedReplay(env, episode, execute=8, horizon=horizon,
                        action_space=action_space, label_reader=drawer_signals,
                        captures={step: obs}, expected_trace=control)
                    cold.advance(step)
                    first = reference if name == 'reference' else candidate_actions(
                        args.candidate_dir / f'episode_{seed}/step_{step}',
                        cand['candidate_differences'][int(name.split('_')[-1])], action_space, horizon)
                    directory = args.out.parent / f'episode_{seed}_step_{step}' / name
                    recorder = EpisodeRecorder(directory, seed, rollout['control_freq'])
                    stream = PolicyStream(policy, saved['after'], device.device)
                    arm_start = time.monotonic()
                    result = run_continuation(cold, first, stream, max_steps=rollout['max_steps'],
                        action_space=action_space, label_reader=drawer_signals, observer=recorder,
                        expected_trace=control if name == 'reference' else None,
                        reference_episode=episode if name == 'reference' else None)
                    path = directory / 'continuation.json'
                    save_report(path, result)
                    row = {k: result[k] for k in ['sim_steps', 'success', 'stop_reason',
                                                 'continuation_policy_calls']}
                    row.update(arm=name, seconds=time.monotonic() - arm_start,
                        artifact=str(path.relative_to(args.out.parent)), artifact_sha256=file_hash(path),
                        reference_exact_verified=name == 'reference')
                    point['arms'].append(row)
                    save_report(args.out, report)
                    print(f'seed {seed} step {step} {name}: {result["stop_reason"]} at '
                          f'{result["sim_steps"]}', flush=True)
                finally:
                    if recorder:
                        recorder.close()
                        metadata_path = recorder.prefix.with_suffix('.metadata.json')
                        metadata = json.loads(metadata_path.read_text())
                        metadata.update(start_sim_step=step,
                            frame_zero='branch observation; frame n is after absolute simulator step start_sim_step+n')
                        save_report(metadata_path, metadata)
                    env.close()
        if not args.smoke:
            report['summary'] = summarize_points(report['points'])
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        save_report(args.out, report)


if __name__ == '__main__':
    main()
