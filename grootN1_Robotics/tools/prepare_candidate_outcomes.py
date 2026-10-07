"""Stage a private source-only pilot using hash-pinned existing cloud outputs."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grootN1_Robotics.tools.prepare_kaggle_baseline import BOOTSTRAP, verify_source_bundle
from grootN1_Robotics.tools.prepare_branch_replay import FILES as BRANCH_FILES, TESTS as BRANCH_TESTS

FILES = [*BRANCH_FILES, 'grootN1_Robotics/candidate_outcomes.py',
    'grootN1_Robotics/tools/eval_candidate_outcomes.py',
    'grootN1_Robotics/tools/analyze_candidate_outcomes.py',
    'grootN1_Robotics/tests/test_candidate_outcomes.py',
    'grootN1_Robotics/tests/test_candidate_analysis.py', 'grootN1_Robotics/CANDIDATE_OUTCOMES.md']
TESTS = [*BRANCH_TESTS, 'test_candidate_outcomes.py', 'test_candidate_analysis.py']

BODY = r'''
    def mounted(name, digest):
        paths = [p for p in pathlib.Path('/kaggle/input').rglob(name)
                 if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
        if len(paths) != 1:
            raise RuntimeError('expected exactly one hash-matched mount: '+name)
        return paths[0]
    captured = mounted('gr1_baseline.json', CAPTURE_SHA256)
    candidates = mounted('decision_replay.json', CANDIDATE_SHA256)
    oracle = mounted('branch_replay.json', ORACLE_SHA256)
    analysis = mounted('branch_analysis.json', ANALYSIS_SHA256)
    status['status'] = 'candidate_outcomes'
    save()
    run([python, root / 'grootN1_Robotics/tools/eval_candidate_outcomes.py',
         '--checkpoint', checkpoint, '--device', 'cuda',
         '--rollout-report', captured, '--capture-dir', captured.parent / 'decisions',
         '--candidate-report', candidates, '--candidate-dir', candidates.parent / 'candidate_predictions',
         '--oracle-report', oracle, '--oracle-analysis', analysis,
         '--out', outputs / 'candidate_outcomes.json'], env=environment, cwd=root)
    run([python, root / 'grootN1_Robotics/tools/analyze_candidate_outcomes.py',
         '--report', outputs / 'candidate_outcomes.json', '--rollout-report', captured,
         '--candidate-report', candidates, '--candidate-dir', candidates.parent / 'candidate_predictions',
         '--oracle-report', oracle, '--out', outputs / 'candidate_analysis.json'], env=environment, cwd=root)
    status['status'] = 'complete'
except BaseException as error:
    status['status'] = 'failed'
    status['error'] = type(error).__name__+': '+str(error)
    traceback.print_exc()
    save()
    raise
finally:
    save()
'''


def build_script(capture, hashes, revision):
    from grootN1_Robotics.candidate_outcomes import PILOT_POINTS
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for relative in FILES:
            data = (ROOT / relative).read_bytes()
            member = tarfile.TarInfo(relative); member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    payload = buffer.getvalue()
    values = {'SOURCE_BUNDLE': base64.b64encode(payload).decode(),
        'SOURCE_BUNDLE_SHA256': hashlib.sha256(payload).hexdigest(),
        'TEST_NAMES': TESTS, 'EPISODES': len(PILOT_POINTS), 'EXECUTE': 8,
        'REFERENCE_SHA256': None, 'CAPTURE_STEPS': [],
        'CHECKPOINT_SHA256': capture['checkpoint_sha256'], 'CHECKPOINT_REVISION': revision,
        'SIM_PACKAGES': capture['packages'], **hashes}
    header = ''.join(f'{name} = {value!r}\n' for name, value in values.items())
    boundary = "    command = [python, root / 'grootN1_Robotics/tools/eval_baseline.py'"
    if BOOTSTRAP.count(boundary) != 1: raise ValueError('baseline bootstrap boundary changed')
    setup = BOOTSTRAP.split(boundary)[0]
    marker = '    environment = dict(os.environ, PYTHONPATH=str(root)'
    if setup.count(marker) != 1: raise ValueError('baseline environment boundary changed')
    setup = setup.replace(marker, "    run([python, '-m', 'pip', 'install', '--no-cache-dir',\n"
        "         *[name+'=='+v for name, v in SIM_PACKAGES.items() if name != 'torch']])\n" + marker)
    code = header + setup + BODY
    compile(code, 'run.py', 'exec')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['smoke-report', 'smoke-analysis', 'capture-report', 'candidate-report',
                 'oracle-report', 'oracle-analysis']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint-revision', required=True)
    parser.add_argument('--owner', default='trishli')
    parser.add_argument('--out', type=Path, default=ROOT / 'grootN1_Robotics/.kaggle_candidate_outcomes')
    args = parser.parse_args()
    from grootN1_Robotics.decision_capture import file_hash
    smoke, audit, capture, candidate, oracle, analysis = [json.loads(p.read_text()) for p in
        [args.smoke_report, args.smoke_analysis, args.capture_report, args.candidate_report,
         args.oracle_report, args.oracle_analysis]]
    if (smoke['status'] != 'complete' or smoke['purpose'] != 'local_smoke' or
            audit['status'] != 'verified' or audit['report_sha256'] != file_hash(args.smoke_report) or
            audit['arms'] != 2 or audit['video_frames'] != 6):
        parser.error('requires the independently verified real two-arm CPU smoke')
    for path, digest in smoke['code_sha256'].items():
        if file_hash(ROOT / path) != digest: parser.error('local candidate smoke is stale: ' + path)
    for point in smoke['points']:
        for arm in point['arms']:
            if file_hash(args.smoke_report.parent / arm['artifact']) != arm['artifact_sha256']:
                parser.error('local smoke artifact changed')
    if (capture['status'] != 'complete' or capture['execute'] != 8 or capture['max_steps'] != 1440 or
            capture['diagnostic_seeds'] != [0, 2, 3, 5, 7, 8] or
            candidate['status'] != 'complete' or oracle['status'] != 'complete' or
            candidate['rollout_report_sha256'] != file_hash(args.capture_report) or
            oracle['rollout_report_sha256'] != file_hash(args.capture_report) or
            analysis['status'] != 'verified' or analysis['report_sha256'] != file_hash(args.oracle_report) or
            analysis['episodes'] != 6 or analysis['captured_inputs'] != 46 or
            analysis['exact_control_frames'] != 6333):
        parser.error('requires the fixed, independently verified cloud capture and oracle')
    hashes = dict(CAPTURE_SHA256=file_hash(args.capture_report), CANDIDATE_SHA256=file_hash(args.candidate_report),
                  ORACLE_SHA256=file_hash(args.oracle_report), ANALYSIS_SHA256=file_hash(args.oracle_analysis))
    code = build_script(capture, hashes, args.checkpoint_revision)
    verify_source_bundle(code)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'run.py').write_text(code)
    metadata = {'id': f'{args.owner}/gr00t-gr1-candidate-outcomes', 'title': 'GR00T GR1 candidate outcomes',
        'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [], 'competition_sources': [],
        'kernel_sources': [f'{args.owner}/gr00t-gr1-decision-capture',
                           f'{args.owner}/gr00t-gr1-cold-branch-replay']}
    (args.out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    from grootN1_Robotics.candidate_outcomes import PILOT_POINTS
    manifest = {'source_files': FILES, 'run_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'smoke_report_sha256': file_hash(args.smoke_report), 'mounted_hashes': hashes,
        'protocol_sha256': file_hash(ROOT / 'grootN1_Robotics/CANDIDATE_OUTCOMES.md'),
        'pilot_points': PILOT_POINTS, 'arms_per_point': 5, 'max_simulator_steps': 40887,
        'selection_rule_evaluated': False}
    (args.out / 'upload_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'staged private candidate pilot: {args.out}; {len(FILES)} allowlisted files; {len(code)} bytes')


if __name__ == '__main__': main()
