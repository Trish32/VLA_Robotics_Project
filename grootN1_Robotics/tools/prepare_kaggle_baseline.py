"""Generate a private Kaggle script from an explicit adaptation-source allowlist.

No checkpoints, upstream source, datasets, credentials or working files are
uploaded. The generated job clones pinned upstream and downloads public assets.
This command stages only; it never authenticates or submits a job.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TESTS = ["test_baseline.py", "test_decision_capture.py", "test_diagnostics.py", "test_asset_extraction.py", "test_asset_compat.py",
         "test_tiny_modules.py", "test_action_head_flow.py", "test_attention_packing.py"]
FILES = ["common/device.py", "grootN1_Robotics/policy.py", "grootN1_Robotics/baseline.py",
         "grootN1_Robotics/diagnostics.py",
         "grootN1_Robotics/decision_capture.py", "grootN1_Robotics/tools/replay_decisions.py",
         "grootN1_Robotics/asset_compat.py", "grootN1_Robotics/tools/eval_baseline.py",
         "grootN1_Robotics/tools/fetch_gr1_assets.py", "grootN1_Robotics/tools/load_checkpoint.py",
         "grootN1_Robotics/tools/check_attention_packing.py",
         *[f"grootN1_Robotics/tests/{name}" for name in TESTS],
         "grootN1_Robotics/patches/0001-build-on-non-flash-non-bf16-targets.patch",
         "grootN1_Robotics/patches/0002-fix-vendored-siglip2-init-and-packing-mask.patch"]

BOOTSTRAP = r'''
import base64, hashlib, io, json, os, pathlib, subprocess, sys, tarfile, traceback
root = pathlib.Path('/tmp/vla_strategy')
root.mkdir(parents=True, exist_ok=True)
outputs = pathlib.Path('/kaggle/working')
payload = base64.b64decode(SOURCE_BUNDLE)
assert hashlib.sha256(payload).hexdigest() == SOURCE_BUNDLE_SHA256
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        assert target.is_relative_to(root) and member.isfile()
    archive.extractall(root, filter='data')
status = {'status': 'setting_up', 'source_bundle_sha256': SOURCE_BUNDLE_SHA256,
          'selection': 'none', 'episodes': EPISODES, 'seed': 0, 'execute': EXECUTE}
if CAPTURE_STEPS:
    status['diagnostic_seeds'] = DIAGNOSTIC_SEEDS
    status['decision_capture_steps'] = CAPTURE_STEPS
def save():
    temp = outputs / 'job_status.json.tmp'
    temp.write_text(json.dumps(status, indent=2))
    temp.replace(outputs / 'job_status.json')
def run(args, **kwargs):
    print('[run]', ' '.join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), check=True, **kwargs)
save()
try:
    reference_path = None
    if REFERENCE_SHA256 is not None:
        matches = [p for p in pathlib.Path('/kaggle/input').rglob('gr1_baseline.json')
                   if hashlib.sha256(p.read_bytes()).hexdigest() == REFERENCE_SHA256]
        if len(matches) != 1:
            raise RuntimeError('expected exactly one hash-matched mounted baseline report')
        reference_path = matches[0]
        status['reference_report_sha256'] = REFERENCE_SHA256
        status['reference_mode'] = REFERENCE_MODE
        status['max_steps'] = MAX_STEPS
        status['budgets'] = BUDGETS
        save()
    upstream = root / 'grootN1_Robotics/upstream'
    run(['git', 'clone', 'https://github.com/NVIDIA/Isaac-GR00T.git', upstream])
    run(['git', '-C', upstream, 'checkout', '9b37aa1ce69c73c6d165233fa88128283bba4508'])
    run(['git', '-C', upstream, 'submodule', 'update', '--init',
         'external_dependencies/robocasa-gr1-tabletop-tasks'])
    for patch in sorted((root / 'grootN1_Robotics/patches').glob('*.patch')):
        run(['git', '-C', upstream, 'apply', '--check', patch])
        run(['git', '-C', upstream, 'apply', patch])
    # Kaggle's system Python is now 3.13, outside pinned GR00T's <3.13 range.
    # A managed interpreter also avoids missing system ensurepip and numpy ABI mixing.
    bootstrap = pathlib.Path('/tmp/gr00t_bootstrap')
    run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--target', bootstrap,
         'uv==0.8.22'])
    uv = bootstrap / 'bin/uv'
    env_dir = pathlib.Path('/tmp/gr00t_eval_env')
    run([uv, 'venv', '--python', '3.11.13', '--seed', env_dir],
        env=dict(os.environ, UV_PYTHON_INSTALL_DIR='/tmp/gr00t_python'))
    python = env_dir / 'bin/python'
    status['python'] = subprocess.check_output([python, '--version'], text=True).strip()
    save()
    run([python, '-m', 'pip', 'install', '--no-cache-dir', 'setuptools', 'wheel'])
    run([python, '-m', 'pip', 'install', '--no-cache-dir', '--index-url',
         'https://download.pytorch.org/whl/cu128', 'torch==2.7.1', 'torchvision==0.22.1'])
    import tomllib
    dependencies = tomllib.loads((upstream / 'pyproject.toml').read_text())['project']['dependencies']
    excluded = ('flash-attn', 'deepspeed', 'tensorrt', 'onnx', 'torchcodec', 'triton')
    status['excluded_inference_dependencies'] = [d for d in dependencies if d.startswith(excluded)]
    requirements = [d for d in dependencies if not d.startswith(excluded)]
    run([python, '-m', 'pip', 'install', '--no-cache-dir', *requirements, 'pytest'])
    sim = upstream / 'external_dependencies/robocasa-gr1-tabletop-tasks'
    run([python, '-m', 'pip', 'install', '--no-cache-dir', '-e', sim,
         'robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@a071383d53568ab798eb315c0e95357911be922d'])
    run([python, '-m', 'pip', 'install', '--no-deps', '-e', upstream])
    environment = dict(os.environ, PYTHONPATH=str(root)+':'+str(upstream),
                       MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                       HF_HOME='/tmp/gr00t_hf_cache', NO_ALBUMENTATIONS_UPDATE='1')
    run([python, '-m', 'pytest', *[root / 'grootN1_Robotics/tests' / n for n in TEST_NAMES], '-q'],
        env=environment, cwd=root)
    run([python, root / 'grootN1_Robotics/tools/fetch_gr1_assets.py',
         '--legacy-fixtures', '--legacy-sites'], env=environment, cwd=root)
    run([python, root / 'grootN1_Robotics/tools/fetch_gr1_assets.py', '--check-assets'],
        env=environment, cwd=root)
    checkpoint = root / 'grootN1_Robotics/checkpoints/GR00T-N1.6-3B'
    download = "from huggingface_hub import snapshot_download; snapshot_download(" + \
        "'nvidia/GR00T-N1.6-3B', revision="+repr(CHECKPOINT_REVISION)+", local_dir="+repr(str(checkpoint))+ \
        ", allow_patterns=['*.json','*.safetensors'])"
    run([python, '-c', download], env=environment, cwd=root)
    for name, expected in CHECKPOINT_SHA256.items():
        path = checkpoint / name
        if not path.exists():
            raise RuntimeError('missing checkpoint file: '+name)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError('local/cloud checkpoint mismatch: '+name)
    run([python, root / 'grootN1_Robotics/tools/load_checkpoint.py', '--checkpoint', checkpoint],
        env=environment, cwd=root)
    command = [python, root / 'grootN1_Robotics/tools/eval_baseline.py', '--checkpoint', checkpoint,
               '--device', 'cuda', '--seed', '0', '--execute', str(EXECUTE), '--max-steps', str(MAX_STEPS),
               '--capture-dir', outputs / 'saved_inputs']
    status['status'] = 'cuda_smoke'
    save()
    smoke_extra = ['--diagnostics-dir', outputs / 'smoke_diagnostics'] if reference_path else []
    if CAPTURE_STEPS:
        smoke_extra += ['--decision-capture-dir', outputs / 'smoke_decisions',
                        '--decision-capture-steps', *map(str, CAPTURE_STEPS)]
    if reference_path and REFERENCE_MODE == 'initial':
        smoke_extra += ['--reference-report', reference_path, '--reference-mode', 'initial']
    run(command+smoke_extra+['--smoke', '--out', outputs / 'gr1_cuda_smoke.json'], env=environment, cwd=root)
    if CAPTURE_STEPS:
        run([python, root / 'grootN1_Robotics/tools/replay_decisions.py', '--checkpoint', checkpoint,
             '--device', 'cuda', '--smoke', '--candidates', '1',
             '--capture-dir', outputs / 'smoke_decisions',
             '--rollout-report', outputs / 'gr1_cuda_smoke.json',
             '--out', outputs / 'replay_cuda_smoke.json'], env=environment, cwd=root)
    status['status'] = 'evaluating'
    save()
    extra = []
    if reference_path:
        extra = ['--reference-report', reference_path, '--reference-mode', REFERENCE_MODE,
                 '--diagnostics-dir', outputs / 'diagnostics',
                 '--budgets', *map(str, BUDGETS)]
    if CAPTURE_STEPS:
        extra += ['--seeds', *map(str, DIAGNOSTIC_SEEDS),
                  '--decision-capture-dir', outputs / 'decisions',
                  '--decision-capture-steps', *map(str, CAPTURE_STEPS)]
    run(command+extra+['--episodes', str(EPISODES), '--out', outputs / 'gr1_baseline.json'],
        env=environment, cwd=root)
    if CAPTURE_STEPS:
        status['status'] = 'replaying_decisions'
        save()
        run([python, root / 'grootN1_Robotics/tools/replay_decisions.py', '--checkpoint', checkpoint,
             '--device', 'cuda', '--candidates', '4', '--capture-dir', outputs / 'decisions',
             '--rollout-report', outputs / 'gr1_baseline.json',
             '--candidate-dir', outputs / 'candidate_predictions',
             '--out', outputs / 'decision_replay.json'], env=environment, cwd=root)
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


def build_script(checkpoint_hashes, revision, episodes, reference_sha256=None, *,
                 execute=8, reference_mode="prefix", capture_steps=None, diagnostic_seeds=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            data = (ROOT / relative).read_bytes()
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    payload = buffer.getvalue()
    header = (f"SOURCE_BUNDLE = {base64.b64encode(payload).decode()!r}\n"
              f"SOURCE_BUNDLE_SHA256 = {hashlib.sha256(payload).hexdigest()!r}\n"
              f"CHECKPOINT_SHA256 = {checkpoint_hashes!r}\n"
              f"CHECKPOINT_REVISION = {revision!r}\nEPISODES = {episodes}\nTEST_NAMES = {TESTS!r}\n")
    header += (f"REFERENCE_SHA256 = {reference_sha256!r}\n"
               f"EXECUTE = {execute}\nREFERENCE_MODE = {reference_mode!r}\n"
               f"MAX_STEPS = {1440 if reference_sha256 else 720}\n"
               f"BUDGETS = {[360, 720, 1080, 1440] if reference_sha256 else []!r}\n")
    header += f"CAPTURE_STEPS = {capture_steps or []!r}\nDIAGNOSTIC_SEEDS = {diagnostic_seeds or []!r}\n"
    return header + BOOTSTRAP


def verify_source_bundle(code):
    """Run the actual uploaded tests in isolation before using cloud quota."""
    namespace = {}
    exec(code.split("import base64, hashlib, io")[0], namespace)
    with tempfile.TemporaryDirectory(prefix="gr00t_upload_gate_") as directory:
        root = Path(directory)
        payload = base64.b64decode(namespace["SOURCE_BUNDLE"])
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            archive.extractall(root)
        (root / "grootN1_Robotics/upstream").symlink_to(
            ROOT / "grootN1_Robotics/upstream", target_is_directory=True)
        environment = dict(os.environ, PYTHONPATH=str(root) + ":" + str(root / "grootN1_Robotics/upstream"),
                           NO_ALBUMENTATIONS_UPDATE="1")
        subprocess.run([sys.executable, "-m", "pytest",
                        *[root / "grootN1_Robotics/tests" / name for name in TESTS], "-q", "--tb=short"],
                       cwd=root, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--owner", default="trishli")
    parser.add_argument("--episodes", type=int, default=10)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--budget-scan-reference", type=Path,
                        help="mount the completed private baseline and compare its exact prefix")
    group.add_argument("--execution-scan-reference", type=Path,
                       help="mount the completed execute-8 budget scan, testing execute 4")
    group.add_argument("--decision-capture-reference", type=Path,
                       help="passively capture four execute-8 failures and two successful controls")
    parser.add_argument("--replay-smoke-report", type=Path,
                        help="completed same-input replay of the supplied local capture smoke")
    parser.add_argument("--out", type=Path, default=ROOT / "grootN1_Robotics/.kaggle_strategy")
    args = parser.parse_args()
    report = json.loads(args.smoke_report.read_text())
    if report.get("status") != "complete" or report.get("purpose") != "local_smoke":
        parser.error("requires a completed actual local environment smoke report")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    reference_sha = None
    reference_path = args.budget_scan_reference or args.execution_scan_reference or args.decision_capture_reference
    reference_mode = "initial" if args.execution_scan_reference else "prefix"
    execute = 4 if args.execution_scan_reference else 8
    if reference_path:
        reference = json.loads(reference_path.read_text())
        expected_steps = 1440 if args.execution_scan_reference or args.decision_capture_reference else 720
        if (reference.get("status") != "complete" or reference.get("max_steps") != expected_steps
                or reference.get("execute") != 8 or reference.get("seed") != 0
                or [e["seed"] for e in reference.get("episodes", [])] != list(range(10)) or args.episodes != 10
                or reference["checkpoint_sha256"] != report["checkpoint_sha256"]):
            parser.error(f"scan requires the completed ten-seed execute-8 {expected_steps}-step baseline with identical weights")
        reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    capture_steps, diagnostic_seeds = [], []
    if args.decision_capture_reference:
        if not args.replay_smoke_report:
            parser.error("decision capture requires --replay-smoke-report")
        replay = json.loads(args.replay_smoke_report.read_text())
        if (replay.get('status') != 'complete' or replay.get('purpose') != 'local_smoke'
                or replay.get('rollout_report_sha256') != hashlib.sha256(args.smoke_report.read_bytes()).hexdigest()
                or len(replay.get('decisions', [])) != 1
                or not replay['decisions'][0].get('exact_replay_verified')
                or not report['episodes'][0].get('decision_captures') or report.get('execute') != 8):
            parser.error("requires exact replay of this actual execute-8 capture smoke")
        for path, digest in report["code_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                parser.error(f"local capture smoke source is stale: {path}")
        capture_steps = [0, 96, 160, 224, 256, 360, 720, 1080, 1432]
        diagnostic_seeds = [0, 2, 3, 5, 7, 8]
        args.episodes = len(diagnostic_seeds)
    code = build_script(report["checkpoint_sha256"], args.checkpoint_revision, args.episodes,
                        reference_sha, execute=execute, reference_mode=reference_mode,
                        capture_steps=capture_steps, diagnostic_seeds=diagnostic_seeds)
    compile(code, "run.py", "exec")
    verify_source_bundle(code)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.py").write_text(code)
    slug = ("gr00t-gr1-decision-capture" if args.decision_capture_reference else
            "gr00t-gr1-execution-scan" if args.execution_scan_reference else
            "gr00t-gr1-budget-scan" if reference_sha else "gr00t-gr1-strategy-baseline")
    metadata = {"id": f"{args.owner}/{slug}", "title": f"GR00T GR1 {slug.removeprefix('gr00t-gr1-').replace('-', ' ')}",
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [f"{args.owner}/gr00t-gr1-budget-scan" if args.execution_scan_reference or args.decision_capture_reference
                                   else f"{args.owner}/gr00t-gr1-strategy-baseline"] if reference_sha else []}
    (args.out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "upload_manifest.json").write_text(json.dumps({
        "source_files": FILES, "run_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "episodes": args.episodes, "checkpoint_revision": args.checkpoint_revision,
        "reference_report_sha256": reference_sha, "reference_mode": reference_mode,
        "execute": execute, "decision_capture_steps": capture_steps,
        "diagnostic_seeds": diagnostic_seeds}, indent=2)+"\n")
    print(f"staged private kernel: {args.out}; {len(FILES)} source files; {len(code)} bytes")


if __name__ == "__main__":
    main()
