"""Stage a private, source-only simulator replay gate; mount existing capture outputs.

Reuses the tested simulator installation from the baseline worker. No model weights,
datasets, capture payloads or credentials are included in the uploaded script.
"""
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

from grootN1_Robotics.tools.prepare_kaggle_baseline import (
    BOOTSTRAP, FILES as BASE_FILES, TESTS as BASE_TESTS, verify_source_bundle,
)

FILES = [*BASE_FILES, "grootN1_Robotics/branch_replay.py",
         "grootN1_Robotics/tools/verify_branch_replay.py",
         "grootN1_Robotics/tests/test_branch_replay.py",
         "grootN1_Robotics/tools/analyze_branch_replay.py",
         "grootN1_Robotics/tests/test_branch_analysis.py"]
TESTS = [*BASE_TESTS, "test_branch_replay.py", "test_branch_analysis.py"]

BRANCH_BODY = r'''
    status['status'] = 'branch_replay'
    save()
    matches = [p for p in pathlib.Path('/kaggle/input').rglob('gr1_baseline.json')
               if hashlib.sha256(p.read_bytes()).hexdigest() == CAPTURE_REPORT_SHA256]
    if len(matches) != 1:
        raise RuntimeError('expected exactly one hash-matched completed capture report')
    captured = matches[0]
    run([python, root / 'grootN1_Robotics/tools/verify_branch_replay.py',
         '--rollout-report', captured, '--capture-dir', captured.parent / 'decisions',
         '--out', outputs / 'branch_replay.json'], env=environment, cwd=root)
    run([python, root / 'grootN1_Robotics/tools/analyze_branch_replay.py',
         '--report', outputs / 'branch_replay.json', '--rollout-report', captured,
         '--capture-dir', captured.parent / 'decisions',
         '--out', outputs / 'branch_analysis.json'], env=environment, cwd=root)
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


def build_script(capture_hash, packages):
    marker = "    checkpoint = root / 'grootN1_Robotics/checkpoints/GR00T-N1.6-3B'"
    if BOOTSTRAP.count(marker) != 1:
        raise ValueError("baseline worker setup boundary changed; review before staging")
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
              f"CAPTURE_REPORT_SHA256 = {capture_hash!r}\n"
              f"SIM_PACKAGES = {packages!r}\n"
              f"TEST_NAMES = {TESTS!r}\nEPISODES = 6\nEXECUTE = 8\n"
              "CAPTURE_STEPS = []\nREFERENCE_SHA256 = None\n")
    setup = BOOTSTRAP.split(marker)[0]
    environment_marker = "    environment = dict(os.environ, PYTHONPATH=str(root)"
    if setup.count(environment_marker) != 1:
        raise ValueError("baseline worker dependency boundary changed; review before staging")
    pin = ("    run([python, '-m', 'pip', 'install', '--no-cache-dir',\n"
           "         *[name+'=='+v for name, v in SIM_PACKAGES.items() if name != 'torch']])\n")
    setup = setup.replace(environment_marker, pin + environment_marker)
    code = header + setup + BRANCH_BODY
    compile(code, "run.py", "exec")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--owner", default="trishli")
    parser.add_argument("--out", type=Path, default=ROOT / "grootN1_Robotics/.kaggle_branch")
    args = parser.parse_args()
    smoke = json.loads(args.smoke_report.read_text())
    if (smoke.get("status") != "complete" or smoke.get("purpose") != "local_smoke"
            or len(smoke.get("episodes", [])) != 1 or
            not smoke["episodes"][0].get("exact_reconstruction_verified")
            or smoke["episodes"][0].get("pause_steps") != [1]):
        parser.error("requires the real two-step cold-prefix pause/continuation smoke")
    for path, digest in smoke["code_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            parser.error(f"local branch smoke is stale: {path}")
    for entry in smoke["episodes"][0]["trace_files"]:
        if hashlib.sha256((args.smoke_report.parent / entry["file"]).read_bytes()).hexdigest() != entry["sha256"]:
            parser.error("local branch smoke trace changed")
    capture = json.loads(args.capture_report.read_text())
    if (capture.get("status") != "complete" or capture.get("execute") != 8
            or capture.get("max_steps") != 1440 or capture.get("diagnostic_seeds") != [0, 2, 3, 5, 7, 8]):
        parser.error("requires the completed fixed six-seed decision capture")
    code = build_script(hashlib.sha256(args.capture_report.read_bytes()).hexdigest(), capture["packages"])
    verify_source_bundle(code)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.py").write_text(code)
    metadata = {"id": f"{args.owner}/gr00t-gr1-cold-branch-replay", "title": "GR00T GR1 cold branch replay",
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [f"{args.owner}/gr00t-gr1-decision-capture"]}
    (args.out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "upload_manifest.json").write_text(json.dumps({"source_files": FILES,
        "run_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "smoke_report_sha256": hashlib.sha256(args.smoke_report.read_bytes()).hexdigest(),
        "capture_report_sha256": hashlib.sha256(args.capture_report.read_bytes()).hexdigest(),
        "candidate_quality_evaluated": False, "diagnostic_seeds": [0, 2, 3, 5, 7, 8]}, indent=2) + "\n")
    print(f"staged private branch oracle: {args.out}; {len(FILES)} source files; {len(code)} bytes")


if __name__ == "__main__":
    main()
