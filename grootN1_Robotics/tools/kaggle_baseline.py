"""Submit, inspect or retrieve the prepared private strategy job.

Uses the Kaggle client's existing authentication, including its OAuth cache.
Never prints credentials or modifies the separate FoundationPose kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["submit", "status", "pull", "logs"])
    parser.add_argument("--stage", type=Path, default=ROOT / "grootN1_Robotics/.kaggle_strategy")
    parser.add_argument("--out", type=Path, default=ROOT / "grootN1_Robotics/data/kaggle_baseline")
    parser.add_argument("--seconds", type=int, default=25,
                        help="bounded live-log snapshot duration; status remains the completion authority")
    args = parser.parse_args()
    metadata = json.loads((args.stage / "kernel-metadata.json").read_text())
    if metadata.get("is_private") is not True:
        parser.error("strategy kernel must remain private")
    ref = metadata["id"]
    if ref.split("/")[-1] not in {"gr00t-gr1-strategy-baseline", "gr00t-gr1-budget-scan",
                                 "gr00t-gr1-execution-scan", "gr00t-gr1-decision-capture",
                                 "gr00t-gr1-branch-replay", "gr00t-gr1-cold-branch-replay",
                                 "gr00t-gr1-candidate-outcomes"}:
        parser.error("refusing to operate on an unrelated kernel")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if args.action == "submit":
        manifest = json.loads((args.stage / "upload_manifest.json").read_text())
        code = args.stage / metadata["code_file"]
        if hashlib.sha256(code.read_bytes()).hexdigest() != manifest["run_sha256"]:
            parser.error("staged script hash differs from the reviewed manifest")
        owned = set()
        page = 1
        while True:
            rows = api.kernels_list(mine=True, page=page, page_size=100) or []
            owned.update(k.ref for k in rows)
            if len(rows) < 100:
                break
            page += 1
        # Kaggle returns 403, not 404, for a new private slug. Establish
        # absence through the authenticated owner's complete list instead.
        if ref not in owned:
            state = "absent"
        else:
            state = str(api.kernels_status(ref).status).lower()
        if state != "absent" and state.split(".")[-1] not in {"complete", "error", "cancelled"}:
            parser.error(f"strategy job is active or has an unknown state: {state}")
        result = api.kernels_push(str(args.stage))
        if result.error:
            raise RuntimeError(f"Kaggle rejected submission: {result.error}")
        args.out.mkdir(parents=True, exist_ok=True)
        for name in (metadata["code_file"], "kernel-metadata.json"):
            shutil.copy2(args.stage / name, args.out / name)
        (args.out / "submitted_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.out / "submission.json").write_text(json.dumps({
            "ref": result.ref, "url": result.url, "version": result.version_number,
            "kernel_id": result.kernel_id}, indent=2) + "\n")
        print(result)
    elif args.action == "status":
        print(f"{ref}: {api.kernels_status(ref).status}")
    elif args.action == "logs":
        import signal
        if args.seconds < 1:
            parser.error("log snapshot seconds must be positive")
        class SnapshotExpired(BaseException):
            pass
        def expire(signum, frame):
            raise SnapshotExpired()
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "live.log"
        print(f"bounded log snapshot: {path}", flush=True)
        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(args.seconds)
        try:
            with path.open("w", buffering=1) as handle:
                for event in api.kernels_logs_stream(ref):
                    handle.write(event.get("data", ""))
        except SnapshotExpired:
            print("snapshot duration reached; query status separately", flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        api.kernels_output(ref, str(args.out))
        print(f"output: {args.out}")


if __name__ == "__main__":
    main()
