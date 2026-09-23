"""Serve the real GR00T N1.6 policy over upstream's ZMQ PolicyServer, on CPU.

The ROS bridge was built and tested against a hand-written fake server, which proves the
wire format but not that the real policy answers the same way. This runs the genuine
3.29B checkpoint behind upstream's own `PolicyServer`, so the bridge can be exercised
end to end without a GPU (~3s per inference on CPU).

`LocalGr00tPolicy` is not a `BasePolicy` subclass, but `PolicyServer` only ever calls
`get_action`, `reset` and `get_modality_config`, all of which it provides -- so it is
served by duck typing rather than by inheritance we would have to fake.

Usage:
    PYTHONPATH=grootN1_Robotics/upstream python grootN1_Robotics/tools/serve_policy.py \
        --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B --port 5555
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--embodiment", default="gr1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--host", default="*")
    args = ap.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.server_client import PolicyServer

    from grootN1_Robotics.policy import LocalGr00tPolicy

    print(f"[serve] loading {args.checkpoint} on cpu/fp32 ...", flush=True)
    policy = LocalGr00tPolicy(EmbodimentTag(args.embodiment), args.checkpoint)

    action_cfg = policy.modality_configs["action"]
    print(f"[serve] embodiment={args.embodiment} "
          f"horizon={len(action_cfg.delta_indices)} keys={action_cfg.modality_keys}",
          flush=True)
    print(f"[serve] use_relative_action={policy.model.config.use_relative_action} "
          "-> this is the bridge's state_relative parameter", flush=True)
    print(f"[serve] listening on tcp://{args.host}:{args.port}", flush=True)

    PolicyServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
