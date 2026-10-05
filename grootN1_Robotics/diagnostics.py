"""Read-only drawer-task telemetry and videos of observations already rendered upstream."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def drawer_signals(env):
    from robocasa.utils.object_utils import obj_inside_of
    task = env.unwrapped.env
    if not hasattr(task, "drawer") or task.behavior != "close":
        raise ValueError("drawer diagnostics require the official drawer-close task")
    obj = task.objects["obj"]
    inside = bool(obj_inside_of(task, obj.name, task.drawer, partial_check=True))
    door = float(task.drawer.get_door_state(env=task)["door"])
    grasps = {side: bool(task._check_grasp(task.robots[0].gripper[side], obj))
              for side in ("left", "right")}
    return {"grasp_contact": grasps, "object_inside_drawer": inside,
            "drawer_open_fraction": door, "drawer_closed": door <= 0.005,
            "object_xyz": task.sim.data.body_xpos[task.obj_body_id[obj.name]].copy().tolist(),
            "task_success": inside and door <= 0.005}


class EpisodeRecorder:
    """Record every simulator observation at the simulator's control frequency.

    No additional renders, RNG calls, policy input edits or task-state writes.
    Camera panels follow the sorted video keys recorded in metadata.
    """
    def __init__(self, directory: Path, seed: int, fps: float, reader=drawer_signals):
        import cv2
        if fps <= 0:
            raise ValueError("video requires positive control frequency")
        self.cv2, self.fps, self.reader = cv2, fps, reader
        directory.mkdir(parents=True, exist_ok=True)
        self.prefix = directory / f"episode_{seed}"
        self.video_path = self.prefix.with_suffix(".mp4")
        self.stages_path = self.prefix.with_suffix(".jsonl")
        if self.video_path.exists() or self.stages_path.exists():
            raise FileExistsError(f"refusing to overwrite episode diagnostics: {self.prefix}")
        self.handle = self.stages_path.open("x", buffering=1)
        self.writer = None
        self.keys = None
        self.frames = 0
        self.rows = []

    def __call__(self, env, observation, sim_step, success):
        keys = sorted(k for k in observation if k.startswith("video."))
        images = [np.asarray(observation[k])[-1] for k in keys]
        if not images or any(a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8 for a in images):
            raise ValueError("diagnostic video requires temporal uint8 RGB observations")
        if len({a.shape for a in images}) != 1:
            raise ValueError("camera panel sizes differ")
        canvas = np.concatenate(images, axis=1)
        if self.writer is None:
            self.keys = keys
            height, width = canvas.shape[:2]
            self.writer = self.cv2.VideoWriter(str(self.video_path),
                self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError("OpenCV MP4 video encoder could not open")
        if keys != self.keys:
            raise ValueError("camera keys changed mid-episode")
        signals = self.reader(env)
        if bool(signals["task_success"]) != bool(success):
            raise ValueError("read-only task predicate differs from official success flag")
        row = {"sim_step": sim_step, "simulation_seconds": sim_step / self.fps,
               "success": bool(success), **signals}
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        self.writer.write(self.cv2.cvtColor(canvas, self.cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.handle.close()
        self.prefix.with_suffix(".metadata.json").write_text(json.dumps({
            "fps": self.fps, "frames": self.frames, "camera_panels": self.keys,
            "frame_zero": "initial observation; frame n is after simulator step n",
            "grasp_contact": "upstream contact predicate, not a proof of stable grasp",
            "inside_drawer": "official partial_check=True success predicate"}, indent=2)+"\n")


def stage_summary(rows, budget):
    visible = [r for r in rows if r["sim_step"] <= budget]
    if not visible:
        raise ValueError("no stage observations within budget")
    final = visible[-1]
    def first(predicate):
        return next((r["sim_step"] for r in visible if predicate(r)), None)
    grasp = first(lambda r: any(r["grasp_contact"].values()))
    inside = first(lambda r: r["object_inside_drawer"])
    success = first(lambda r: r["success"])
    if success is not None:
        category = "success"
    elif final["object_inside_drawer"]:
        category = "inside_drawer_without_task_success"
    elif inside is not None:
        category = "placement_lost"
    elif grasp is not None:
        category = "grasp_contact_without_placement"
    else:
        category = "no_grasp_contact_detected"
    return {"budget": budget, "observed_through_step": final["sim_step"],
            "first_grasp_contact_step": grasp, "first_inside_drawer_step": inside,
            "first_success_step": success, "category": category,
            "final": final}
