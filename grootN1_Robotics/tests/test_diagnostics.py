import json
import numpy as np
import pytest

from grootN1_Robotics.diagnostics import EpisodeRecorder, stage_summary


def signals(env=None):
    return {"grasp_contact": {"left": False, "right": False},
            "object_inside_drawer": False, "drawer_open_fraction": 0.9,
            "drawer_closed": False, "object_xyz": [0., 0., 0.], "task_success": False}


def test_video_and_stage_frames_are_aligned_and_rgb_input_is_preserved(tmp_path):
    import cv2
    image = np.zeros((1, 16, 16, 3), np.uint8)
    image[..., 0] = 220
    original = image.copy()
    recorder = EpisodeRecorder(tmp_path, 1, 20, reader=signals)
    try:
        for step in range(3):
            recorder(None, {"video.camera": image}, step, False)
    finally:
        recorder.close()
    np.testing.assert_array_equal(image, original)
    rows = [json.loads(s) for s in recorder.stages_path.read_text().splitlines()]
    assert [r["sim_step"] for r in rows] == [0, 1, 2]
    assert rows[-1]["simulation_seconds"] == .1
    capture = cv2.VideoCapture(str(recorder.video_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    ok, decoded_bgr = capture.read()
    capture.release()
    assert ok and decoded_bgr[..., 2].mean() > 200 and decoded_bgr[..., 0].mean() < 10
    assert json.loads(recorder.prefix.with_suffix(".metadata.json").read_text())["frames"] == 3
    with pytest.raises(FileExistsError):
        EpisodeRecorder(tmp_path, 1, 20, reader=signals)


def test_stage_predicate_disagreement_is_an_error(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 1, 20, reader=signals)
    try:
        with pytest.raises(ValueError, match="predicate differs"):
            recorder(None, {"video.camera": np.zeros((1, 16, 16, 3), np.uint8)}, 1, True)
    finally:
        recorder.close()


def test_stage_summary_does_not_look_past_the_deadline():
    rows = []
    for step in range(9):
        row = {"sim_step": step, "success": step == 8, **signals()}
        row["grasp_contact"]["left"] = step >= 3
        row["object_inside_drawer"] = step >= 6
        rows.append(row)
    assert stage_summary(rows, 2)["category"] == "no_grasp_contact_detected"
    assert stage_summary(rows, 4)["category"] == "grasp_contact_without_placement"
    assert stage_summary(rows, 7)["category"] == "inside_drawer_without_task_success"
    assert stage_summary(rows, 8)["category"] == "success"
    assert stage_summary(rows, 4)["first_inside_drawer_step"] is None


def test_stage_summary_separates_lost_placement_from_no_placement():
    rows = [{"sim_step": 0, "success": False, **signals()},
            {"sim_step": 1, "success": False, **signals()},
            {"sim_step": 2, "success": False, **signals()}]
    rows[1]["object_inside_drawer"] = True
    assert stage_summary(rows, 2)["category"] == "placement_lost"
