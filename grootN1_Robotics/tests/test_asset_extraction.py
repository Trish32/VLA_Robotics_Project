from pathlib import Path
import zipfile

import pytest

from grootN1_Robotics.tools.fetch_gr1_assets import extract_missing


def test_extract_preserves_pinned_xml_and_is_idempotent(tmp_path):
    archive = tmp_path / "fixtures.zip"
    destination = tmp_path / "assets"
    fixture = destination / "fixtures/toaster"
    fixture.mkdir(parents=True)
    (fixture / "model.xml").write_text("pinned XML")
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("fixtures/toaster/model.xml", "newer XML")
        zipped.writestr("fixtures/toaster/visuals/model.obj", "mesh data")
    assert extract_missing(archive, destination) == ["fixtures/toaster/visuals/model.obj"]
    assert (fixture / "model.xml").read_text() == "pinned XML"
    assert (fixture / "visuals/model.obj").read_text() == "mesh data"
    assert extract_missing(archive, destination) == []


def test_rejects_invalid_archive_before_writing_any_file(tmp_path):
    archive = tmp_path / "bad.zip"
    destination = tmp_path / "assets"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("valid.obj", "mesh")
        zipped.writestr("../outside.obj", "bad")
    with pytest.raises(ValueError, match="escapes"):
        extract_missing(archive, destination)
    assert not (destination / "valid.obj").exists()
    assert not (tmp_path / "outside.obj").exists()


def test_rejects_symlinks(tmp_path):
    archive = tmp_path / "link.zip"
    member = zipfile.ZipInfo("link")
    member.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(member, "/tmp/elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        extract_missing(archive, tmp_path / "assets")
