"""Fetch pinned GR1 assets, preserving the upstream model definitions.

The pinned simulator's legacy Box Objaverse URL returns 404. RoboCasa publishes
the archive in its own Hugging Face dataset. Download/extraction manifests live
inside ignored upstream assets; source archives remain in ignored working data.
Missing legacy fixtures can be supplemented from a hash-verified community archive
whose shared model XML is checked byte-for-byte against pinned source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "grootN1_Robotics/upstream/external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/models/assets"
SOURCES = [
    ("robocasa/robocasa-assets", "1b92c3d02ca4354984fec961357db0bff7b32166", "objaverse.zip", "objects"),
    ("nvidia/PhysicalAI-DigitalCousin-Assets", "1b018839a6da865dffecd3185fe054211bc71270", "sketchfab.zip", "objects"),
    ("nvidia/PhysicalAI-DigitalCousin-Assets", "1b018839a6da865dffecd3185fe054211bc71270", "lightwheel.zip", "objects"),
    ("robocasa/robocasa-assets", "1b92c3d02ca4354984fec961357db0bff7b32166", "textures.zip", "."),
    ("robocasa/robocasa-assets", "1b92c3d02ca4354984fec961357db0bff7b32166", "generative_textures.zip", "."),
    ("robocasa/robocasa-assets", "1b92c3d02ca4354984fec961357db0bff7b32166", "fixtures.zip", "."),
]

LEGACY_FIXTURES = ("jianzhang96/robocasa-assets", "866be1d2158a6486af2211156b83eeaeb031f54f",
                   "bd993f47d0ae5f5f3d89cf5f9fc28d55ceb878359c49136d30052de0bb1ef231")


def extract_missing(archive_path, destination):
    """Keep pinned XML and existing assets; add missing files only."""
    destination = Path(destination).resolve()
    added = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"archive path escapes asset directory: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"archive contains symlink: {member.filename}")
        for member in archive.infolist():
            target = destination / member.filename
            if target.exists():
                continue
            archive.extract(member, destination)
            if not member.is_dir():
                added.append(member.filename)
    return added


def supplement_legacy_fixtures():
    from huggingface_hub import hf_hub_download
    repo, commit, expected = LEGACY_FIXTURES
    path = Path(hf_hub_download(repo, "fixtures.zip", repo_type="dataset", revision=commit,
                               local_dir=ROOT / "grootN1_Robotics/data/legacy_fixture_archive"))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("legacy fixture archive SHA256 mismatch")
    # A mirror is usable here because its XML matches the pinned source, not
    # merely because it supplies filenames. Refuse mismatched model definitions.
    common = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            current = ASSETS / name
            if name.endswith("model.xml") and current.exists():
                if archive.read(name) != current.read_bytes():
                    raise ValueError(f"legacy fixture XML differs from pinned source: {name}")
                common.append(name)
    added = extract_missing(path, ASSETS)
    marker = ASSETS / ".legacy_fixtures.manifest.json"
    # Retain the original additions when the command is rerun idempotently.
    if marker.exists():
        added = sorted(set(added) | set(json.loads(marker.read_text())["added_files"]))
    marker.write_text(json.dumps({"repo": repo, "commit": commit,
        "provenance": "community archive; common XML checked against pinned upstream",
        "sha256": expected, "matching_xml": common, "added_files": added}, indent=2) + "\n")
    print(f"legacy fixtures: {len(common)} matching XML; {len(added)} added files", flush=True)


def audit_assets():
    missing = []
    checked = 0
    for path in sorted(ASSETS.rglob("model.xml")):
        root = ET.fromstring(path.read_bytes())
        compiler = root.find("compiler")
        for kind in ("mesh", "texture"):
            subdir = compiler.get(f"{kind}dir", "") if compiler is not None else ""
            for element in root.findall(f"./asset/{kind}"):
                filename = element.get("file")
                if filename is None:
                    continue
                checked += 1
                target = path.parent / subdir / filename
                if not target.exists():
                    missing.append({"xml": str(path.relative_to(ASSETS)), "file": filename})
    report = {"checked_references": checked, "missing_count": len(missing), "missing": missing}
    output = ROOT / "grootN1_Robotics/data/asset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"asset references: {checked} checked, {len(missing)} missing; report: {output}", flush=True)
    return not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[s[2] for s in SOURCES])
    parser.add_argument("--legacy-sites", action="store_true",
                        help="add non-physical legacy sites from authoritative reg_bbox metadata")
    parser.add_argument("--check-assets", action="store_true",
                        help="audit local mesh/texture references without downloading anything")
    parser.add_argument("--legacy-fixtures", action="store_true",
                        help="supplement missing fixtures from a hash-verified legacy community archive")
    args = parser.parse_args()
    if args.check_assets:
        sys.exit(0 if audit_assets() else 1)
    from huggingface_hub import HfApi, hf_hub_download

    for repo, revision, filename, folder in SOURCES:
        if args.only and filename not in args.only:
            continue
        marker = ASSETS / f".{filename}.manifest.json"
        if marker.exists():
            print(f"already extracted: {filename}", flush=True)
            continue
        commit = HfApi().dataset_info(repo, revision=revision).sha
        print(f"fetch {repo}@{commit}/{filename}", flush=True)
        path = Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=commit,
                                   filename=filename,
                                   local_dir=ROOT / "grootN1_Robotics/data/asset_archives"))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        destination = (ASSETS / folder).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        extract_missing(path, destination)
        marker.write_text(json.dumps({"repo": repo, "commit": commit, "filename": filename,
                                      "sha256": digest.hexdigest()}, indent=2) + "\n")
        print(f"extracted: {filename}", flush=True)
    if args.legacy_fixtures:
        supplement_legacy_fixtures()
    if args.legacy_sites:
        from grootN1_Robotics.asset_compat import legacy_bounding_sites
        changes = {}
        for path in sorted((ASSETS / "objects/objaverse").glob("*/*/model.xml")):
            original = path.read_bytes()
            converted = legacy_bounding_sites(original)
            if converted == original:
                continue
            backup = path.with_name("model.xml.pristine")
            if not backup.exists():
                backup.write_bytes(original)
            path.write_bytes(converted)
            changes[str(path.relative_to(ASSETS))] = {
                "before": hashlib.sha256(original).hexdigest(),
                "after": hashlib.sha256(converted).hexdigest()}
        if changes:
            (ASSETS / ".legacy_sites.manifest.json").write_text(json.dumps({
                "normalizer_sha256": hashlib.sha256((ROOT / "grootN1_Robotics/asset_compat.py").read_bytes()).hexdigest(),
                "changes": changes}, indent=2) + "\n")
        print(f"legacy site metadata added to {len(changes)} assets", flush=True)


if __name__ == "__main__":
    main()
