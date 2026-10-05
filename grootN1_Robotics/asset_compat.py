"""Restore legacy bounding sites from official assets' reg_bbox metadata.

Sites are non-physical metadata. Meshes, collision geoms, material definitions,
solver parameters and object scale are preserved. Unsupported layouts raise.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def legacy_bounding_sites(xml: bytes) -> bytes:
    root = ET.fromstring(xml)
    names = {"bottom_site", "top_site", "horizontal_radius_site"}
    present = {site.get("name") for site in root.iter("site")} & names
    if present == names:
        return xml
    if present:
        raise ValueError("partial legacy bounding sites; refuse to guess")
    parent = root.find("./worldbody/body")
    bbox = root.find("./worldbody/body/body[@name='object']/geom[@name='reg_bbox']")
    if parent is None or bbox is None or bbox.get("type") != "box":
        raise ValueError("no authoritative reg_bbox metadata")
    if any(key in bbox.attrib for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("rotated bbox is unsupported")
    body = root.find("./worldbody/body/body[@name='object']")
    if any(key in body.attrib for key in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("transformed object body is unsupported")
    centre = np.fromstring(bbox.get("pos", "0 0 0"), sep=" ")
    half = np.fromstring(bbox.get("size", ""), sep=" ")
    if centre.shape != (3,) or half.shape != (3,) or not np.isfinite([centre, half]).all() or (half <= 0).any():
        raise ValueError("invalid bbox metadata")
    radius = np.maximum(np.abs(centre - half), np.abs(centre + half))
    positions = {"bottom_site": [0, 0, centre[2] - half[2]],
                 "top_site": [0, 0, centre[2] + half[2]],
                 "horizontal_radius_site": [radius[0], radius[1], 0]}
    for name, position in positions.items():
        ET.SubElement(parent, "site", name=name, pos=" ".join(str(x) for x in position),
                      size="0.001", rgba="0 0 0 0", group="3")
    return ET.tostring(root, encoding="utf-8")
