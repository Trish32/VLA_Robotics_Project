import xml.etree.ElementTree as ET

import numpy as np
import pytest

from grootN1_Robotics.asset_compat import legacy_bounding_sites

XML = b'''<mujoco><worldbody><body><body name="object">
<geom name="reg_bbox" type="box" pos="0 0 0.01" size="0.02 0.03 0.04"
group="1" contype="0" conaffinity="0" mass="0"/>
<geom type="sphere" size="0.02" friction="0.9 0.3 0.1"/>
</body></body></worldbody></mujoco>'''


def test_metadata_and_idempotency():
    converted = legacy_bounding_sites(XML)
    root = ET.fromstring(converted)
    positions = {s.get("name"): np.fromstring(s.get("pos"), sep=" ") for s in root.iter("site")}
    np.testing.assert_allclose(positions["bottom_site"], [0, 0, -.03])
    np.testing.assert_allclose(positions["top_site"], [0, 0, .05])
    np.testing.assert_allclose(positions["horizontal_radius_site"], [.02, .03, 0])
    assert legacy_bounding_sites(converted) == converted


def test_physical_model_is_unchanged():
    mujoco = pytest.importorskip("mujoco")
    before = mujoco.MjModel.from_xml_string(XML.decode())
    after = mujoco.MjModel.from_xml_string(legacy_bounding_sites(XML).decode())
    assert (before.ngeom, before.nbody, before.nq, before.nv) == (after.ngeom, after.nbody, after.nq, after.nv)
    for attribute in ["geom_type", "geom_pos", "geom_size", "geom_quat", "geom_friction",
                      "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity",
                      "body_mass", "body_inertia"]:
        np.testing.assert_array_equal(getattr(before, attribute), getattr(after, attribute))


@pytest.mark.parametrize("xml", [
    XML.replace(b'type="box"', b'type="sphere"'),
    XML.replace(b'pos="0 0 0.01"', b'pos="0 0 0.01" euler="0 0 1"'),
    XML.replace(b'<body name="object">', b'<body name="object" pos="1 0 0">'),
    XML.replace(b'</body></worldbody>', b'<site name="bottom_site"/></body></worldbody>')])
def test_unsupported_metadata_raises(xml):
    with pytest.raises(ValueError):
        legacy_bounding_sites(xml)
