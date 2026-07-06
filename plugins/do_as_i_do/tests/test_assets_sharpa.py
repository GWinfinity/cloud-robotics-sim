import xml.etree.ElementTree as ET

from do_as_i_do.core.assets import ensure_bimanual_robot


def test_sharpa_merged_urdf_generated():
    """Check that the merged dual UR3 + Sharpa Wave URDF is generated and valid."""
    path = ensure_bimanual_robot(hand_type="sharpa")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert root.tag == "robot"

    link_names = {link.get("name") for link in root.findall("link")}
    assert "base_link" in link_names
    assert "right_tool0" in link_names
    assert "left_tool0" in link_names
    assert "right_hand_flange" in link_names
    assert "left_hand_flange" in link_names

    joint_names = {joint.get("name") for joint in root.findall("joint")}
    assert "right_hand_thumb_CMC_FE" in joint_names
    assert "left_hand_pinky_DIP" in joint_names
