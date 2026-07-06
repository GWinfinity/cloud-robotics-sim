import xml.etree.ElementTree as ET

from do_as_i_do.core.assets import ensure_bimanual_robot


def test_merged_urdf_generated():
    """Check that the merged dual-arm URDF is generated and valid."""
    path = ensure_bimanual_robot()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert root.tag == "robot"
    link_names = {link.get("name") for link in root.findall("link")}
    assert "base_link" in link_names
    assert "right_tool0" in link_names
    assert "left_tool0" in link_names
    assert "right_hand_palm_link" in link_names
    assert "left_hand_palm_link" in link_names
