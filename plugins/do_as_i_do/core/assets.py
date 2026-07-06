"""Asset loading and URDF merging for the do-as-i-do reproduction.

The original do-as-i-do platform uses dual UR3e arms + Sharpa Wave hands.
This scaffold can use either:

* locally available UR3 arms + Allegro Hands (default, no extra download), or
* Sharpa Wave hands from https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

from .retargeting import SUPPORTED_HAND_TYPES

DEFAULT_ROBOT_DATA_REPO = "awesome-robot-descriptions-main"
DEFAULT_ASSET_ROOT = Path("robot_descriptions/Educational/example_robot_data")


def _project_root() -> Path:
    """Return genesis-cloud-sim repository root.

    assets.py lives at plugins/do_as_i_do/core/assets.py, so parents[3] is the
    repository root.
    """
    return Path(__file__).resolve().parents[3]


def default_asset_root() -> Path:
    """Default root containing UR3 / Allegro URDFs."""
    return _project_root().parent / DEFAULT_ROBOT_DATA_REPO / DEFAULT_ASSET_ROOT


def _sharpa_asset_root() -> Path:
    """Root of the downloaded Sharpa Wave URDF/USD/XML assets."""
    return Path(__file__).resolve().parents[1] / "assets" / "sharpa-urdf-usd-xml-main"


def _package_prefix_map(asset_root: Path) -> dict[str, Path]:
    """Map common ROS package prefixes to local directories."""
    sharpa_root = _sharpa_asset_root() / "wave_01"
    return {
        "example-robot-data/robots/ur_description": asset_root / "robots/ur_description",
        "example-robot-data/robots/allegro_hand_description": asset_root
        / "robots/allegro_hand_description",
        "right_sharpa_wave": sharpa_root / "right_sharpa_wave",
        "left_sharpa_wave": sharpa_root / "left_sharpa_wave",
    }


def resolve_ros_package_paths(urdf_text: str, asset_root: Path | None = None) -> str:
    """Rewrite ``package://...`` mesh paths to absolute filesystem paths.

    Genesis does not resolve ROS package prefixes, so we rewrite them before
    loading.
    """
    if asset_root is None:
        asset_root = default_asset_root()

    prefix_map = _package_prefix_map(asset_root)

    def replacer(match: re.Match) -> str:
        package_path = match.group(1)
        for prefix, local_dir in prefix_map.items():
            if package_path.startswith(prefix):
                rel = package_path[len(prefix) + 1 :]  # strip leading slash
                abs_path = (local_dir / rel).resolve().as_posix()
                return f'filename="{abs_path}"'
        # Fallback: treat the whole package path relative to asset_root.
        abs_path = (asset_root / package_path).resolve().as_posix()
        return f'filename="{abs_path}"'

    return re.sub(r'filename="package://([^"]+)"', replacer, urdf_text)


def _prefix_tag(
    elem: ET.Element,
    prefix: str,
    attr: str,
    exclude: Optional[set[str]] = None,
) -> None:
    """Prefix the value of an attribute on an element if it is not excluded."""
    if exclude is None:
        exclude = set()
    val = elem.get(attr)
    if val and val not in exclude:
        elem.set(attr, f"{prefix}{val}")


def _prefix_urdf(
    urdf_text: str,
    prefix: str,
    keep_base_link: bool = False,
) -> str:
    """Prefix all names in a URDF so that multiple copies can coexist.

    Args:
        urdf_text: URDF XML string.
        prefix: prefix to add (e.g. ``left_``).
        keep_base_link: if True, do not rename the original ``base_link`` so it
            can be attached to a new world link.
    """
    root = ET.fromstring(urdf_text)

    protected = {"base_link"} if keep_base_link else set()

    # Prefix robot name.
    name_attr = root.get("name")
    if name_attr:
        root.set("name", f"{prefix}{name_attr}")

    for link in root.findall("link"):
        _prefix_tag(link, prefix, "name", protected)

    for joint in root.findall("joint"):
        _prefix_tag(joint, prefix, "name")
        parent = joint.find("parent")
        if parent is not None:
            _prefix_tag(parent, prefix, "link", protected)
        child = joint.find("child")
        if child is not None:
            _prefix_tag(child, prefix, "link", protected)

    for transmission in root.findall("transmission"):
        _prefix_tag(transmission, prefix, "name")

    for gazebo in root.findall("gazebo"):
        _prefix_tag(gazebo, prefix, "reference")

    # Prefix material names to avoid collisions.
    for material in root.findall("material"):
        _prefix_tag(material, prefix, "name")
    for elem in root.iter():
        if elem.get("material"):
            _prefix_tag(elem, prefix, "material")

    return ET.tostring(root, encoding="unicode")


def _extract_children(root: ET.Element) -> list[ET.Element]:
    """Return all child elements of ``robot`` in order, excluding ``name`` attr."""
    return list(root)


def _prepare_sharpa_hand_urdf(urdf_text: str, side: str) -> str:
    """Rename a Sharpa hand URDF so its links/joints use ``{side}_hand_`` prefix.

    The original files already contain ``right_`` / ``left_`` prefixes.  We
    rewrite them to ``right_hand_`` / ``left_hand_`` to match the arm-side
    naming convention and to avoid double-prefixing.
    """
    old_prefix = f"{side}_"
    new_prefix = f"{side}_hand_"

    # Rename XML name/link/reference/material attributes, but do not rename
    # names that already start with ``right_hand_`` / ``left_hand_``.
    pattern = rf'(?P<attr>name|link|reference)="{old_prefix}(?!hand_)'

    def _repl(m: re.Match) -> str:
        return f'{m.group("attr")}="{new_prefix}'

    text = re.sub(pattern, _repl, urdf_text)

    # Distinguish left/right materials that use an empty name.
    root = ET.fromstring(text)
    material_idx = 0
    for material in root.findall("material"):
        if not material.get("name"):
            material.set("name", f"{new_prefix}material_{material_idx}")
            material_idx += 1
    for elem in root.iter():
        mat = elem.get("material")
        if mat == "":
            elem.set("material", f"{new_prefix}material")

    return ET.tostring(root, encoding="unicode")


def _add_inertial(link_elem: ET.Element) -> None:
    """Add a tiny valid inertial block to a link element."""
    inertial = ET.SubElement(link_elem, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.001",
            "ixy": "0",
            "ixz": "0",
            "iyy": "0.001",
            "iyz": "0",
            "izz": "0.001",
        },
    )


def merge_bimanual_urdf(
    output_path: str | Path,
    arm_urdf: str | Path,
    right_hand_urdf: str | Path,
    left_hand_urdf: str | Path,
    right_hand_attach_link: str,
    left_hand_attach_link: str,
    robot_name: str,
    hand_prepare: Optional[Callable[[str, str], str]] = None,
    arm_mount_offset: float = 0.35,
    hand_z_offset: float = 0.0,
) -> Path:
    """Create a single URDF with two UR3 arms + two hands.

    Args:
        output_path: where to write the merged URDF.
        arm_urdf: path to UR3 robot URDF.
        right_hand_urdf: path to right hand URDF.
        left_hand_urdf: path to left hand URDF.
        right_hand_attach_link: link on the right hand that mounts to the arm tool0.
        left_hand_attach_link: link on the left hand that mounts to the arm tool0.
        robot_name: name of the merged robot.
        hand_prepare: optional callable ``(text, side) -> text`` used to rewrite
            a hand URDF before prefixing/appending.
        arm_mount_offset: half distance between the two arm base links on the
            shared world link.
        hand_z_offset: extra z offset from arm tool0 to hand attach link.
    """
    output_path = Path(output_path)
    asset_root = default_asset_root()

    arm_text = Path(arm_urdf).read_text(encoding="utf-8")
    right_hand_text = Path(right_hand_urdf).read_text(encoding="utf-8")
    left_hand_text = Path(left_hand_urdf).read_text(encoding="utf-8")

    arm_text = resolve_ros_package_paths(arm_text, asset_root)
    right_hand_text = resolve_ros_package_paths(right_hand_text, asset_root)
    left_hand_text = resolve_ros_package_paths(left_hand_text, asset_root)

    if hand_prepare is not None:
        right_hand_text = hand_prepare(right_hand_text, "right")
        left_hand_text = hand_prepare(left_hand_text, "left")
    else:
        right_hand_text = _prefix_urdf(right_hand_text, "right_hand_")
        left_hand_text = _prefix_urdf(left_hand_text, "left_hand_")

    # Prefix all links/joints in each arm so they can coexist.
    right_arm_text = _prefix_urdf(arm_text, "right_")
    left_arm_text = _prefix_urdf(arm_text, "left_")

    # Parse parts.
    right_arm_root = ET.fromstring(right_arm_text)
    left_arm_root = ET.fromstring(left_arm_text)
    right_hand_root = ET.fromstring(right_hand_text)
    left_hand_root = ET.fromstring(left_hand_text)

    # Create new root robot.
    merged = ET.Element("robot", {"name": robot_name})

    # Shared world/base link.
    base_link = ET.SubElement(merged, "link", {"name": "base_link"})
    _add_inertial(base_link)

    # Append all elements from each sub-urdf, skipping transmission/gazebo/mujoco
    # and the original world link / world joint (we attach bases ourselves).
    skip_tags = {"transmission", "gazebo", "mujoco"}
    skip_names = {"right_world", "left_world"}
    for part_root in [right_arm_root, left_arm_root, right_hand_root, left_hand_root]:
        for child in _extract_children(part_root):
            tag = child.tag.split("}")[-1]
            name = child.get("name", "")
            if tag in skip_tags or name in skip_names:
                continue
            # Skip fixed joints that tied the original world link to the arm base.
            if tag == "joint" and child.get("type") == "fixed":
                parent = child.find("parent")
                if parent is not None and parent.get("link") in skip_names:
                    continue
            merged.append(child)

    # Attach arm bases to shared base_link.
    def add_fixed_joint(name: str, parent: str, child: str, xyz: str, rpy: str) -> None:
        joint = ET.SubElement(
            merged,
            "joint",
            {"name": name, "type": "fixed"},
        )
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})

    add_fixed_joint(
        "right_arm_base_joint",
        "base_link",
        "right_base_link",
        xyz=f"0 -{arm_mount_offset} 0",
        rpy="0 0 0",
    )
    add_fixed_joint(
        "left_arm_base_joint",
        "base_link",
        "left_base_link",
        xyz=f"0 {arm_mount_offset} 0",
        rpy="0 0 0",
    )

    # Attach hands to arm tool0 links.
    add_fixed_joint(
        "right_hand_mount_joint",
        "right_tool0",
        right_hand_attach_link,
        xyz=f"0 0 {hand_z_offset}",
        rpy="0 0 0",
    )
    add_fixed_joint(
        "left_hand_mount_joint",
        "left_tool0",
        left_hand_attach_link,
        xyz=f"0 0 {hand_z_offset}",
        rpy="0 0 0",
    )

    # Write output.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_text = ET.tostring(merged, encoding="unicode")
    # Strip any empty namespace prefixes introduced by ElementTree.
    xml_text = xml_text.replace("xmlns:ns0=", "xmlns=")
    output_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n' + xml_text,
        encoding="utf-8",
    )
    return output_path


def merge_bimanual_urdf_allegro(
    output_path: str | Path,
    arm_urdf: str | Path,
    right_hand_urdf: str | Path,
    left_hand_urdf: str | Path,
) -> Path:
    """Merge UR3 arms with Allegro Hands."""
    return merge_bimanual_urdf(
        output_path=output_path,
        arm_urdf=arm_urdf,
        right_hand_urdf=right_hand_urdf,
        left_hand_urdf=left_hand_urdf,
        right_hand_attach_link="right_hand_palm_link",
        left_hand_attach_link="left_hand_palm_link",
        robot_name="dual_ur3_allegro",
    )


def merge_bimanual_urdf_sharpa(
    output_path: str | Path,
    arm_urdf: str | Path,
    right_hand_urdf: str | Path,
    left_hand_urdf: str | Path,
) -> Path:
    """Merge UR3 arms with Sharpa Wave hands."""
    return merge_bimanual_urdf(
        output_path=output_path,
        arm_urdf=arm_urdf,
        right_hand_urdf=right_hand_urdf,
        left_hand_urdf=left_hand_urdf,
        right_hand_attach_link="right_hand_flange",
        left_hand_attach_link="left_hand_flange",
        robot_name="dual_ur3_sharpa",
        hand_prepare=_prepare_sharpa_hand_urdf,
    )


def ensure_bimanual_robot(
    output_dir: str | Path | None = None,
    hand_type: str = "allegro",
) -> Path:
    """Generate the merged dual-UR3 robot URDF if it does not exist.

    Args:
        output_dir: directory for the generated URDF.
        hand_type: ``allegro`` or ``sharpa``.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "assets" / "robots"
    output_dir = Path(output_dir)

    asset_root = default_asset_root()
    arm_urdf = asset_root / "robots/ur_description/urdf/ur3_robot.urdf"
    if not arm_urdf.exists():
        raise FileNotFoundError(f"Missing arm asset: {arm_urdf}")

    if hand_type not in SUPPORTED_HAND_TYPES:
        raise ValueError(f"Unknown hand_type: {hand_type}. Supported: {list(SUPPORTED_HAND_TYPES)}")

    if hand_type == "allegro":
        output_path = output_dir / "dual_ur3_allegro.urdf"
        if output_path.exists():
            return output_path
        right_hand_urdf = asset_root / "robots/allegro_hand_description/urdf/allegro_right_hand.urdf"
        left_hand_urdf = asset_root / "robots/allegro_hand_description/urdf/allegro_left_hand.urdf"
        merger = merge_bimanual_urdf_allegro
    elif hand_type == "sharpa":
        output_path = output_dir / "dual_ur3_sharpa.urdf"
        if output_path.exists():
            return output_path
        sharpa_root = _sharpa_asset_root() / "wave_01"
        right_hand_urdf = sharpa_root / "right_sharpa_wave/right_sharpa_wave_with_flange.urdf"
        left_hand_urdf = sharpa_root / "left_sharpa_wave/left_sharpa_wave_with_flange.urdf"
        merger = merge_bimanual_urdf_sharpa

    for p in [right_hand_urdf, left_hand_urdf]:
        if not p.exists():
            raise FileNotFoundError(f"Missing hand asset: {p}")

    merger(
        output_path=output_path,
        arm_urdf=arm_urdf,
        right_hand_urdf=right_hand_urdf,
        left_hand_urdf=left_hand_urdf,
    )
    return output_path


if __name__ == "__main__":
    out = ensure_bimanual_robot()
    print(f"Generated merged robot URDF: {out}")
