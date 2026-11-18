"""Tests for MJCF staging transforms (backends/genesis_backend._stage_mjcf_component).

The upstream mirror stays byte-identical; spawn-time fixes (freejoint
injection, shell inertia, duplicate-name repair) land in content-hashed
staged siblings. These tests exercise the pure XML transforms without a
Genesis engine.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from cloud_robotics_sim.backends.genesis_backend import _stage_mjcf_component

_DUP_TEXTURE_XML = """<mujoco>
  <asset>
    <texture name="tex-knob" builtin="flat" width="1" height="1" type="2d"/>
    <material name="knob_base" texture="tex-knob"/>
    <texture name="tex-knob" builtin="flat" width="1" height="1" type="2d"/>
    <material name="knob_button" texture="tex-knob"/>
  </asset>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.01 0.01 0.01" material="knob_base"/>
    </body>
    <body name="base">
      <geom type="box" size="0.01 0.01 0.01" material="knob_button"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write(tmp_path: Path, content: str) -> Path:
    source = tmp_path / "comp.xml"
    source.write_text(content, encoding="utf-8")
    return source


def test_duplicate_texture_names_renamed_and_rebound(tmp_path: Path) -> None:
    """Later duplicate asset names get __dupN; following references follow.

    Body/site duplicate names are tolerated by the engine and stay untouched.
    """
    source = _write(tmp_path, _DUP_TEXTURE_XML)
    staged = _stage_mjcf_component(source, dynamic=False)
    root = ET.parse(staged).getroot()
    textures = [t.get("name") for t in root.findall(".//asset/texture")]
    assert textures == ["tex-knob", "tex-knob__dup2"]
    materials = {
        m.get("name"): m.get("texture") for m in root.findall(".//asset/material")
    }
    # the material after the second texture binds the renamed one
    assert materials == {
        "knob_base": "tex-knob",
        "knob_button": "tex-knob__dup2",
    }
    bodies = [b.get("name") for b in root.findall(".//worldbody/body")]
    assert bodies == ["base", "base__dup2"]


def test_staging_is_content_hashed_and_idempotent(tmp_path: Path) -> None:
    """The same source stages to the same sibling; reruns reuse it."""
    source = _write(tmp_path, _DUP_TEXTURE_XML)
    first = _stage_mjcf_component(source, dynamic=False)
    second = _stage_mjcf_component(source, dynamic=False)
    assert first == second
    assert first.name != source.name
    # mirror untouched
    assert source.read_text(encoding="utf-8") == _DUP_TEXTURE_XML


def test_dynamic_freejoint_injection_single_body(tmp_path: Path) -> None:
    """A lone top-level body gains a freejoint in dynamic mode."""
    source = _write(
        tmp_path,
        '<mujoco><worldbody><body name="ball">'
        '<geom type="sphere" size="0.01"/></body></worldbody></mujoco>',
    )
    staged = _stage_mjcf_component(source, dynamic=True)
    root = ET.parse(staged).getroot()
    body = root.find(".//worldbody/body")
    assert body is not None and body.find("freejoint") is not None


def test_static_mode_keeps_body_untouched(tmp_path: Path) -> None:
    """Static fixtures are not given freejoints."""
    source = _write(
        tmp_path,
        '<mujoco><worldbody><body name="fix">'
        '<geom type="box" size="0.01 0.01 0.01"/></body></worldbody></mujoco>',
    )
    staged = _stage_mjcf_component(source, dynamic=False)
    root = ET.parse(staged).getroot()
    body = root.find(".//worldbody/body")
    assert body is not None and body.find("freejoint") is None
