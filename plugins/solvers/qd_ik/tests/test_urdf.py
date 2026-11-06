"""URDF ingestion tests via genesis's vendored urdfpy (no mesh assets)."""

import numpy as np
import pytest

TOY_URDF = """
<robot name="toy">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 1.5707963267948966"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0.2 0 0" rpy="0.1 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.5" velocity="1.0" effort="10"/>
  </joint>
</robot>
"""


@pytest.fixture()
def toy_urdf(tmp_path):
    """Write the toy URDF into a tmp file."""
    p = tmp_path / "toy.urdf"
    p.write_text(TOY_URDF)
    return p


def test_parse_fixed_base(toy_urdf):
    """Fixed-base parse: tree, placements, axes, limits."""
    from plugins.solvers.qd_ik.urdf import parse_urdf_kinematics

    m = parse_urdf_kinematics(toy_urdf)
    assert m.nj == 3 and m.n_act == 2 and m.nv == 2 and m.nq == 2
    assert m.names == ["universe", "j1", "j2"]
    assert m.parents.tolist() == [0, 0, 1]
    assert np.allclose(m.Rpl[1], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    assert np.allclose(m.ppl[1], [0, 0, 0.1])
    assert np.allclose(m.axis[1], [0, 1, 0])
    assert np.allclose(m.q_min, [-1.0, 0.0])
    assert np.allclose(m.q_max, [1.0, 0.5])
    assert np.allclose(m.v_max, [2.0, 1.0])
    # prismatic: origin rotation kept, no joint rotation
    assert np.allclose(m.ppl[2], [0.2, 0, 0])


def test_parse_floating_base(toy_urdf):
    """Floating-base parse: dof/q indices shifted by the base."""
    from plugins.solvers.qd_ik.urdf import parse_urdf_kinematics

    m = parse_urdf_kinematics(toy_urdf, floating_base=True)
    assert m.nv == 8 and m.nq == 9
    assert m.iv[1] == 6 and m.iq[1] == 7
    assert m.iv[2] == 7 and m.iq[2] == 8
    q0 = m.default_configuration(batch=4)
    assert q0.shape == (4, 9) and np.allclose(q0[:, 3], 1.0)


def test_joint_ordering(toy_urdf):
    """joint_ordering pins the dof order, topology still wins for parents."""
    from plugins.solvers.qd_ik.urdf import parse_urdf_kinematics

    m = parse_urdf_kinematics(toy_urdf, joint_ordering=["j2", "j1"])
    # dof order follows joint_ordering; parents still precede children
    assert m.names.index("j1") < m.names.index("j2")
    assert [m.iv[m.names.index(n)] for n in ("j1", "j2")] == [1, 0]
    with pytest.raises(ValueError):
        parse_urdf_kinematics(toy_urdf, joint_ordering=["j1"])
