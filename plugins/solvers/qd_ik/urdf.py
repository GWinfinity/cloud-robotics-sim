"""URDF -> Model adapter over genesis's vendored urdfpy parser.

Parsing goes through ``genesis.ext.urdfpy.URDF.load`` (pure XML, no
``gs.init``, no mesh loading) so qd_ik shares one URDF semantics
implementation with the simulator instead of carrying its own.  What stays
here is only what urdfpy does not decide for us: the joint *array* order
(the SOA kernels require parents-before-children, and the dof order may be
pinned by the caller through ``joint_ordering``), the fixed-joint tree nodes,
and the actuated limit vectors.
"""

import numpy as np

from .model import FIX, PRIS, REV, Model

_JTYPE_MAP = {
    "revolute": REV,
    "continuous": REV,
    "prismatic": PRIS,
    "fixed": FIX,
}


def parse_urdf_kinematics(path, floating_base=False, joint_ordering=None):
    """Parse ``path`` into a :class:`~qd_ik.model.Model`.

    Args:
        path: URDF file path (xacro is not handled here; expand it first).
        floating_base: attach a quaternion floating base at the root link.
        joint_ordering: optional joint-name sequence fixing the actuated dof
            order.  Must name every movable joint exactly once; fixed joints
            keep their breadth-first positions.

    Raises:
        ValueError: on unsupported joint types, a disconnected joint graph,
            or a joint_ordering that does not match the document.
    """
    # Lazy: importing genesis takes seconds; keep the plugin base import light
    # and gs.init out of the picture entirely (urdfpy parses XML only).
    from genesis.ext.urdfpy import URDF

    robot = URDF.load(str(path))
    joints = robot.joints
    root_link = robot.base_link.name

    movable = [j for j in joints if j.joint_type != "fixed"]
    if joint_ordering is not None:
        wanted = list(joint_ordering)
        if set(wanted) != {j.name for j in movable} or len(wanted) != len(movable):
            raise ValueError("joint_ordering does not match the URDF movable joints")
        rank = {name: k for k, name in enumerate(wanted)}
        dof_order_names = list(wanted)
    else:
        bfs = _breadth_first(joints, root_link)
        rank = {j.name: k for k, j in enumerate(bfs) if j.joint_type != "fixed"}
        dof_order_names = None

    # priority BFS: a joint becomes ready when its parent link is placed;
    # among ready joints, movable ones go first, ordered by `rank`
    added = set()
    ready = []

    def push(link):
        for x in joints:
            if x.parent == link and x.name not in added:
                ready.append(x)
                added.add(x.name)

    push(root_link)
    order = []
    while ready:
        ready.sort(
            key=lambda j: (0, rank[j.name]) if j.joint_type != "fixed" else (1, 0)
        )
        j = ready.pop(0)
        order.append(j)
        push(j.child)
    if len(order) != len(joints):
        raise ValueError(f"{path}: joint graph is not a single tree")

    names = ["universe"]
    parents, jtype = [0], [FIX]
    axis, Rpl, ppl = [np.zeros(3)], [np.eye(3)], [np.zeros(3)]
    q_min, q_max, v_max = [], [], []
    link_to_joint = {}
    for j in order:
        if j.joint_type not in _JTYPE_MAP:
            raise ValueError(f"unsupported joint type {j.joint_type!r} in {path}")
        parent_joint = link_to_joint.get(j.parent, 0)
        link_to_joint[j.child] = len(names)
        names.append(j.name)
        parents.append(parent_joint)
        jt = _JTYPE_MAP[j.joint_type]
        jtype.append(jt)
        T = np.asarray(j.origin, dtype=np.float64).reshape(4, 4)
        Rpl.append(T[:3, :3])
        ppl.append(T[:3, 3])
        if j.axis is None:
            axis.append(np.array([1.0, 0.0, 0.0]))
        else:
            a = np.asarray(j.axis, dtype=np.float64)
            axis.append(a / np.linalg.norm(a))
        if jt != FIX:
            lim = j.limit
            if j.joint_type == "continuous" or lim is None:
                lo, hi, vel = -np.inf, np.inf, np.inf
            else:
                lo = -np.inf if lim.lower is None else float(lim.lower)
                hi = np.inf if lim.upper is None else float(lim.upper)
                vel = np.inf if lim.velocity is None else abs(float(lim.velocity))
            q_min.append(lo)
            q_max.append(hi)
            v_max.append(vel)

    return Model(
        parents=np.array(parents, dtype=np.int32),
        jtype=np.array(jtype, dtype=np.int32),
        axis=np.array(axis),
        Rpl=np.array(Rpl),
        ppl=np.array(ppl),
        q_min=np.array(q_min) if q_min else None,
        q_max=np.array(q_max) if q_max else None,
        v_max=np.array(v_max) if v_max else None,
        names=names,
        floating_base=floating_base,
        dof_order=(
            [names.index(n) for n in dof_order_names] if dof_order_names else None
        ),
    )


def _breadth_first(joints, root_link):
    """Plain BFS joint order (parents before children)."""
    added = set()
    ready = [j for j in joints if j.parent == root_link]
    added.update(j.name for j in ready)
    order = []
    while ready:
        j = ready.pop(0)
        order.append(j)
        for x in joints:
            if x.parent == j.child and x.name not in added:
                ready.append(x)
                added.add(x.name)
    return order
