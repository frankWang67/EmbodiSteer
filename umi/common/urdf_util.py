from collections import deque
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import scipy.spatial.transform as st
import yaml


def _origin_to_mat(origin):
    mat = np.eye(4, dtype=np.float64)
    if origin is None:
        return mat

    xyz = origin.attrib.get("xyz", "0 0 0")
    rpy = origin.attrib.get("rpy", "0 0 0")
    xyz = np.fromstring(xyz, sep=" ", dtype=np.float64)
    rpy = np.fromstring(rpy, sep=" ", dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"Invalid URDF origin xyz={xyz} rpy={rpy}")

    roll, pitch, yaw = rpy
    rot = (
        st.Rotation.from_euler("z", yaw).as_matrix()
        @ st.Rotation.from_euler("y", pitch).as_matrix()
        @ st.Rotation.from_euler("x", roll).as_matrix()
    )
    mat[:3, :3] = rot
    mat[:3, 3] = xyz
    return mat


def resolve_fixed_joint_transform(urdf_path, parent_link, child_link):
    """Resolve tx_parent_child through fixed joints in a URDF."""
    urdf_path = pathlib.Path(urdf_path).expanduser()
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    edges = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        tx_parent_child = _origin_to_mat(joint.find("origin"))
        edges.setdefault(parent_name, []).append((child_name, tx_parent_child))
        edges.setdefault(child_name, []).append((parent_name, np.linalg.inv(tx_parent_child)))

    queue = deque([(parent_link, np.eye(4, dtype=np.float64))])
    visited = {parent_link}
    while queue:
        link_name, tx_parent_link = queue.popleft()
        if link_name == child_link:
            return tx_parent_link
        for next_link, tx_link_next in edges.get(link_name, []):
            if next_link in visited:
                continue
            visited.add(next_link)
            queue.append((next_link, tx_parent_link @ tx_link_next))

    raise ValueError(
        f"Could not resolve fixed-joint path from {parent_link!r} to "
        f"{child_link!r} in {urdf_path}"
    )


def _resolve_curobo_urdf_path(robot_config_path, urdf_path):
    urdf_path = pathlib.Path(urdf_path).expanduser()
    if urdf_path.is_absolute():
        return urdf_path

    robot_config_path = pathlib.Path(robot_config_path).expanduser().resolve()
    for parent in [robot_config_path.parent, *robot_config_path.parents]:
        if parent.name == "content":
            candidate = parent / "assets" / urdf_path
            if candidate.exists():
                return candidate

    candidate = robot_config_path.parent / urdf_path
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve cuRobo URDF path {urdf_path!s} from "
        f"{robot_config_path!s}"
    )


def resolve_curobo_ee_transform(robot_config_path, controller_ee_link):
    """Return tx_controller_ee_policy_ee from a cuRobo robot config."""
    robot_config_path = pathlib.Path(robot_config_path).expanduser().resolve()
    with open(robot_config_path, "r") as f:
        robot_config = yaml.safe_load(f)

    kinematics = robot_config["robot_cfg"]["kinematics"]
    ee_link = kinematics["ee_link"]
    urdf_path = _resolve_curobo_urdf_path(robot_config_path, kinematics["urdf_path"])
    tx_controller_ee_policy_ee = resolve_fixed_joint_transform(
        urdf_path=urdf_path,
        parent_link=controller_ee_link,
        child_link=ee_link,
    )
    return tx_controller_ee_policy_ee, ee_link, urdf_path
