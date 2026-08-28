"""
Visualize robot body mesh and collision spheres using meshcat or trimesh.

Usage:
    python scripts/visualize_robot_spheres.py [--robot-file franka.yml]

If meshcat is installed it will try to open a live viewer; otherwise it will
fall back to assembling a trimesh.Scene and calling `scene.show()`.
"""
import argparse
import traceback
import trimesh
import numpy as np

from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig


def visualize_with_trimesh(trimesh_robot, spheres_np):
    import trimesh

    meshes = [trimesh_robot]
    for s in spheres_np:
        x, y, z, r = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        sph = trimesh.creation.icosphere(subdivisions=3, radius=r)
        sph.apply_translation([x, y, z])
        meshes.append(sph)

    scene = trimesh.Scene(meshes)
    print("Opening trimesh viewer...")
    scene.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-file', type=str, default='franka.yml')
    parser.add_argument('--use-sample-q', action='store_true', help='use sampled q rather than retract_config')
    args = parser.parse_args()

    tensor_args = TensorDeviceType()

    # load robot config
    robot_dict = load_yaml(join_path(get_robot_configs_path(), args.robot_file))
    # Ensure meshes/mesh_link_names are loaded so we can extract meshes
    try:
        if 'robot_cfg' in robot_dict and 'kinematics' in robot_dict['robot_cfg']:
            robot_dict['robot_cfg']['kinematics']['load_link_names_with_mesh'] = True
            robot_dict['robot_cfg']['kinematics']['load_meshes'] = True
            robot_cfg = RobotConfig.from_dict(robot_dict['robot_cfg'], tensor_args)
        else:
            # fallback
            if 'kinematics' in robot_dict:
                robot_dict['kinematics']['load_link_names_with_mesh'] = True
                robot_dict['kinematics']['load_meshes'] = True
            robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)
    except Exception:
        print('Warning: failed to enable mesh loading in robot yaml; proceeding without meshes')
        if 'robot_cfg' in robot_dict:
            robot_cfg = RobotConfig.from_dict(robot_dict['robot_cfg'], tensor_args)
        else:
            robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)

    kin_model = CudaRobotModel(robot_cfg.kinematics)

    # choose a joint configuration: use retract_config (safe default)
    # CudaRobotModel does not provide sample_configs; use retract_config which
    # is a resting joint configuration (shape [dof] or [1,dof]).
    q = kin_model.retract_config.unsqueeze(0) if kin_model.retract_config.dim() == 1 else kin_model.retract_config

    # get robot meshes and combined trimesh
    meshes = kin_model.get_robot_as_mesh(q)
    world = WorldConfig(mesh=meshes[:])
    try:
        robot_mesh = WorldConfig.create_merged_mesh_world(world, process_color=False).mesh[0].get_trimesh_mesh()
    except Exception:
        # fallback: try to convert first mesh only
        try:
            robot_mesh = meshes[0].get_trimesh_mesh()
        except Exception:
            robot_mesh = None

    # get spheres: returns list of lists of Sphere objects
    sph_list = kin_model.get_robot_as_spheres(q)
    # take first (single configuration)
    if len(sph_list) > 0:
        spheres = sph_list[0]
    else:
        spheres = []

    spheres_np = []
    for s in spheres:
        # Sphere has .pose and .radius
        pose = s.pose
        # pose: [x,y,z,qw,qx,qy,qz]
        spheres_np.append([pose[0], pose[1], pose[2], s.radius])
    spheres_np = np.array(spheres_np)

    # Try meshcat first, then trimesh
    if robot_mesh is None:
        print("Warning: unable to assemble robot mesh, will visualize spheres only.")

    try:
        if robot_mesh is None:
            # create empty scene from spheres only
            dummy = trimesh.Trimesh()
            visualize_with_trimesh(dummy, spheres_np)
        else:
            visualize_with_trimesh(robot_mesh, spheres_np)
    except Exception:
        print('trimesh not available or failed to show. Will attempt to save meshes to disk.')
        traceback.print_exc()
        # try to save mesh and spheres to disk so user can open them locally
        try:
            if robot_mesh is not None:
                world.save_world_as_mesh('robot_mesh.stl', process_color=False)
                print('Saved robot mesh to robot_mesh.stl')
            if len(spheres) > 0:
                WorldConfig(sphere=spheres).save_world_as_mesh('robot_spheres.stl')
                print('Saved robot spheres mesh to robot_spheres.stl')
        except Exception:
            print('Failed to save mesh files automatically.')
            traceback.print_exc()


if __name__ == '__main__':
    main()
