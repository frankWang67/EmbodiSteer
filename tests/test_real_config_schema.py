from pathlib import Path

from embodisteer.adapters.real import load_obstacle_config, load_yaml_mapping, validate_robot_config


ROOT = Path(__file__).resolve().parents[1]


def test_robot_template_is_explicitly_incomplete():
    config = load_yaml_mapping(ROOT / "configs" / "real" / "robot.template.yaml")
    validate_robot_config(config, require_addresses=False)


def test_obstacle_layout_is_loadable_and_normalized():
    obstacles = load_obstacle_config(
        ROOT / "configs" / "real" / "obstacles" / "door_frame.yaml"
    )
    assert len(obstacles) == 4
    assert all(tuple(item["quat"].shape) == (1, 4) for item in obstacles)
