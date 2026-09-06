"""Construct the selected gripper backend for either UMI environment."""


def create_gripper_controller(shm_manager, gripper_config):
    """Forward only the selected backend's connection and observation settings.

    Robotiq is the default type, but its serial device must be explicit in the
    config. Import only the selected controller; calling this function may
    connect to and activate hardware, so it is not a configuration preflight.
    """
    gripper_type = gripper_config.get("gripper_type", "robotiq")
    if gripper_type == "robotiq":
        from umi.real_world.robotiq_controller import RobotiqController

        return RobotiqController(
            shm_manager=shm_manager,
            port=gripper_config["gripper_serial_port"],
            slave_id=gripper_config.get("gripper_slave_id", 9),
            receive_latency=gripper_config["gripper_obs_latency"],
        )
    if gripper_type == "wsg50":
        from umi.real_world.wsg_controller import WSGController

        return WSGController(
            shm_manager=shm_manager,
            hostname=gripper_config["gripper_ip"],
            port=gripper_config.get("gripper_port", 1000),
            receive_latency=gripper_config["gripper_obs_latency"],
            use_meters=True,
        )
    raise ValueError(
        f"Unsupported gripper_type {gripper_type!r}; expected 'robotiq' or 'wsg50'"
    )
