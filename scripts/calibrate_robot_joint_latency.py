# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.common.precise_sleep import precise_wait, precise_sleep
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt


def _parse_joint_indices(value):
    if value is None or len(value.strip()) == 0:
        return np.arange(6, dtype=np.int64)
    joint_indices = np.array([int(x.strip()) for x in value.split(',')], dtype=np.int64)
    if np.any(joint_indices < 0) or np.any(joint_indices >= 6):
        raise click.BadParameter("joint indices must be in [0, 5]")
    return np.unique(joint_indices)


# %%
@click.command()
@click.option('-rh', '--robot_hostname', required=True,
              help='Robot controller hostname or IP address.')
@click.option('-f', '--frequency', type=float, default=10)
@click.option('--controller_frequency', type=float, default=500)
@click.option('--duration', type=float, default=20.0)
@click.option('--warmup', type=float, default=1.0)
@click.option('--amplitude', type=float, default=0.04, help="Per-joint sine amplitude in radians.")
@click.option('--signal_period', type=float, default=3.0, help="Sine period in seconds.")
@click.option('--joints', default='0,1,2,3,4,5', help="Comma-separated joint indices to excite.")
@click.option('--tcp_offset', type=float, default=0.21)
@click.option('--lookahead_time', type=float, default=0.1)
@click.option('--gain', type=int, default=300)
def main(
        robot_hostname,
        frequency,
        controller_frequency,
        duration,
        warmup,
        amplitude,
        signal_period,
        joints,
        tcp_offset,
        lookahead_time,
        gain):
    """
    Calibrate UR joint-space action latency for the RTDE servoJ path.

    This mirrors scripts/calibrate_robot_latency.py, but sends joint-space
    waypoints with schedule_joint_waypoint() and estimates latency from ActualQ.
    """
    assert frequency > 0
    assert controller_frequency > 0
    assert duration > 0
    assert warmup >= 0
    assert amplitude > 0
    assert signal_period > 0

    joint_indices = _parse_joint_indices(joints)
    dt = 1 / frequency
    command_latency = dt / 2
    n_steps = int(np.ceil(duration / dt))

    # Phase offsets keep all joints informative without asking them to move in lockstep.
    phase = np.linspace(0, np.pi, 6, endpoint=False)
    moving_mask = np.zeros(6, dtype=bool)
    moving_mask[joint_indices] = True

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=controller_frequency,
            lookahead_time=lookahead_time,
            gain=gain,
            max_pos_speed=0.1 * np.linalg.norm([1, 1, 1]),
            max_rot_speed=0.3 * np.linalg.norm([1, 1, 1]),
            tcp_offset_pose=[0, 0, tcp_offset, 0, 0, 0],
            get_max_k=int(controller_frequency * (duration + warmup + 5)),
            verbose=False
        ) as controller:
            print('Ready!')
            state = controller.get_state()
            q0 = np.asarray(state['ActualQ'], dtype=np.float64)
            target_joints = q0.copy()

            print(f"Initial joints: {np.array2string(q0, precision=5)}")
            print(
                f"Exciting joints {joint_indices.tolist()} with amplitude={amplitude:.4f}rad, "
                f"period={signal_period:.3f}s at {frequency:.3f}Hz command rate."
            )
            print("Press Ctrl+C to abort if the motion envelope is not safe.")

            if warmup > 0:
                controller.schedule_joint_waypoint(q0, time.time() + 0.2)
                precise_sleep(warmup)

            t_target = list()
            x_target = list()

            t_start = time.time()
            try:
                for iter_idx in range(n_steps):
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    precise_wait(t_sample, time_func=time.time)

                    elapsed = iter_idx * dt
                    sine = np.sin(2 * np.pi * elapsed / signal_period + phase)
                    target_joints = q0.copy()
                    target_joints[moving_mask] = q0[moving_mask] + amplitude * sine[moving_mask]

                    t_target.append(t_command_target)
                    x_target.append(target_joints.copy())

                    controller.schedule_joint_waypoint(
                        joints=target_joints,
                        target_time=t_command_target)

                    precise_wait(t_cycle_end, time_func=time.time)
            finally:
                # Return close to the starting posture before collecting the final buffer.
                controller.schedule_joint_waypoint(q0, time.time() + 0.5)
                precise_sleep(1.0)
                states = controller.get_all_state()

    t_target = np.asarray(t_target)
    x_target = np.asarray(x_target)
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualQ']

    latencies = list()
    infos = list()
    n_dims = 6
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 15, forward=True)

    for i in range(n_dims):
        latency, info = get_latency(
            x_target=x_target[..., i],
            t_target=t_target,
            x_actual=x_actual[..., i],
            t_actual=t_actual)
        latencies.append(latency)
        infos.append(info)

        row = axes[i]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"Joint {i} Cross Correlation")

        ax = row[1]
        ax.plot(t_target, x_target[..., i], label='target')
        ax.plot(t_actual, x_actual[..., i], label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('joint angle (rad)')
        ax.legend()
        ax.set_title(f"Joint {i} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples - latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time')
        ax.set_ylabel('joint angle (rad)')
        ax.legend()
        ax.set_title(f"Joint {i} Aligned with latency={latency:.4f}s")

    latencies = np.asarray(latencies)
    active_latencies = latencies[joint_indices]
    print("Joint-space action latency estimates:")
    for i, latency in enumerate(latencies):
        active = "*" if i in joint_indices else " "
        print(f"  {active} joint {i}: {latency:.6f}s")
    print(f"Median active-joint latency: {np.median(active_latencies):.6f}s")
    print(f"Mean active-joint latency:   {np.mean(active_latencies):.6f}s")

    fig.tight_layout()
    plt.show()


# %%
if __name__ == '__main__':
    main()
