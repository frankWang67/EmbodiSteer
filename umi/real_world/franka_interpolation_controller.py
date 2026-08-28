import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.joint_trajectory_interpolator import JointTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    SCHEDULE_JOINT_WAYPOINT = 3


class FrankaInterface:
    def __init__(self, ip: str, port=4242):
        """Connect to an explicitly configured Franka interface server."""
        if not ip:
            raise ValueError("Franka interface host must be provided explicitly")
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        return np.array(self.server.get_ee_pose())
    
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    def start_joint_impedance(self, Kq: np.ndarray, Kqd: np.ndarray):
        self.server.start_joint_impedance(
            Kq.tolist(),
            Kqd.tolist()
        )

    def start_hybrid_impedance(
            self,
            Kq: np.ndarray,
            Kqd: np.ndarray,
            Kx: np.ndarray,
            Kxd: np.ndarray):
        self.server.start_hybrid_impedance(
            Kq.tolist(),
            Kqd.tolist(),
            Kx.tolist(),
            Kxd.tolist()
        )
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def update_desired_joint_positions(self, positions: np.ndarray):
        self.server.update_desired_joint_positions(positions.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()


class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        max_joint_speed=0.6,
        Kq_scale=1.0,
        Kqd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        tx_controller_ee_tip=None,
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        assert 0 < max_joint_speed
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.max_joint_speed = max_joint_speed
        self.Kq = np.array([40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0]) * Kq_scale
        self.Kqd = np.array([4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0]) * Kqd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        if tx_controller_ee_tip is None:
            tx_controller_ee_tip = np.eye(4)
        tx_controller_ee_tip = np.asarray(tx_controller_ee_tip, dtype=np.float64)
        assert tx_controller_ee_tip.shape == (4, 4)
        self.tx_controller_ee_tip = tx_controller_ee_tip
        self.tx_tip_controller_ee = np.linalg.inv(tx_controller_ee_tip)

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_joints': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)
        example['TargetTCPPose'] = np.zeros(6)
        example['TargetQ'] = np.zeros(7)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def schedule_joint_waypoint(self, joints, target_time):
        joints = np.array(joints)
        assert joints.shape == (7,)

        message = {
            'cmd': Command.SCHEDULE_JOINT_WAYPOINT.value,
            'target_joints': joints,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # start polymetis interface
        robot = FrankaInterface(self.robot_ip, self.robot_port)
        tx_controller_ee_tip = self.tx_controller_ee_tip
        tx_tip_controller_ee = self.tx_tip_controller_ee

        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency
            curr_controller_ee_pose = robot.get_ee_pose()
            curr_pose = mat_to_pose(pose_to_mat(curr_controller_ee_pose) @ tx_controller_ee_tip)
            curr_joints = robot.get_joint_positions()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_pose_waypoint_time = curr_t
            last_joint_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            joint_interp = JointTrajectoryInterpolator(
                times=[curr_t],
                joints=[curr_joints]
            )
            use_joint_control = False
            joint_command = curr_joints

            # Start one long-lived Polymetis HybridJointImpedanceControl
            # policy. Both Cartesian and joint commands update the same
            # joint_pos_desired parameter, avoiding controller restarts when
            # switching action spaces.
            robot.start_hybrid_impedance(
                Kq=self.Kq,
                Kqd=self.Kqd,
                Kx=self.Kx,
                Kxd=self.Kxd
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                tip_pose = pose_interp(t_now)
                controller_ee_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_controller_ee)

                # send command to robot
                if use_joint_control:
                    joint_command = joint_interp(t_now)
                    robot.update_desired_joint_positions(joint_command)
                else:
                    robot.update_desired_ee_pose(controller_ee_pose)

                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    value = getattr(robot, func_name)()
                    if key == 'ActualTCPPose':
                        value = mat_to_pose(pose_to_mat(value) @ tx_controller_ee_tip)
                    state[key] = value
                if use_joint_control:
                    # Joint-space commands do not advance pose_interp, so using
                    # it as TargetTCPPose would expose a stale Cartesian target
                    # when eval_real.py hands control back to human teleop.
                    state['TargetTCPPose'] = state['ActualTCPPose']
                else:
                    state['TargetTCPPose'] = tip_pose
                state['TargetQ'] = joint_command

                    
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        curr_time = t_now + dt
                        if use_joint_control:
                            curr_controller_ee_pose = robot.get_ee_pose()
                            curr_pose = mat_to_pose(
                                pose_to_mat(curr_controller_ee_pose) @ tx_controller_ee_tip
                            )
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[curr_time],
                                poses=[curr_pose]
                            )
                            last_pose_waypoint_time = curr_time
                            use_joint_control = False

                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_pose_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        if use_joint_control:
                            curr_controller_ee_pose = robot.get_ee_pose()
                            curr_pose = mat_to_pose(
                                pose_to_mat(curr_controller_ee_pose) @ tx_controller_ee_tip
                            )
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[curr_time],
                                poses=[curr_pose]
                            )
                            last_pose_waypoint_time = curr_time
                            use_joint_control = False
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_pose_waypoint_time
                        )
                        last_pose_waypoint_time = pose_interp.times[-1]
                    elif cmd == Command.SCHEDULE_JOINT_WAYPOINT.value:
                        target_joints = command['target_joints']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        if not use_joint_control:
                            current_joints = robot.get_joint_positions()
                            joint_interp = JointTrajectoryInterpolator(
                                times=[curr_time],
                                joints=[current_joints]
                            )
                            last_joint_waypoint_time = curr_time
                        joint_interp = joint_interp.schedule_waypoint(
                            joints=target_joints,
                            time=target_time,
                            max_joint_speed=self.max_joint_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_joint_waypoint_time
                        )
                        last_joint_waypoint_time = joint_interp.times[-1]
                        use_joint_control = True
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            try:
                robot.terminate_current_policy()
            except Exception as e:
                if self.verbose:
                    print(
                        "[FrankaPositionalController] Ignoring terminate_current_policy "
                        f"failure during cleanup: {e}"
                    )
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
