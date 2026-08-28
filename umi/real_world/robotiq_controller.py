import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class RobotiqController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            port: str = "/dev/ttyUSB0",
            frequency=30,
            move_max_speed=255,
            move_max_force=255,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            verbose=False
            ):
        super().__init__(name="RobotiqController", daemon=True)
        try:
            from robotiq_gripper import RobotiqModBusGripper
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The serial Robotiq backend is optional and is only required "
                "when robot_type selects that gripper. Install the matching "
                "hardware driver before real-robot execution."
            ) from exc
        self.gripper = RobotiqModBusGripper(width=0.085, port=port)
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.move_max_force = move_max_force
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        self.gripper.activate()

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # 输入队列
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0, # 0.0 (开) - 1.0 (关) 或 mm
            'target_time': 0.0
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # 状态环形缓冲区
        example = {
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait: 
            self.ready_event.wait(self.launch_timeout)
        if self.verbose:
            print(f"[RobotiqController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
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
    def schedule_waypoint(self, pos: float, target_time: float):
        self.input_queue.put({
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        })

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

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
        try:
            gripper = self.gripper
            
            curr_pos = self.gripper.width
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos, 0, 0, 0, 0, 0]]
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            
            while keep_running:
                t_now = time.monotonic()
                target_val = pose_interp(t_now)[0]
                
                # 发送位置
                gripper.move(target_val)

                # 获取状态并存入 ring_buffer
                # 注：频繁读取夹爪状态可能会降低控制频率
                gripper_pos = gripper.get_position()
                t_recv = time.time()
                state = {
                    'gripper_position': gripper_pos,
                    'gripper_velocity': 0.0,
                    'gripper_force': 0.0,
                    'gripper_receive_timestamp': t_recv,
                    'gripper_timestamp': t_recv - self.receive_latency
                }
                self.ring_buffer.put(state)

                # 处理队列指令
                try:
                    commands = self.input_queue.get_all()
                    for i in range(len(commands['cmd'])):
                        cmd = commands['cmd'][i]
                        if cmd == Command.SHUTDOWN.value:
                            keep_running = False
                            break
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            t_target = commands['target_time'][i]
                            # 全局时间转单调时间
                            t_target_mono = time.monotonic() - time.time() + t_target
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[commands['target_pos'][i], 0, 0, 0, 0, 0],
                                time=t_target_mono,
                                max_pos_speed=self.move_max_speed,
                                curr_time=t_now,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = t_target_mono
                except Empty:
                    pass

                if iter_idx == 0: self.ready_event.set()
                iter_idx += 1
                precise_wait(t_start + iter_idx / self.frequency, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
