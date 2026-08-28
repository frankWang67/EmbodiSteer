import multiprocessing as mp
import numpy as np
import time
try:
    from pynput import keyboard
    PYNPUT_IMPORT_ERROR = None
except Exception as exc:  # allow headless CLI help and static imports
    keyboard = None
    PYNPUT_IMPORT_ERROR = exc
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait

class KeyboardSpacemouse(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200, 
            launch_timeout=3.0,
            deadzone=0.05, # 保持接口一致，虽然键盘不需要死区
            dtype=np.float32, 
            n_buttons=2, 
            verbose=False):
        super().__init__(name="KeyboardSpacemouse", daemon=True)
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.n_buttons = n_buttons
        self.dtype = dtype
        
        # 严格参照 Spacemouse 的共享内存数据结构
        example = {
            'motion': np.zeros(6, dtype=dtype),
            'buttons': np.zeros(n_buttons, dtype=np.int64),
            'receive_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

    # ========= API 方法 (与 Spacemouse 完全一致) =========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(self.launch_timeout)
            if not self.ready_event.is_set():
                raise RuntimeError(
                    "KeyboardSpacemouse failed to become ready within "
                    f"{self.launch_timeout}s. This usually means the "
                    "pynput keyboard listener could not start."
                )
            assert self.is_alive()

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_motion_state(self):
        state = self.ring_buffer.get()
        return state['motion']

    def get_motion_state_transformed(self):
        """
        返回变换后的位姿，对应 eval_replay_real_robot.py 中的调用
        """
        state = self.ring_buffer.get()
        return state['motion']

    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]

    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['buttons']

    # ========= 子进程运行逻辑 =========

    def run(self):
        if PYNPUT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "KeyboardSpacemouse requires a graphical pynput backend and "
                "cannot run on a headless machine."
            ) from PYNPUT_IMPORT_ERROR
        # 键盘状态映射
        # 位置: WASD (XY), RF (Z)
        # 旋转: IJKL (Roll/Pitch), UO (Yaw)
        # 按钮: 1, 2
        active_keys = set()

        def on_press(key):
            try:
                active_keys.add(key.char)
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char in active_keys:
                    active_keys.remove(key.char)
            except AttributeError:
                pass

        # 启动非阻塞监听器
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        try:
            iter_idx = 0
            t_start = time.monotonic()
            while not self.stop_event.is_set():
                # 构造 motion 向量 [x, y, z, roll, pitch, yaw]
                motion = np.zeros(6, dtype=self.dtype)
                if 'w' in active_keys: motion[0] = 1.0  # 前
                if 's' in active_keys: motion[0] = -1.0 # 后
                if 'a' in active_keys: motion[1] = 1.0  # 左
                if 'd' in active_keys: motion[1] = -1.0 # 右
                if 'r' in active_keys: motion[2] = 1.0  # 上
                if 'f' in active_keys: motion[2] = -1.0 # 下
                
                if 'i' in active_keys: motion[3] = 1.0
                if 'k' in active_keys: motion[3] = -1.0
                if 'j' in active_keys: motion[4] = 1.0
                if 'l' in active_keys: motion[4] = -1.0
                if 'u' in active_keys: motion[5] = 1.0
                if 'o' in active_keys: motion[5] = -1.0

                # 构造按钮状态
                buttons = np.zeros(self.n_buttons, dtype=np.int64)
                if 'z' in active_keys: buttons[0] = 1
                if 'x' in active_keys: buttons[1] = 1

                # 写入共享内存
                data = {
                    'motion': motion,
                    'buttons': buttons,
                    'receive_timestamp': time.time()
                }
                self.ring_buffer.put(data)

                if iter_idx == 0:
                    self.ready_event.set()
                
                iter_idx += 1
                precise_wait(t_start + iter_idx / self.frequency, time_func=time.monotonic)
        finally:
            listener.stop()
            if self.verbose:
                print("[KeyboardSpacemouse] Listener stopped.")
