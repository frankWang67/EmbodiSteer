# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import cv2
import time
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.uvc_camera import UvcCamera
from umi.common.usb_util import create_usb_list, get_sorted_v4l_paths
from umi.common.precise_sleep import precise_wait

# %%
def main():
    cv2.setNumThreads(1)

    v4l_paths = get_sorted_v4l_paths()
    dev_video_path = v4l_paths[0]  # change if you want to use a different camera

    # enumerate UBS device to find Elgato Capture Card
    device_list = create_usb_list()
    dev_usb_path = None
    for dev in device_list:
        if 'Elgato' in dev['description']:
            dev_usb_path = dev['path']
            print('Found :', dev['description'])
            break
    
    fps = 60
    dt = 1 / fps

    with SharedMemoryManager() as shm_manager:
        with UvcCamera(
            shm_manager=shm_manager,
            dev_video_path=dev_video_path,
            # dev_usb_path=dev_usb_path,
            resolution=(1920, 1080),
            capture_fps=fps,
        ) as camera:
            print('Ready!')
            t_start = time.monotonic()
            iter_idx = 0
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt

                data = camera.get()
                img = data['color']

                # Display the resulting frame
                cv2.imshow('frame', img)
                if cv2.pollKey() & 0xFF == ord('q'):
                    break

                precise_wait(t_cycle_end)
                iter_idx += 1
                            
# %%
if __name__ == '__main__':
    main()
