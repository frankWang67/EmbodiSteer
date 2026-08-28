from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si


class JointTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, joints: np.ndarray):
        assert len(times) >= 1
        assert len(joints) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._joints = joints
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.joint_interp = si.interp1d(times, joints, axis=0, assume_sorted=True)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        return self.joint_interp.x

    @property
    def joints(self) -> np.ndarray:
        if self.single_step:
            return self._joints
        return self.joint_interp.y

    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_joints = self(all_times)
        return JointTrajectoryInterpolator(times=all_times, joints=all_joints)

    def schedule_waypoint(
        self,
        joints,
        time,
        max_joint_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ) -> "JointTrajectoryInterpolator":
        assert max_joint_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        trimmed_interp = self.trim(start_time, end_time)

        new_joints = np.asarray(joints)
        duration = time - end_time
        end_joints = trimmed_interp(end_time)
        joint_min_duration = np.max(np.abs(new_joints - end_joints)) / max_joint_speed
        duration = max(duration, joint_min_duration)
        time = end_time + duration

        times = np.append(trimmed_interp.times, [time], axis=0)
        joints_arr = np.append(trimmed_interp.joints, [new_joints], axis=0)
        return JointTrajectoryInterpolator(times=times, joints=joints_arr)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        if self.single_step:
            joints = np.repeat(self._joints[[0]], len(t), axis=0)
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            joints = self.joint_interp(t)

        if is_single:
            joints = joints[0]
        return joints
