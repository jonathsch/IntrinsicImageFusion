from typing import List

import mitsuba
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

from iif.utils.logging import init_logger


class FlyingSphereLightingTrajectory:
    def __init__(self,
                 size,
                 positions,
                 frames_per_segment, 
                 trajectory_mode="cubic",
                 time_mode="quadratic"):
        self.module_logger = init_logger()
        
        self.positions = positions
        self.size = size
        self.frames_per_segment = frames_per_segment
        self.trajectory_mode = trajectory_mode
        self.time_mode = time_mode

        self.trajectory = self.prepare_trajectory()

    def prepare_trajectory(self):
        return interpolate_positions(positions=self.positions,
                                     frames_per_segment=self.frames_per_segment,
                                     trajectory_mode=self.trajectory_mode,
                                     time_mode=self.time_mode)

    def __len__(self):
        return len(self.trajectory)
    
    def __iter__(self):
        for idx in range(len(self.trajectory)):
            yield self[idx]

    def __getitem__(self, idx):
        position = self.trajectory[idx]
        return mitsuba.ScalarTransform4f().translate(mitsuba.ScalarPoint3f(position))\
                                          .scale(self.size)


def interpolate_positions(
    positions,
    frames_per_segment,
    trajectory_mode="cubic",
    time_mode="linear",
):
    """
    Interpolates smoothly between key 3D positions, with optional nonlinear time scaling.

    Args:
        positions: (N, 3) array of key positions.
        frames_per_segment: list of ints of length N-1 specifying
            how many frames to interpolate between each pair.
        trajectory_mode: 'cubic' | 'linear' | 'polynomial'  (spatial interpolation)
        time_mode: 'linear' | 'quadratic'         (temporal interpolation)

            - 'linear': constant-speed parameterization
            - 'quadratic': ease-in/ease-out (accelerate then decelerate)
    
    Returns:
        (sum(frames_per_segment)+1, 3) array of interpolated positions.
    """
    positions = np.asarray(positions)
    assert positions.ndim == 2 and positions.shape[1] == 3, \
        "positions must be (N, 3)"
    assert len(frames_per_segment) == len(positions) - 1, \
        "frames_per_segment must be one less than number of keyframes."

    # --- Define normalized key times (0 → 1) ---
    segment_durations = np.array(frames_per_segment, dtype=float)
    t = np.concatenate([[0], np.cumsum(segment_durations)])
    t /= t[-1]

    # --- Spatial interpolation (x, y, z) ---
    if trajectory_mode in ("cubic", "polynomial"):
        cs_x = CubicSpline(t, positions[:, 0])
        cs_y = CubicSpline(t, positions[:, 1])
        cs_z = CubicSpline(t, positions[:, 2])
        def interp(ts):
            return np.stack([cs_x(ts), cs_y(ts), cs_z(ts)], axis=1)
    elif trajectory_mode == "linear":
        def interp(ts):
            return np.array([
                np.interp(ts, t, positions[:, 0]),
                np.interp(ts, t, positions[:, 1]),
                np.interp(ts, t, positions[:, 2])
            ]).T
    else:
        raise ValueError(f"Unknown mode: {trajectory_mode}")

    # --- Build linear time samples per segment ---
    t_interp = []
    for i, n in enumerate(frames_per_segment):
        segment_times = np.linspace(t[i], t[i+1], n, endpoint=False)
        t_interp.append(segment_times)
    t_interp.append([t[-1]])  # final keyframe
    t_interp = np.concatenate(t_interp)

    # --- Optional nonlinear time warping (ease-in/ease-out) ---
    if time_mode == "cubic":
        # Ease-in/ease-out mapping (t -> 3t² - 2t³)
        # Starts slow, accelerates, then slows down again
        t_interp = 3 * t_interp**2 - 2 * t_interp**3
    elif time_mode == "quadratic":
        def quadratic_warp(t_):
            t_ = np.clip(t_, 0, 1)
            out = np.empty_like(t_)
            acc = 1
            for i, x in enumerate(t_):
                if x < 0.5:
                    # Quadratic acceleration
                    out[i] = acc * x**2
                else:
                    # Quadratic deceleration
                    out[i] = acc * 0.5**2 + (acc * 0.5**2 - acc * (1 - x)**2)
            # Normalize so final value = 1
            out /= out[-1]
            return out
        t_interp = quadratic_warp(t_interp)
    elif time_mode != "linear":
        raise ValueError(f"Unknown time_mode: {time_mode}")

    # --- Evaluate interpolant ---
    interp_positions = interp(t_interp)

    return interp_positions