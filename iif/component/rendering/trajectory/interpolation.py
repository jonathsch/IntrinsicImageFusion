

from typing import List

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

from iif.utils.logging import init_logger


class InterpolationTrajectory:
    def __init__(self,
                 cam2worlds,
                 frames_per_segment, 
                 rotation_mode='slerp',
                 translation_mode="cubic",
                 time_mode="quadratic"):
        self.module_logger = init_logger()
        
        self.cam2worlds = cam2worlds
        self.frames_per_segment = frames_per_segment
        self.rotation_mode = rotation_mode
        self.translation_mode = translation_mode
        self.time_mode = time_mode

        self.trajectory = self.prepare_trajectory()

    def prepare_trajectory(self):
        return interpolate_cam2worlds(cam2worlds=self.cam2worlds,
                                      frames_per_segment=self.frames_per_segment,
                                      rotation_mode=self.rotation_mode,
                                      translation_mode=self.translation_mode,
                                      time_mode=self.time_mode)
    
    def __len__(self):
        return len(self.trajectory)
    
    def __iter__(self):
        for cam2world in self.trajectory:
            yield cam2world

    def __getitem__(self, idx):
        cam2world = self.trajectory[idx]
        return cam2world


def interpolate_cam2worlds(
    cam2worlds,
    frames_per_segment,
    rotation_mode="slerp",
    translation_mode="cubic",
    time_mode="linear",
):
    """
    Interpolates smoothly between key camera-to-world transformations.

    Args:
        cam2worlds: (N, 4, 4) array of key camera-to-world transforms.
        frames_per_segment: list of ints of length N-1 specifying
            how many frames to interpolate between each pair.
        rotation_mode: 'slerp' | 'polynomial' (rotation interpolation type).
        translation_mode: 'cubic' | 'linear' | 'polynomial' (translation interpolation).
    
    Returns:
        (sum(frames_per_segment)+1, 4, 4) array of interpolated cam2worlds.
    """
    cam2worlds = np.asarray(cam2worlds)
    assert cam2worlds.ndim == 3 and cam2worlds.shape[1:] == (4, 4)
    assert len(frames_per_segment) == len(cam2worlds) - 1, \
        "frames_per_segment must be one less than number of keyframes."

    n_keyframes = len(cam2worlds)
    # Assign normalized times for each keyframe, spacing proportional to segment length
    segment_durations = np.array(frames_per_segment, dtype=float)
    t = np.concatenate([[0], np.cumsum(segment_durations)])
    t /= t[-1]  # normalize to [0, 1]

    # --- 1️⃣ Translation interpolation ---
    translations = cam2worlds[:, :3, 3]
    if translation_mode in ("cubic", "polynomial"):
        cs_x = CubicSpline(t, translations[:, 0])
        cs_y = CubicSpline(t, translations[:, 1])
        cs_z = CubicSpline(t, translations[:, 2])
        def interp_translation(ts):
            return np.stack([cs_x(ts), cs_y(ts), cs_z(ts)], axis=1)
    elif translation_mode == "linear":
        def interp_translation(ts):
            return np.array([
                np.interp(ts, t, translations[:, 0]),
                np.interp(ts, t, translations[:, 1]),
                np.interp(ts, t, translations[:, 2])
            ]).T
    else:
        raise ValueError(f"Unknown translation_mode: {translation_mode}")

    # --- 2️⃣ Rotation interpolation ---
    rotations = R.from_matrix(cam2worlds[:, :3, :3])

    if rotation_mode == "slerp":
        slerp = Slerp(t, rotations)
        def interp_rotation(ts):
            return slerp(ts)
    elif rotation_mode == "polynomial":
        # Polynomial interpolation in quaternion space (less physically correct)
        quats = rotations.as_quat()
        cs_q = [CubicSpline(t, quats[:, i]) for i in range(4)]
        def interp_rotation(ts):
            q_interp = np.stack([cs(ts) for cs in cs_q], axis=1)
            q_interp /= np.linalg.norm(q_interp, axis=1, keepdims=True)
            return R.from_quat(q_interp)
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")

    # --- 3️⃣ Build time samples per segment ---
    t_interp = []
    for i, n in enumerate(frames_per_segment):
        segment_times = np.linspace(t[i], t[i+1], n, endpoint=False)
        t_interp.append(segment_times)
    t_interp.append([t[-1]])  # add final keyframe
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

    # --- 4️⃣ Evaluate interpolants ---
    interp_trans = interp_translation(t_interp)
    interp_rots = interp_rotation(t_interp)

    # --- 5️⃣ Assemble full cam2world matrices ---
    cam2world_interp = np.repeat(np.eye(4)[None, :, :], len(t_interp), axis=0)
    cam2world_interp[:, :3, :3] = interp_rots.as_matrix()
    cam2world_interp[:, :3, 3] = interp_trans

    return cam2world_interp