

import json
from typing import List

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

from iif.utils.logging import init_logger


class DefinedTrajectory:
    def __init__(self,
                 transforms_path: str,
                 convention: str = "OPEN_CV"):
        self.module_logger = init_logger()

        self.transforms_path = transforms_path
        self.convention = convention

        self.trajectory = self.prepare_trajectory()

    def prepare_trajectory(self):
        camera_path_data = json.load(open(self.transforms_path, "r"))

        # Extract the camera poses from the loaded JSON data
        cam2worlds = []
        for frame in camera_path_data["frames"]:
            cam2world = np.array(frame["transform_matrix"])
            if self.convention == "OPEN_GL":
                # Convert from OpenGL to OpenCV convention
                convert_matrix = np.array([[-1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, -1, 0],
                                            [0, 0, 0, 1]])
                cam2world = cam2world @ convert_matrix
            cam2worlds.append(cam2world)
        cam2worlds = np.array(cam2worlds)

        return cam2worlds

    def __len__(self):
        return len(self.trajectory)
    
    def __iter__(self):
        for cam2world in self.trajectory:
            yield cam2world

    def __getitem__(self, idx):
        cam2world = self.trajectory[idx]
        return cam2world

