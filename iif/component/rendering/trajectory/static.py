from typing import List

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

from iif.utils.logging import init_logger


class StaticTrajectory:
    def __init__(self,
                 cam2world,
                 num_frames):
        self.module_logger = init_logger()
        
        self.cam2world = np.array(cam2world)
        self.num_frames = num_frames
    
    def __len__(self):
        return self.num_frames
    
    def __iter__(self):
        for _ in range(self.num_frames):
            yield self.cam2world

    def __getitem__(self, idx):
        return self.cam2world

