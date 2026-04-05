from typing import List

import mitsuba
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

from iif.utils.logging import init_logger


class StaticTrajectory:
    def __init__(self,
                 size,
                 position,
                 num_frames):
        self.module_logger = init_logger()

        self.position = position
        self.size = size
        self.num_frames = num_frames

        self.trajectory = self.prepare_trajectory()

    def prepare_trajectory(self):
        trajectory = []
        for i in range(self.num_frames):
            trajectory.append(self.position)
        return trajectory
        

    def __len__(self):
        return len(self.trajectory)
    
    def __iter__(self):
        for idx in range(len(self.trajectory)):
            yield self[idx]

    def __getitem__(self, idx):
        position = self.trajectory[idx]
        return mitsuba.ScalarTransform4f().translate(mitsuba.ScalarPoint3f(position))\
                                          .scale(self.size)

