import os


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import warnings

import cv2
from PIL import Image
import numpy as np
from torchvision import transforms


def save_video(imgs, path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, imgs.shape[1:3])
    for img in imgs:
        out.write(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

