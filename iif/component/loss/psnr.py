import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio as torchmetrics_image_PeakSignalNoiseRatio


class PeakSignalNoiseRatio(torchmetrics_image_PeakSignalNoiseRatio):
    MAX_PSNR = 100.0
    def compute(self):
        psnr = super().compute()
        psnr = torch.nan_to_num(psnr, posinf=self.MAX_PSNR)
        return psnr
