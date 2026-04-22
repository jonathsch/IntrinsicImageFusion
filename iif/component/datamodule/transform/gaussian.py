import math

from einops import rearrange
from kornia.filters import laplacian
import torch
from torch import nn
import torch.nn.functional as F


class GaussianFiltering(nn.Module):
    """
    Adapted from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self,
                 kernel_size,
                 dim=2):
        super(GaussianFiltering, self).__init__()
        self.kernel_size = kernel_size
        self.dim = dim

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim

        sigma = [k / 6 for k in kernel_size]

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * torch.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())

        self.register_buffer('weight', kernel)

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        shape = x.shape
        x = x.view(-1, 1, *x.shape[-self.dim:])
        x = F.pad(x, [self.kernel_size // 2, self.kernel_size // 2] * self.dim, mode='reflect')
        x = self.conv(x, weight=self.weight)
        x = x.view(shape)
        return x