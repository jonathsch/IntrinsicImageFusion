from einops import rearrange
from kornia.filters import laplacian
import torch
from torch import nn
import torch.nn.functional as F


class LaplacianTransform(torch.nn.Module):
    def __init__(self,
                 kernel_size=3,
                 dim=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = dim

        kernel = get_laplacian_kernel(kernel_size, dim)
        self.laplacian = nn.Parameter(kernel.view((1, 1, *kernel.shape)), requires_grad=False)

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
        shape = x.shape
        x = x.reshape(-1, 1, *x.shape[-self.dim:])
        x = F.pad(x, [self.kernel_size // 2, self.kernel_size // 2] * self.dim, mode='reflect')
        x = self.conv(x, self.laplacian)      # x = laplacian(x, self.kernel_size)
        x = x.view(shape)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


def get_log_kernel(kernel_size=3, dim=3):
    """
    Get the Laplacian of Gaussian kernel.
                             1       /    x^2 + y^2 + ...  \     - (x^2 + y^2 + ... ) / (2 * sigma^2)
    LoG(x,y,...) = - -------------- | 1 - ---------------  | * e
                       pi * sigma^4 \      2 * sigma^2    /
    NOTE: Only isotropic Gaussian is supported for now. The kernel is normalized to sum to 1.
    :param kernel_size:
    :param dim:
    :return:
    """
    # Determine sigma
    sigma = kernel_size / 6.0

    # Prepare the kernel
    grids = torch.meshgrid([torch.arange(- (kernel_size // 2), kernel_size // 2 + 1, 1)] * dim)

    kernel = 0
    # Calculate the inner part of the kernel - (x^2 + y^2 + ... ) / (2 * sigma^2)
    for coordinate in grids:
        kernel += (coordinate ** 2)

    # Calculate the first half of the kernel
    kernel = - (1 / (torch.pi * (sigma ** 4))) * (1 - kernel / (2 * sigma ** 2))

    # Calculate the exponential part. Multiplication is addition in the exponential space
    for coordinate in grids:
        kernel *= torch.exp(-(coordinate ** 2 / (2. * sigma ** 2)))  # e^-(x^2 + y^2 + ...  / 2*sigma^2)

    return kernel


def get_laplacian_kernel(kernel_size=3, dim=3):
    """
    Discrete Laplacian kernel
    """
    kernel = torch.ones([kernel_size] * dim)
    mid = kernel_size // 2

    kernel[tuple([mid] * dim)] = 1 - kernel_size ** dim

    return kernel