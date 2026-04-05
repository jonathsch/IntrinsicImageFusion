from typing import Optional, Union

from kornia.core import pad
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE, KORNIA_CHECK
from kornia.filters.bilateral import _bilateral_blur
from kornia.filters.kernels import _unpack_2d_ks, get_gaussian_kernel2d
from kornia.filters.median import _compute_zero_padding
from torch import Tensor

from iif.component.datamodule.transform.dynamic import DynamicTransform
from iif.utils.image_io import show_image


class GuidedBilateralTransform(DynamicTransform):
    def __init__(self,
                 guidance_cfg,
                 kernel_size=3,
                 sigma_color=0.1,
                 sigma_space=(1.5, 1.5),
                 border_type="reflect",
                 color_distance_type="l1"):
        super().__init__()
        self.guidance_cfg = guidance_cfg
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = tuple(sigma_space)
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def forward(self, x, batch):
        # Apply guided bilateral filter
        x = self._bilateral_blur(x, batch, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type)
        return x

    def _bilateral_blur(
            self,
            input: Tensor,
            guidances: Optional[dict[str, Tensor]],
            kernel_size: Union[tuple[int, int], int],
            sigma_color: Union[float, Tensor],
            sigma_space: Union[tuple[float, float], Tensor],
            border_type: str = "reflect",
            color_distance_type: str = "l1",
    ) -> Tensor:
        """
        Single implementation for both Bilateral Filter and Joint Bilateral Filter
        Adapted from kornia
        """
        KORNIA_CHECK_IS_TENSOR(input)
        KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
        if guidances is not None:
            # NOTE: allow guidance and input having different number of channels
            for guidance_key in self.guidance_cfg.keys():
                assert guidance_key in guidances, f"Guidance_key {guidance_key} not found in guidance"
                guidance = guidances[guidance_key]

                KORNIA_CHECK_IS_TENSOR(guidance)
                KORNIA_CHECK_SHAPE(guidance, ["B", "C", "H", "W"])
                KORNIA_CHECK(
                    (guidance.shape[0] == input.shape[0]) and (guidance.shape[-2:] == input.shape[-2:]),
                    "guidance and input should have the same batch size and spatial dimensions",
                )
        else:
            assert len(self.guidance_cfg) == 0, f"Guidance {list(self.guidance_cfg.keys())} is required for Joint Bilateral Filter"

        # Prepare the input
        if isinstance(sigma_color, Tensor):
            KORNIA_CHECK_SHAPE(sigma_color, ["B"])
            sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1)

        ky, kx = _unpack_2d_ks(kernel_size)
        pad_y, pad_x = _compute_zero_padding(kernel_size)

        padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

        # Prepare the guidance
        if len(self.guidance_cfg) == 0:
            guidance = input
            unfolded_guidance = unfolded_input

            color_distance_sq = self._get_color_distance_sq(unfolded_guidance, guidance, color_distance_type)
            color_kernel = (-0.5 / sigma_color ** 2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)
        else:
            # Joint Bilateral Filtering
            color_kernel = 0
            for guidance_key, guidance_cfg in self.guidance_cfg.items():
                guidance = guidances[guidance_key]

                padded_guidance = pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
                unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

                color_distance_sq = self._get_color_distance_sq(unfolded_guidance, guidance, guidance_cfg["distance"])
                color_kernel = color_kernel + (-0.5 / guidance_cfg["sigma"] ** 2 * color_distance_sq)  # (B, 1, H, W, Ky x Kx)
            color_kernel = color_kernel.exp()

        # Calculate the spatial kernel
        space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
        space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

        # Apply the filter
        kernel = space_kernel * color_kernel
        out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
        return out

    def _get_color_distance_sq(self, neighbors, values, difference_type):
        # Calculate the range kernel
        if difference_type == "l1":
            difference = neighbors - values.unsqueeze(-1)
            color_distance_sq = difference.abs().sum(1, keepdim=True).square()
        elif difference_type == "l2":
            difference = neighbors - values.unsqueeze(-1)
            color_distance_sq = difference.square().sum(1, keepdim=True)
        elif difference_type == "cosine":  # Weird artifact
            difference = neighbors * values.unsqueeze(-1)
            color_distance_sq = difference.sum(1, keepdim=True).square()
        else:
            raise ValueError(f"Unknown color distance type: {difference_type}.")
        return color_distance_sq

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(guidance_key={self.guidance_key})"
