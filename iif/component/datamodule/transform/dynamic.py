import functools

import torch
from einops import rearrange, reduce

from iif.utils.image_io import show_image


class DynamicTransform(torch.nn.Module):
    pass


class ScaleMatchingTransform(DynamicTransform):
    def __init__(self,
                 target_key,
                 invariant_dims):
        super().__init__()
        self.target_key = target_key
        self.invariant_dims = invariant_dims

    def forward(self, x, batch):
        target = batch[self.target_key]

        # Scale the input to the target
        target_mean = target.mean(dim=self.invariant_dims, keepdim=True)
        x_mean = x.mean(dim=self.invariant_dims, keepdim=True)
        x_mean[x_mean == 0] = 1e-6
        x = x / x_mean * target_mean

        return x


class GridSampleTransform(DynamicTransform):
    def __init__(self,
                 grid_key,
                 interpolation_mode='bilinear'):
        super().__init__()
        self.grid_key = grid_key
        self.interpolation_mode = interpolation_mode

        self.grid = None

    def forward(self, x, batch):
        # Create the sampling grid
        grid = batch[self.grid_key]  # Expected in the [-1, 1] range
        grid = grid.detach()
        grid_shape = grid.shape

        if grid.shape[0] == 1:
            grid = grid.unsqueeze(-1)  # (B, D, H, W, 1)
            if self.grid is None:
                self.grid = (torch.stack(
                    torch.meshgrid(
                        torch.linspace(-1, 1, steps=grid.shape[-3]),
                        torch.linspace(-1, 1, steps=grid.shape[-2])), dim=-1)
                             .unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1, 1)
                             .to(grid.device).to(grid.dtype))  # (B, D, H, W, 2)
            grid = torch.cat([grid, self.grid.expand(grid.shape[0], *self.grid.shape[1:])], dim=-1)
        else:
            grid = grid.permute(0, 2, 3, 1).unsqueeze(1)

        # Prepare x for sampling
        x_shape = x.shape
        x = x.view(x_shape[0], functools.reduce(lambda x, y: x*y, x_shape[1:-3]), *x.shape[-3:])

        # Grid Sample
        x = torch.nn.functional.grid_sample(input=x,
                                            grid=grid,
                                            mode=self.interpolation_mode,
                                            align_corners=False,
                                            padding_mode="reflection")
        x = x.squeeze(-3).view(*x_shape[:-3], *grid_shape[-2:])

        return x


class TriPlaneQueryTransform(DynamicTransform):
    def __init__(self,
                 position_key,
                 interpolation_mode='bilinear',
                 **kwargs):
        super().__init__()
        self.position_key = position_key
        self.interpolation_mode = interpolation_mode

    def forward(self, x, batch):
        """
        :param x: TriPlanes
        :param batch: meta information, stores the positions in [-1, 1] range
        :return:
        """
        # return x
        # return x[:, :, 0, :, :]
        # return x[:, :1, :, :, :]

        # Create the positions
        position = batch[self.position_key]  # Expected in the [-1, 1] range
        position = position.detach()
        position_shape = position.shape
        position = rearrange(position, "B Np Hp Wp -> B (Hp Wp) Np")

        # Prepare the grid
        grid = torch.stack(
            (position[..., [0, 1]], position[..., [0, 2]], position[..., [1, 2]]),
            dim=-3,
        )  # (B, Np, X, 2)

        # Grid Sample
        if x.ndim == 4:
            x = rearrange(x, "B (Np Cf) Hp Wp -> B Np Cf Hp Wp", Np=3)

        # return x[:,0,:,:,:]


        x = torch.nn.functional.grid_sample(
            rearrange(x, "B Np Cf Hf Wf -> (B Np) Cf Hf Wf", Np=3),
            rearrange(grid, "B Np X Nd -> (B Np) () X Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        x = reduce(x, "(B Np) Cf () X -> B X Cf", Np=3, reduction="mean")
        x = rearrange(x, "B (Hp Wp) Cf -> B Cf Hp Wp", Hp=position_shape[-2], Wp=position_shape[-1])

        # return x[:, 0, :3, :, :]

        return x


class MaskTransform(DynamicTransform):
    def __init__(self,
                 mask_key):
        super().__init__()
        self.mask_key = mask_key

    def forward(self, x, batch):
        mask = batch[self.mask_key] == 0
        x = x * mask
        return x
