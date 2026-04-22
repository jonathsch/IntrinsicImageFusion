import torch
import torch.nn.functional as F

from collections import defaultdict

import torchvision.transforms.functional as F
from einops import rearrange
from torchvision.transforms import RandomCrop

from iif.utils.logging import init_logger


class PCATransform(torch.nn.Module):
    FIXED_PARAMS = defaultdict(lambda: None)

    def __init__(self,
                 num_dims,
                 std_threshold=3,
                 fixing_id: str = None):
        super().__init__()

        self.num_dims = num_dims
        self.std_threshold = std_threshold
        self.fixing_id = fixing_id

        self.module_logger = init_logger()

    def reset_parameters(self):
        PCATransform.FIXED_PARAMS = defaultdict(lambda: None)

    def forward(self, img):
        # Get the potentially fixed parameters
        if self.fixing_id is None:
            pca_stats = self.get_params(img)
        else:
            if PCATransform.FIXED_PARAMS[self.fixing_id] is None:
                self.set_params(self.get_params(img))
            pca_stats = PCATransform.FIXED_PARAMS[self.fixing_id]

        return self.get_pca_map(img, pca_stats)

    def get_params(self, img):
        img = rearrange(img, "b c h w -> (b h w) c")
        return self.get_robust_pca(img, self.num_dims, self.std_threshold)

    def set_params(self, params):
        PCATransform.FIXED_PARAMS[self.fixing_id] = params

    def is_param_fixed(self):
        return PCATransform.FIXED_PARAMS[self.fixing_id] is not None

    @staticmethod
    def get_robust_pca(features: torch.Tensor,
                       lowrank_dim=3,
                       m: float = 2,
                       remove_first_component=False):
        # features: (N, C)
        # m: a hyperparam controlling how many std dev outside for outliers
        assert len(features.shape) == 2, "features should be (N, C)"
        reduction_mat = torch.pca_lowrank(features, q=lowrank_dim, niter=20)[2]
        colors = features @ reduction_mat

        if remove_first_component:
            colors_min = colors.min(dim=0).values
            colors_max = colors.max(dim=0).values
            tmp_colors = (colors - colors_min) / (colors_max - colors_min)
            fg_mask = tmp_colors[..., 0] < 0.2
            reduction_mat = torch.pca_lowrank(features[fg_mask], q=lowrank_dim, niter=20)[2]
            colors = features @ reduction_mat
        else:
            fg_mask = torch.ones_like(colors[:, 0]).bool()

        d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
        mdev = torch.median(d, dim=0).values
        s = d / mdev
        colors_tmp = colors.clone()
        try:
            # Ignore outliers
            colors_tmp[fg_mask][s < m] = torch.nan
        except:
            pass
        colors_min = colors_tmp.min(dim=0).values
        colors_max = colors_tmp.max(dim=0).values

        return reduction_mat, colors_min.to(reduction_mat), colors_max.to(reduction_mat)

    @staticmethod
    def get_pca_map(
            feature_map: torch.Tensor,
            pca_stats,
    ):
        feature_map = rearrange(feature_map, "c h w -> h w c")

        reduct_mat, color_min, color_max = pca_stats

        pca_color = feature_map @ reduct_mat
        pca_color = (pca_color - color_min) / (color_max - color_min)
        pca_color = pca_color.clamp(0, 1)
        pca_color = rearrange(pca_color, "h w c -> c h w")
        return pca_color
