import torch
from einops import rearrange
from torch import nn

from pbd.scripts.playground import get_pca_map
from pbd.utils.datastructure import Batch
from pbd.utils.image_io import show_image


class MatchingLoss(nn.Module):
    def __init__(self,
                 loss_layer: nn.Module,
                 matching_threshold=1e-4,
                 subsampling=1024,
                 context="",
                 reduction="mean",
                 seed=0,
                 **loss_kwargs):
        super().__init__()
        self.loss_layer = loss_layer
        self.loss_kwargs = loss_kwargs
        self.context = context
        self.reduction = reduction
        self.seed = seed

        default_device = torch.empty(0).device
        self.generator_0 = torch.Generator(default_device).manual_seed(self.seed)
        self.generator_1 = torch.Generator(default_device).manual_seed(self.seed + 1)

        self.matching_threshold = matching_threshold
        self.subsampling = subsampling

    def forward(self, im1, im2, guide1, guide2, loss_info=None):
        """

        :param im1: C x H x W
        :param im2: C x H x W
        :param guide1: C x H x W
        :param guide2: C x H x W
        :param loss_info: dict
        :return:
        """
        assert guide1.ndim == 3, "Guide image 1 should be 3D"
        assert guide2.ndim == 3, "Guide image 2 should be 3D"
        assert im1.ndim == 3, "Image 1 should be 3D"
        assert im2.ndim == 3, "Image 2 should be 3D"

        # Collect inter-sample matches from the guide images
        guide1_shape = guide1.shape
        guide2_shape = guide2.shape
        guide1 = rearrange(guide1, 'C H W -> C (H W) 1')
        guide2 = rearrange(guide2, 'C H W -> C 1 (H W)')

        # Subsample the guide images
        if self.subsampling is not None:
            subsampling_mask1 = torch.randperm(guide1.shape[-2], generator=self.generator_0, device=guide1.device)[:self.subsampling]
            subsampling_mask2 = torch.randperm(guide2.shape[-1], generator=self.generator_1, device=guide2.device)[:self.subsampling]
        else:
            subsampling_mask1 = torch.arange(guide1.shape[-2], device=guide1.device)
            subsampling_mask2 = torch.arange(guide2.shape[-1], device=guide2.device)

        guide1 = guide1[:, subsampling_mask1, :]
        guide2 = guide2[:, :, subsampling_mask2]

        diff = (guide1 - guide2).square().sum(0)  # (H W) x (H W)
        matching_weight = diff < self.matching_threshold  # Binary mask for matching pixels
        matching_weight = rearrange(matching_weight, 'A B -> 1 1 A B')

        # Apply the loss
        im1 = rearrange(im1, 'C H W -> 1 C (H W) 1')
        im2 = rearrange(im2, 'C H W -> 1 C 1 (H W)')
        im1 = im1[:, :, subsampling_mask1, :]
        im2 = im2[:, :, :, subsampling_mask2]

        loss = self.loss_layer(input=im1, target=im2, **self.loss_kwargs)
        loss = torch.where(matching_weight.repeat(1, im1.shape[1], 1, 1), loss, (self.matching_threshold - loss).clamp(0))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Log the matching
        if loss_info is not None:
            # Add Matching Matrix
            matching = torch.zeros(1, 3, matching_weight.shape[-2] + 1, matching_weight.shape[-1] + 1, dtype=torch.float, device=im1.device)
            matching[:, :, 1:, 1:] = matching_weight.repeat(1, 3, 1, 1)
            matching[:, :, 1:, :1] = guide1
            matching[:, :, :1, 1:] = guide2

            # Add Matching summary
            num_matches = matching_weight.sum()

            extra_info = Batch({f"{self.context}matching": matching, f"{self.context}num_matches": num_matches})
            if "extra_loss_data" in loss_info:
                loss_info["extra_loss_data"].update(extra_info)
            else:
                loss_info["extra_loss_data"] = extra_info

        return loss


class NoisyWeightedMatchingLoss(MatchingLoss):
    def __init__(self,
                 loss_layer: nn.Module,
                 noise_level,
                 matching_threshold=1e-4,
                 subsampling=1024,
                 context="",
                 reduction="mean",
                 **loss_kwargs):
        super().__init__(loss_layer=loss_layer,
                         matching_threshold=matching_threshold,
                         subsampling=subsampling,
                         context=context,
                         reduction=reduction,
                         **loss_kwargs)
        self.noise_level = noise_level

    def forward(self, im1, im2, guide1, guide2, loss_info=None):
        """

        :param im1: C x H x W
        :param im2: C x H x W
        :param guide1: C x H x W
        :param guide2: C x H x W
        :param loss_info: dict
        :return:
        """
        assert guide1.ndim == 3, "Guide image 1 should be 3D"
        assert guide2.ndim == 3, "Guide image 2 should be 3D"
        assert im1.ndim == 3, "Image 1 should be 3D"
        assert im2.ndim == 3, "Image 2 should be 3D"

        # Collect inter-sample matches from the guide images
        guide1_shape = guide1.shape
        guide2_shape = guide2.shape
        guide1 = rearrange(guide1, 'C H W -> C (H W) 1')
        guide2 = rearrange(guide2, 'C H W -> C 1 (H W)')

        # Subsample the guide images
        if self.subsampling is not None:
            subsampling_mask1 = torch.randperm(guide1.shape[-2], generator=self.generator_0, device=guide1.device)[:self.subsampling]
            subsampling_mask2 = torch.randperm(guide2.shape[-1], generator=self.generator_1, device=guide2.device)[:self.subsampling]
        else:
            subsampling_mask1 = torch.arange(guide1.shape[-2], device=guide1.device)
            subsampling_mask2 = torch.arange(guide2.shape[-1], device=guide2.device)

        guide1 = guide1[:, subsampling_mask1, :]
        guide2 = guide2[:, :, subsampling_mask2]

        diff = (guide1 - guide2).square().sum(0)  # (H W) x (H W)
        matching_weight = diff < self.matching_threshold  # Binary mask for matching pixels
        matching_weight = rearrange(matching_weight, 'A B -> 1 1 A B')

        # Add noise to the matching weight
        all_matches = matching_weight.numel()
        num_matches = matching_weight.sum()
        num_no_matches = all_matches - num_matches
        fp_matches = int(num_matches * self.noise_level["fp"])
        fn_matches = int(num_matches * self.noise_level["fn"])
        if fp_matches + fn_matches > 0:
            noise = torch.rand(matching_weight.shape, device=matching_weight.device)
            matching_weight = torch.where(matching_weight, noise >= self.noise_level["fn"], noise > (1 - self.noise_level["fp"] * num_matches / num_no_matches))
        num_matches = matching_weight.sum()

        # Apply the loss
        im1 = rearrange(im1, 'C H W -> 1 C (H W) 1')
        im2 = rearrange(im2, 'C H W -> 1 C 1 (H W)')
        im1 = im1[:, :, subsampling_mask1, :]
        im2 = im2[:, :, :, subsampling_mask2]

        loss = self.loss_layer(input=im1, target=im2, **self.loss_kwargs)[matching_weight.repeat(1,im1.shape[1],1,1)]
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Log the matching
        if loss_info is not None:
            matching = torch.zeros(1, 3, matching_weight.shape[-2] + 1, matching_weight.shape[-1] + 1, dtype=torch.float, device=im1.device)
            matching[:, :, 1:, 1:] = matching_weight.repeat(1, 3, 1, 1)
            matching[:, :, 1:, :1] = guide1
            matching[:, :, :1, 1:] = guide2
            extra_info = Batch({f"{self.context}matching": matching,
                                f"{self.context}num_matches": num_matches})
            if "extra_loss_data" in loss_info:
                loss_info["extra_loss_data"].update(extra_info)
            else:
                loss_info["extra_loss_data"] = extra_info

        return loss


class WeightedMatchingLoss(MatchingLoss):
    def forward(self, im1, im2, guide1, guide2, loss_info=None):
        """

        :param im1: C x H x W
        :param im2: C x H x W
        :param guide1: C x H x W
        :param guide2: C x H x W
        :param loss_info: dict
        :return:
        """
        assert guide1.ndim == 3, "Guide image 1 should be 3D"
        assert guide2.ndim == 3, "Guide image 2 should be 3D"
        assert im1.ndim == 3, "Image 1 should be 3D"
        assert im2.ndim == 3, "Image 2 should be 3D"

        # Collect inter-sample matches from the guide images
        guide1_shape = guide1.shape
        guide2_shape = guide2.shape
        guide1 = rearrange(guide1, 'C H W -> C (H W) 1')
        guide2 = rearrange(guide2, 'C H W -> C 1 (H W)')

        # Subsample the guide images
        if self.subsampling is not None:
            subsampling_mask1 = torch.randperm(guide1.shape[-2], generator=self.generator_0, device=guide1.device)[:self.subsampling]
            subsampling_mask2 = torch.randperm(guide2.shape[-1], generator=self.generator_1, device=guide2.device)[:self.subsampling]
        else:
            subsampling_mask1 = torch.arange(guide1.shape[-2], device=guide1.device)
            subsampling_mask2 = torch.arange(guide2.shape[-1], device=guide2.device)

        guide1 = guide1[:, subsampling_mask1, :]
        guide2 = guide2[:, :, subsampling_mask2]

        diff = (guide1 - guide2).square().mean(0)  # (H W) x (H W)
        matching_weight = diff < self.matching_threshold  # Binary mask for matching pixels
        matching_weight = rearrange(matching_weight, 'A B -> 1 1 A B')

        # Apply the loss
        im1 = rearrange(im1, 'C H W -> 1 C (H W) 1')
        im2 = rearrange(im2, 'C H W -> 1 C 1 (H W)')
        im1 = im1[:, :, subsampling_mask1, :]
        im2 = im2[:, :, :, subsampling_mask2]

        loss = self.loss_layer(input=im1, target=im2, **self.loss_kwargs)[matching_weight.repeat(1,im1.shape[1],1,1)]
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Log the matching
        if loss_info is not None:
            # Add Matching Matrix
            matching = torch.zeros(1, 3, matching_weight.shape[-2] + 1, matching_weight.shape[-1] + 1,
                                   dtype=torch.float, device=im1.device)
            matching[:, :, 1:, 1:] = matching_weight.repeat(1, 3, 1, 1)
            matching[:, :, 1:, :1] = guide1
            matching[:, :, :1, 1:] = guide2

            # Add Matching summary
            num_matches = matching_weight.sum()

            extra_info = Batch({f"{self.context}matching": matching, f"{self.context}num_matches": num_matches})
            if "extra_loss_data" in loss_info:
                loss_info["extra_loss_data"].update(extra_info)
            else:
                loss_info["extra_loss_data"] = extra_info

        return loss


class WeightedMultiMatchingLoss(WeightedMatchingLoss):
    def __init__(self,
                 loss_layer: nn.Module,
                 matching_threshold_0=1e-4,
                 matching_threshold_1=1e-4,
                 subsampling=1024,
                 context="",
                 reduction="mean",
                 seed=0,
                 **loss_kwargs):
        super().__init__(loss_layer=loss_layer)
        self.loss_kwargs = loss_kwargs
        self.context = context
        self.reduction = reduction
        self.seed = seed

        default_device = torch.empty(0).device
        self.generator_0 = torch.Generator(default_device).manual_seed(self.seed)
        self.generator_1 = torch.Generator(default_device).manual_seed(self.seed + 1)

        self.matching_threshold_0 = matching_threshold_0
        self.matching_threshold_1 = matching_threshold_1
        self.subsampling = subsampling

    def forward(self, im1, im2, guide1_0, guide2_0, guide1_1, guide2_1, loss_info=None):
        assert guide1_0.ndim == 3, "Guide image 1 should be 3D"
        assert guide2_0.ndim == 3, "Guide image 2 should be 3D"
        assert guide1_1.ndim == 3, "Guide image 1 should be 3D"
        assert guide2_1.ndim == 3, "Guide image 2 should be 3D"
        assert im1.ndim == 3, "Image 1 should be 3D"
        assert im2.ndim == 3, "Image 2 should be 3D"

        # # PCA mappings
        # pca_stats = None
        # feature_map_1, pca_stats = get_pca_map(guide1_1.permute(1, 2, 0), guide1_1.shape[-2:], pca_stats=pca_stats,
        #                                        return_pca_stats=True)
        # feature_map_2, pca_stats = get_pca_map(guide2_1.permute(1, 2, 0), guide1_1.shape[-2:], pca_stats=pca_stats,
        #                                        return_pca_stats=True)

        # Collect inter-sample matches from the guide images
        guide1_shape = guide1_0.shape
        guide2_shape = guide2_0.shape
        guide1_0 = rearrange(guide1_0, 'C H W -> C (H W) 1')
        guide2_0 = rearrange(guide2_0, 'C H W -> C 1 (H W)')
        guide1_1 = rearrange(guide1_1, 'C H W -> C (H W) 1')
        guide2_1 = rearrange(guide2_1, 'C H W -> C 1 (H W)')

        # Subsample the guide images
        if self.subsampling is not None:
            subsampling_mask1 = torch.randperm(guide1_0.shape[-2], generator=self.generator_0, device=guide1_0.device)[:self.subsampling]
            subsampling_mask2 = torch.randperm(guide2_0.shape[-1], generator=self.generator_1, device=guide2_0.device)[:self.subsampling]
        else:
            subsampling_mask1 = torch.arange(guide1_0.shape[-2], device=guide1_0.device)
            subsampling_mask2 = torch.arange(guide2_0.shape[-1], device=guide2_0.device)

        guide1_0 = guide1_0[:, subsampling_mask1, :]
        guide2_0 = guide2_0[:, :, subsampling_mask2]
        guide1_1 = guide1_1[:, subsampling_mask1, :]
        guide2_1 = guide2_1[:, :, subsampling_mask2]

        diff_0 = (guide1_0 - guide2_0).square().mean(0)  # (H W) x (H W)
        diff_1 = (guide1_1 - guide2_1).square().mean(0)  # (H W) x (H W)
        matching_weight = diff_0 < self.matching_threshold_0  # Binary mask for matching pixels
        matching_weight &= diff_1 < self.matching_threshold_1  # Binary mask for matching pixels
        matching_weight = rearrange(matching_weight, 'A B -> 1 1 A B')

        # Apply the loss
        im1 = rearrange(im1, 'C H W -> 1 C (H W) 1')
        im2 = rearrange(im2, 'C H W -> 1 C 1 (H W)')
        im1 = im1[:, :, subsampling_mask1, :]
        im2 = im2[:, :, :, subsampling_mask2]

        loss = self.loss_layer(input=im1, target=im2, **self.loss_kwargs)[matching_weight.repeat(1, im1.shape[1], 1, 1)]
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Log the matching
        if loss_info is not None:
            # Add Matching Matrix
            matching = torch.zeros(1, 3, matching_weight.shape[-2] + 1, matching_weight.shape[-1] + 1,
                                   dtype=torch.float, device=im1.device)
            matching[:, :, 1:, 1:] = matching_weight.repeat(1, 3, 1, 1)
            # matching[:, :, 1:, :1] = guide1
            # matching[:, :, :1, 1:] = guide2

            # Add Matching summary
            num_matches = matching_weight.sum()

            extra_info = Batch({f"{self.context}matching": matching, f"{self.context}num_matches": num_matches})
            if "extra_loss_data" in loss_info:
                loss_info["extra_loss_data"].update(extra_info)
            else:
                loss_info["extra_loss_data"] = extra_info

        return loss


class FixedMatchingLoss(MatchingLoss):
    def forward(self, im1, im2, guide1, guide2, loss_info=None):
        """

        :param im1: C x H x W
        :param im2: C x H x W
        :param guide1: C x H x W
        :param guide2: C x H x W
        :param loss_info: dict
        :return:
        """
        assert guide1.ndim == 3, "Guide image 1 should be 3D"
        assert guide2.ndim == 3, "Guide image 2 should be 3D"
        assert im1.ndim == 3, "Image 1 should be 3D"
        assert im2.ndim == 3, "Image 2 should be 3D"

        # Collect inter-sample matches from the guide images
        guide1_shape = guide1.shape
        guide2_shape = guide2.shape
        guide1 = rearrange(guide1, 'C H W -> C (H W)')
        guide2 = rearrange(guide2, 'C H W -> C (H W)')

        # Subsample the guide images
        if self.subsampling is not None:
            subsampling_mask = torch.randperm(guide1.shape[-1], device=guide1.device)[:self.subsampling]
        else:
            subsampling_mask = torch.arange(guide1.shape[-1], device=guide1.device)

        guide1 = guide1[:, subsampling_mask]
        guide2 = guide2[:, subsampling_mask]

        diff = (guide1 - guide2).square().sum(0)  # (H W)
        matching_weight = diff < self.matching_threshold  # Binary mask for matching pixels
        matching_weight = rearrange(matching_weight, 'A -> 1 1 A')

        # Apply the loss
        im1 = rearrange(im1, 'C H W -> 1 C (H W)')
        im2 = rearrange(im2, 'C H W -> 1 C (H W)')
        im1 = im1[:, :, subsampling_mask]
        im2 = im2[:, :, subsampling_mask]

        loss = self.loss_layer(input=im1, target=im2, **self.loss_kwargs)
        loss = torch.where(matching_weight.repeat(1, im1.shape[1], 1, 1), loss, (self.matching_threshold - loss).clamp(0))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Log the matching
        if loss_info is not None:
            matching_weight = torch.diag(matching_weight.squeeze()).unsqueeze(0).unsqueeze(0)
            matching = torch.zeros(1, 3, matching_weight.shape[-2] + 1, matching_weight.shape[-1] + 1, dtype=torch.float, device=im1.device)
            matching[:, :, 1:, 1:] = matching_weight.repeat(1, 3, 1, 1)
            matching[0, :, 1:, 0] = guide1
            matching[0, :, 0, 1:] = guide2
            extra_info = Batch({f"{self.context}matching": matching})
            if "extra_loss_data" in loss_info:
                loss_info["extra_loss_data"].update(extra_info)
            else:
                loss_info["extra_loss_data"] = extra_info

        return loss