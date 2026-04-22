import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from iif.utils.image_io import save_image
from iif.utils.logging import init_logger

class UVAtlasTexture(nn.Module):
    def __init__(self, 
                 texture_size=1024,
                 texture_channels={'albedo': 3, 'roughness': 1, 'metallic': 1, 'emission': 3},
                 texture_init={'albedo': 0., 'roughness': 1., 'metallic': 0.0, 'emission': 0.0},
                 texture_extensions={'albedo': '.png', 'roughness': '.png', 'metallic': '.png', 'emission': '.exr'}):
        super(UVAtlasTexture, self).__init__()

        self.texture_size = texture_size
        self.texture_channels = texture_channels
        self.texture_init = texture_init
        self.texture_extensions = texture_extensions

        # Initialize a learnable texture atlas
        params = {}
        for key, channels in self.texture_channels.items():
            params[key] = nn.Parameter(torch.ones(channels, texture_size, texture_size) * self.texture_init[key])
        self.params = nn.ParameterDict(params)

    def forward(self, uvs):
        """
        uvs: Tensor of shape (N, 2) with UV coordinates in [0, 1]
        Returns: Tensor of shape (N, 3) with RGB colors
        """
        # Scale UVs to the size of the texture atlas
        uvs_scaled = uvs * (self.texture_size - 1)
        u_coords = uvs_scaled[:, 0].long().clamp(0, self.texture_size - 1)
        v_coords = uvs_scaled[:, 1].long().clamp(0, self.texture_size - 1)

        # Sample colors from the texture atlas
        colors = {}
        for key in self.texture_channels.keys():
            texture_atlas = self.params[key]
            colors[key] = texture_atlas[:, v_coords, u_coords].permute(1, 0)  # Shape (N, 3)
        return colors
    
    def save(self, out_path):
        out_path = os.path.dirname(out_path)
        os.makedirs(out_path, exist_ok=True)

        for key in self.texture_channels.keys():
            texture_atlas = self.params[key].detach().cpu().permute(1, 2, 0).numpy()  # Shape (H, W, C)
            save_image(texture_atlas, f"{out_path}/{key}{self.texture_extensions[key]}")


class AggregatableUVAtlasTexture(nn.Module):
    def __init__(self, 
                 texture_size=1024,
                 texture_channels={'albedo': 3, 'roughness': 1, 'metallic': 1, 'emission': 3},
                 texture_init={'albedo': 0., 'roughness': 1., 'metallic': 0.0, 'emission': 0.0},
                 texture_extensions={'albedo': '.png', 'roughness': '.png', 'metallic': '.png', 'emission': '.exr'},
                 texture_inpainting={'albedo': True, 'roughness': True, 'metallic': True, 'emission': False}):
        super().__init__()
        self.module_logger = init_logger()
        self.texture_size = texture_size
        self.texture_channels = texture_channels
        self.texture_init = texture_init
        self.texture_extensions = texture_extensions
        self.texture_inpainting = texture_inpainting

        # Initialize a learnable texture atlas
        self.params = {}
        self.counts = {}
        for key, channels in self.texture_channels.items():
            self.params[key] = torch.zeros(channels, texture_size, texture_size)
            self.counts[key] = torch.zeros(1, texture_size, texture_size)
        
        self.params = nn.ParameterDict(self.params)
        self.counts = nn.ParameterDict(self.counts)

    def forward(self, uvs):
        """
        uvs: Tensor of shape (N, 2) with UV coordinates in [0, 1]
        Returns: Tensor of shape (N, 3) with RGB colors
        """
        # Scale UVs to the size of the texture atlas
        uvs_scaled = uvs * (self.texture_size - 1)
        u_coords = uvs_scaled[:, 0].long().clamp(0, self.texture_size - 1)
        v_coords = uvs_scaled[:, 1].long().clamp(0, self.texture_size - 1)

        # Sample colors from the texture atlas
        colors = {}
        for key in self.texture_channels.keys():
            texture_atlas = self.params[key]
            colors[key] = texture_atlas[:, v_coords, u_coords].permute(1, 0)  # Shape (N, 3)
        return colors

    def add(self, uvs, albedo, roughness, metallic, emission):
        """
        uvs: Tensor of shape (N, 2) with UV coordinates in [0, 1]
        albedo: Tensor of shape (N, 3) with RGB colors
        roughness: Tensor of shape (N, 1)
        metallic: Tensor of shape (N, 1)
        emission: Tensor of shape (N, 3)
        """
        # Scale UVs to the size of the texture atlas
        uvs_scaled = uvs * self.texture_size
        u_coords = torch.floor(uvs_scaled[:, 0]).long().clamp(0, self.texture_size - 1)
        v_coords = torch.floor(uvs_scaled[:, 1]).long().clamp(0, self.texture_size - 1)

        # Update the texture atlas
        self.params['albedo'][:, v_coords, u_coords] += albedo.permute(1, 0)
        self.params['roughness'][:, v_coords, u_coords] += roughness.permute(1, 0)
        self.params['metallic'][:, v_coords, u_coords] += metallic.permute(1, 0)
        self.params['emission'][:, v_coords, u_coords] += emission.permute(1, 0)

        # Update the counts
        self.counts['albedo'][:, v_coords, u_coords] += 1
        self.counts['roughness'][:, v_coords, u_coords] += 1
        self.counts['metallic'][:, v_coords, u_coords] += 1
        self.counts['emission'][:, v_coords, u_coords] += 1

    def average(self):
        for key in self.texture_channels.keys():
            count = self.counts[key]
            mask = (count == 0)
            self.params[key] /= count.clamp(1)
            self.params[key][mask.expand_as(self.params[key])] = self.texture_init[key]

            self.counts[key].zero_()
            self.counts[key][~mask] = 1
            # self.counts[key] += 1

    def inpaint(self):
        # Inpaints the missing regions with the nearest color values
        for key in self.texture_channels.keys():
            if not self.texture_inpainting[key]:
                continue
            texture = self.params[key]
            count = self.counts[key]
            mask = (count == 0).float()  # Shape (H, W)
            self.module_logger.info(f'Inpainting {key}, mask: {mask}')
            texture = inpaint_nearest(texture.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

            self.params[key].data = texture
    
    def save(self, out_path):
        os.makedirs(out_path, exist_ok=True)

        for key in self.texture_channels.keys():
            texture_atlas = self.params[key].detach().cpu().permute(1, 2, 0).numpy()  # Shape (H, W, C)
            save_image(texture_atlas, f"{out_path}/{key}{self.texture_extensions[key]}")

            count_atlas = self.counts[key].detach().cpu().squeeze(0).numpy()  # Shape (H, W)
            save_image(count_atlas / count_atlas.max(), f"{out_path}/{key}_count.png")


# def inpaint_nearest(image: torch.Tensor, mask: torch.Tensor, max_steps: int = 64):
#     """
#     Fast large-scale inpainting using nearest-value propagation.

#     Args:
#         image: (B, C, H, W) tensor
#         mask: (B, 1, H, W) tensor — 1 where masked (to fill), 0 elsewhere
#         max_steps: number of propagation steps (controls reach distance)

#     Returns:
#         Inpainted (B, C, H, W) tensor
#     """
#     image = image.clone()
#     mask = mask.clone().float()

#     B, C, H, W = image.shape
#     device = image.device

#     kernel = torch.ones(1, C, 3, 3, device=device)
#     known_mask = 1 - mask  # 1 = known, 0 = unknown

#     for _ in range(max_steps):
#         # Stop if everything is filled
#         if known_mask.all():
#             break

#         # Average neighbors (only within known regions)
#         neighbor_sum = F.conv2d(image * known_mask, kernel, padding=1, groups=1)
#         neighbor_count = F.conv2d(known_mask, kernel, padding=1, groups=1).clamp_min(1.0)
#         avg_neighbor = neighbor_sum / neighbor_count

#         # Fill only newly reachable pixels
#         new_known = (F.conv2d(known_mask, kernel[:1], padding=1) > 0).float()
#         to_fill = (new_known - known_mask).clamp_min(0.0)

#         image = image * (1 - to_fill) + avg_neighbor * to_fill
#         known_mask = torch.clamp(known_mask + to_fill, max=1.0)

#     return image


def inpaint_nearest(image: torch.Tensor, mask: torch.Tensor, max_iter: int = 100):
    """
    Inpaints masked regions in an image tensor using nearest-neighbor sampling.

    Args:
        image: (B, C, H, W) tensor — the image to inpaint
        mask: (B, 1, H, W) tensor — 1 where masked (to fill), 0 elsewhere
        max_iter: number of iterations to propagate known pixels

    Returns:
        Inpainted image tensor of the same shape.
    """
    # Ensure same device and type
    image = image.clone()
    mask = mask.clone().float()

    B, C, H, W = image.shape

    # Define a 3x3 averaging kernel for neighborhood propagation
    kernel = torch.ones(C, 1, 3, 3, device=image.device)

    for _ in tqdm(range(max_iter)):
        # Compute which pixels are still masked
        masked_pixels = (mask > 0.5).float()
        if masked_pixels.sum() == 0:
            break  # All filled

        # Average neighboring known pixels
        weighted_sum = F.conv2d(image * (1 - mask), kernel, padding=1, groups=image.shape[1])
        neighbor_count = F.conv2d((1 - mask), kernel, padding=1)

        # Avoid division by zero
        neighbor_count = torch.clamp(neighbor_count, min=1.0)
        filled = weighted_sum / neighbor_count

        # Only update masked pixels
        image = image * (1 - masked_pixels) + filled * masked_pixels

        # Shrink the mask gradually (simulate nearest-neighbor propagation)
        mask = 1 - (F.conv2d((1 - mask), kernel[:1], padding=1) > 0).float()

    return image