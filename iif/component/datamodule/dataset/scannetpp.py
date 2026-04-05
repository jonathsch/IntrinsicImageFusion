import csv
import glob
import json
import os
from typing import Optional, Callable, Union, Mapping, Any, List, Dict

import cv2
import einops
import mitsuba
import numpy as np
import torch
from torch.utils.data import Dataset

from iif.component.datamodule.transform.fixable import reset_transform_params
from iif.component.model.crf.emor import parse_emor_file
from iif.utils.datastructure import Batch, LoadableObjectCache
from iif.utils.image_io import load_exr_image, show_image, load_image
from iif.utils.logging import init_logger
from iif.utils.stage import TrainStage


class ScanNetPPDataset(Dataset):
    FEATURES = ["rays", 
                "rgbs_hdr", 
                "rgbs_ldr", 
                "albedo", 
                "diffuse_color", 
                "roughness", 
                "metallic", 
                "emission", 
                "segmentation", 
                "exposure",
                "crf",
                "index",
                "focal",
                "albedo_ref",
                "roughness_ref",
                "metallic_ref",
                "diffuse_shading_cache",
                "specular0_shading_cache",
                "specular1_shading_cache",
                "path",
                "per_image_segmentation"]

    def __init__(self,
                 root: str,
                 single_view_ref_root: Optional[str] = None,
                 shading_cache_root: Optional[str] = None,
                 stage: TrainStage = TrainStage.Training,
                 features_to_include: Optional[list] = None,
                 cache_size = None,
                 transform: Union[Optional[Callable], Mapping[str, Callable]] = None,
                 transforms_file: str = "transforms.json",
                 res_scale: float = 1.0,
                 subsampling_factor: int = 1,  # To take every n-th frame
                 *args, **kwargs):
        super().__init__()
        self.root = root
        self.module_logger = init_logger()

        self.single_view_ref_root = single_view_ref_root
        self.shading_cache_root = shading_cache_root
        self.transform = transform

        self._split_folder_path = None
        self._single_view_split_folder_path = None
        self.resolution = None

        self.transforms_file = transforms_file
        self.res_scale = res_scale

        self.subsampling_factor = subsampling_factor

        self.stage = stage if isinstance(stage, TrainStage) else TrainStage(stage)
        assert self.stage == TrainStage.Training, f"Invalid stage {self.stage}!"
        self.features_to_include = features_to_include if features_to_include is not None else self.FEATURES

        self.module_logger.debug(f"Loading {self.stage} dataset from {self.root}{' + ' + str(self.single_view_ref_root) if self.single_view_ref_root is not None else ''}!")
        self.data = self.load_dataset()
        self.module_logger.debug(f"Dataset {self.stage} from {self.root} loaded (length={len(self)})!")

        self.samples = LoadableObjectCache(self._load_sample, auto_load=True, max_size=cache_size)


    @property
    def split_folder_path(self) -> str:
        return self.root
    
    @property
    def single_view_split_folder_path(self) -> str:
        return self.single_view_ref_root

    def get_split_folder(self, root, stage: TrainStage) -> str:
        if root is None:
            return None
        
        if stage == TrainStage.Training:
            return os.path.join(root, "train")
        elif stage == TrainStage.Validation:
            return os.path.join(root, "val")
        elif stage == TrainStage.Test:
            self.module_logger.warning(
                f"Test split is not defined for {self.__class__.__name__}, using the val set!")
            return os.path.join(self.root, "val")
        else:
            raise ValueError(f"Invalid stage {stage}!")


    def load_dataset(self):
        data = Batch()
        data['samples_info'] = []

        # Read the metadata file
        with open(os.path.join(self.split_folder_path, self.transforms_file), 'r') as f:
            meta_data = json.load(f)

        # Collect the metadata
        # Focal length
        num_frames = int(np.ceil(len(meta_data['frames']) / self.subsampling_factor))

        fx, fy = meta_data['fl_x'], meta_data['fl_y']
        cx, cy = meta_data['cx'], meta_data['cy']
        img_h, img_w = meta_data['h'], meta_data['w']
        img_hw = (int(img_h*self.res_scale), int(img_w*self.res_scale))
        focal = fx

        self.resolution = img_hw
        K = np.array([
            [fx, 0, cx], 
            [0, fy, cy],
            [0, 0 ,1]
        ])
        K[:2] *= self.res_scale
        K = torch.from_numpy(K).float()

        # CRF
        _, vectors = parse_emor_file(inv=False)
        crf_mean = vectors[1]

        # Meta
        data['meta'] = Batch({
            "focal": focal,
            "exposure": torch.from_numpy(np.ones(num_frames, dtype=np.float32)),
            "crf": torch.from_numpy(np.stack([crf_mean, crf_mean, crf_mean]).astype(np.float32)),
            "per_image_segmentation": self._read_per_image_segmentation()
        })

        for cur_idx in range(len(meta_data['frames'])):
            if cur_idx % self.subsampling_factor != 0:
                continue

            frame = meta_data['frames'][cur_idx]
            data['samples_info'].append(Batch())

            # Collect the path
            cur_path = frame["file_path"].split('/')[-1]
            data["samples_info"][-1]["path"] = cur_path

            # Collect the paths for the features
            for feature in self.features_to_include:
                if feature == "rays":
                    # Ray origins and directions
                    c2w = np.array(frame['transform_matrix'])
                    c2w[:3, 1:3] *= -1 # to OpenCV
                    # c2w[:3, 1] *= -1 # to OpenCV
                    # c2w[:3, 2] *= -1 # to OpenCV
                    # c2w[2, 3] *= -1
                    c2w = c2w[:3]
                    c2w = torch.from_numpy(c2w).float()

                    rays_d = get_direction(K, img_hw)
                    rays_o, rays_d, dxdu, dydv = to_world(rays_d, c2w, True, K)

                    rays = torch.cat([rays_o, rays_d, dxdu, dydv], -1).permute(1,0).reshape(-1, *img_hw) 
                    data['samples_info'][-1][feature] = rays

                elif feature == "rgbs_ldr":
                    # RGB Images
                    image_path = os.path.join(self.split_folder_path, "images", cur_path)
                    data['samples_info'][-1][feature] = image_path

                elif feature == "rgbs_hdr":
                    # RGB Images
                    data['samples_info'][-1][feature] = None

                elif feature == "albedo":
                    # Albedo Images
                    data['samples_info'][-1][feature] = None

                elif feature == "diffuse_color":
                    # Diffuse Images
                    data['samples_info'][-1][feature] = None

                elif feature == "roughness":
                    # Roughness Images
                    data['samples_info'][-1][feature] = None

                elif feature == "metallic":
                    # Metallic Images
                    data['samples_info'][-1][feature] = None

                elif feature == "emission":
                    # Emission Images
                    data['samples_info'][-1][feature] = None

                elif feature == "segmentation":
                    # Segmentation Images
                    segmentation_path = os.path.join(self.split_folder_path, "seg", cur_path.replace(".JPG", ".png"))
                    if not os.path.exists(segmentation_path):
                        segmentation_path = segmentation_path.replace(".png", ".exr")
                    data['samples_info'][-1][feature] = segmentation_path

                elif feature == "albedo_ref":
                    # Reference Albedo Images
                    if self.single_view_ref_root is not None:
                        # Collect the reference predictions
                        assert self.single_view_split_folder_path is not None, "Single view reference root must be set!"
                        albedo_ref_paths = list(sorted(glob.glob(os.path.join(self.single_view_split_folder_path, "albedo", f'{cur_path.replace(".JPG", "").replace(".png", "")}_*.png'), recursive=False)))
                        data['samples_info'][-1][feature] = albedo_ref_paths
                
                elif feature == "roughness_ref":
                    # Reference Roughness Images
                    if self.single_view_ref_root is not None:
                        # Collect the reference predictions
                        assert self.single_view_split_folder_path is not None, "Single view reference root must be set!"
                        roughness_ref_paths = list(sorted(glob.glob(os.path.join(self.single_view_split_folder_path, "roughness", f'{cur_path.replace(".JPG", "").replace(".png", "")}_*.png'), recursive=False)))
                        data['samples_info'][-1][feature] = roughness_ref_paths

                elif feature == "metallic_ref":
                    # Reference Metallic Images
                    if self.single_view_ref_root is not None:
                        # Collect the reference predictions
                        assert self.single_view_split_folder_path is not None, "Single view reference root must be set!"
                        metallic_ref_paths = list(sorted(glob.glob(os.path.join(self.single_view_split_folder_path, "metallic", f'{cur_path.replace(".JPG", "").replace(".png", "")}_*.png'), recursive=False)))
                        data['samples_info'][-1][feature] = metallic_ref_paths

                elif feature == "diffuse_shading_cache":
                    # Collect the diffuse shading cache
                    assert self.shading_cache_root is not None, "Shading cache root must be set!"
                    shading_cache_diffuse_path = os.path.join(self.shading_cache_root, "diffuse", cur_path.replace(".JPG", ".exr").replace(".png", ".exr"))
                    data['samples_info'][-1][feature] = shading_cache_diffuse_path
                
                elif feature == "specular0_shading_cache":
                    # Collect the specular shading cache
                    assert self.shading_cache_root is not None, "Shading cache root must be set!"
                    shading_cache_specular0_path = list(sorted(glob.glob(os.path.join(self.shading_cache_root, "specular", f'{cur_path.replace(".JPG", "").replace(".png", "")}_0_*.exr'), recursive=False)))
                    data['samples_info'][-1][feature] = shading_cache_specular0_path

                elif feature == "specular1_shading_cache":
                    # Collect the specular shading cache
                    assert self.shading_cache_root is not None, "Shading cache root must be set!"
                    shading_cache_specular1_path = list(sorted(glob.glob(os.path.join(self.shading_cache_root, "specular", f'{cur_path.replace(".JPG", "").replace(".png", "")}_1_*.exr'), recursive=False)))
                    data['samples_info'][-1][feature] = shading_cache_specular1_path

        return data
    
    def _read_per_image_segmentation(self):
        if "per_image_segmentation" not in self.features_to_include:
            return None

        per_image_segmentation = Batch()
        object_ids_path = os.path.join(self.split_folder_path, "seg", "object_ids.txt")
        # object_ids_path = os.path.join(self.split_folder_path, "gt_seg", "object_ids.txt")
        object_ids = []
        with open(object_ids_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row_idx, row in enumerate(reader):
                if row_idx % self.subsampling_factor != 0:
                    continue
                object_ids.append(np.array([int(x) for x in row]))

        per_image_segmentation['object_ids'] = object_ids

        per_image_segmentation['start_idx'] = [0]
        for ids in object_ids[:-1]:
            per_image_segmentation['start_idx'].append(per_image_segmentation['start_idx'][-1] + len(ids))
        
        return per_image_segmentation

    def _load_sample(self, index: int) -> Any:
        sample = Batch(on_failure="keep")

        # Load the features
        image_path = None
        for feature in self.features_to_include:
            if feature in ("rgbs_hdr", "albedo", "diffuse_color", "roughness", "metallic", "emission"):
                # Unavailable features will be loaded as zero maps
                sample[feature] = np.zeros((*self.resolution, 3), dtype=np.float32)

                if feature in ("roughness", "metallic"):
                    # Take only a single value
                    sample[feature] = sample[feature][..., 0:1]

            elif feature in ("rgbs_ldr", "diffuse_shading_cache"):
                # Load image features
                image_path = self.data["samples_info"][index][feature]
                sample[feature] = load_image(image_path)

            elif feature in ("segmentation",):
                # Load image features
                image_path = self.data["samples_info"][index][feature]
                sample[feature] = load_image(image_path, dtype=np.uint8)[..., 0:1].astype(np.float32)
                # sample[feature] = load_image(image_path)[..., 0:1]

            elif feature in ("albedo_ref", "roughness_ref", "metallic_ref", "specular0_shading_cache", "specular1_shading_cache"):
                # Load reference image features
                ref_paths = self.data["samples_info"][index][feature]
                sample[feature] = np.stack([load_image(ref_path) for ref_path in ref_paths])

                if feature in ("roughness_ref", "metallic_ref"):
                    # Take only a single value
                    sample[feature] = sample[feature][..., 0:1]

            elif feature == "rays": 
                # Load ray features
                sample[feature] = self.data["samples_info"][index][feature]

            elif feature == "crf":
                # Load numpy features
                sample[feature] = self.data['meta'][feature]

            elif feature == "exposure":
                # Load numpy features
                sample[feature] = self.data['meta'][feature][index]

            elif feature == "focal":
                # Load focal length
                sample[feature] = self.data['meta']['focal']

            elif feature == "index":
                # Load index
                sample[feature] = torch.tensor([index], dtype=torch.long)

            elif feature == "per_image_segmentation":
                # Define a mapping from image_idx and object id to a unique segment id
                assert "segmentation" in self.features_to_include, "Segmentation must be included to use per_image_segmentation!"
                start_idx = self.data['meta']['per_image_segmentation']['start_idx'][index]
                object_ids = self.data['meta']['per_image_segmentation']['object_ids'][index]
                image_segment_idx = np.searchsorted(object_ids, sample["segmentation"])
                sample["per_image_segmentation"] = start_idx + image_segment_idx

            elif feature == "path":
                sample[feature] = self.data["samples_info"][index][feature]

        # Transform the features
        try:
            if self.transform is not None:
                reset_transform_params(self.transform)
                # Apply different transformation to the different features
                sample = self.transform(sample)
        except Exception as exc:
            self.module_logger.warning(f"Transform failed for sample {index} with error: {exc}")
            raise exc

        return sample

    def __len__(self) -> int:
        return len(self.data['samples_info'])


    def __getitem__(self, index: int) -> Any:
        # index = 0
        samples = self.samples[index]

        # Reshape the image-level features
        for feature in self.features_to_include:
            if feature in ("index", "focal"):
                # Reshape to (B, C, H, W,)
                samples[feature] = samples[feature][:, None, None].expand(-1, *self.resolution)
            elif feature == "crf":
                # Reshape to (B, C, H, W,)
                samples[feature] = samples[feature][:, :, None, None].expand(-1, -1, *self.resolution)
            elif feature in ("exposure"):
                samples[feature] = samples[feature][None, None, None].expand(-1, *self.resolution)

        return samples
    
    def get_image_from_path(self, path:str):
        # Find the index of the image
        for idx in range(len(self)):
            if self.data['samples_info'][idx]['path'] == path:
                return self[idx]
        raise ValueError(f"Image with path {path} not found in the dataset!")
    

def get_direction(k,img_hw):
    """ get camera ray direction (unormzlied)
        k: 3x3 camera intrinsics
        img_hw: image height and width
    """
    screen_y,screen_x = torch.meshgrid(torch.linspace(0.5,img_hw[0]-0.5,img_hw[0]),
                                torch.linspace(0.5,img_hw[1]-0.5,img_hw[1]))
    rays_d = torch.stack([
        (screen_x-k[0,2])/k[0,0],
        (screen_y-k[1,2])/k[1,1],
        torch.ones_like(screen_y)
    ],-1).reshape(-1,3)
    return rays_d


def to_world(rays_d,c2w,ray_diff,k):
    """ world sapce camera ray origin and direction
    Args:
        rays_d: HWx3 unormalized camera ray direction (local)
        c2w: 3x4 camera to world matrix
        ray_diff: True if return ray differentials
        k: 3x3 camera intrinsics
    Return:
        HWx3 camera origin
        HWx3 camera direction (unormzlied) if ray_diff==True
        HWx3 dxdu if ray_diff==True
        HWx3 dydv if ray_diff==True
    """
    rays_x = c2w[:,3:].T*torch.ones_like(rays_d)
    rays_d = rays_d@c2w[:3,:3].T
    if ray_diff:
        dxdu = torch.tensor([1.0/k[0,0],0,0])[None].expand_as(rays_d)@c2w[:3,:3].T
        dydv = torch.tensor([0,1.0/k[1,1],0])[None].expand_as(rays_d)@c2w[:3,:3].T
        return rays_x,rays_d,dxdu,dydv
    else:
        return rays_x,NF.normalize(rays_d,dim=-1)


def get_split_ids(n_total, split='train'):
    val_ids = [i*10 for i in range(16)]
    train_ids = [i for i in range(n_total) if i not in val_ids]
    if split == 'train':
        return train_ids
    else:
        return val_ids 


def get_ray_directions(H, W, focal):
    """ get camera ray direction
    Args:
        H,W: height and width
        focal: focal length
    x: left, y: up, z: forward
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords], indexing="ij")
    directions = \
        torch.stack([-(i-W/2)/focal, -(j-H/2)/focal, torch.ones_like(i)], -1) 

    return directions


def get_rays(directions, c2w, focal=None, flatten=False):
    """ world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:,:3]
    rays_d = directions @ R.T
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    if flatten:
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
        dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
        if flatten:
            dxdu = dxdu.view(-1,3)
            dydv = dydv.view(-1,3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return
    

class ScanNetPPRayDataset(ScanNetPPDataset):
    def __init__(self,
                 root: str,
                 single_view_ref_root: Optional[str] = None,
                 shading_cache_root: Optional[str] = None,
                 stage: TrainStage = TrainStage.Training,
                 features_to_include: Optional[list] = None,
                 chunk_size = None,
                 drop_last = True,
                 transform: Union[Optional[Callable], Mapping[str, Callable]] = None,
                 transforms_file: str = "transforms.json",
                 res_scale: float = 1.0,
                 subsampling_factor: int = 1,  # To take every n-th frame
                 *args, **kwargs):
        self.samples = None
        super(ScanNetPPDataset).__init__()
        self.root = root
        self.module_logger = init_logger()

        self.single_view_ref_root = single_view_ref_root
        self.shading_cache_root = shading_cache_root
        self.transform = transform

        self._split_folder_path = None
        self._single_view_split_folder_path = None

        self.transforms_file = transforms_file
        self.res_scale = res_scale
        
        self.subsampling_factor = subsampling_factor

        self.stage = stage if isinstance(stage, TrainStage) else TrainStage(stage)
        self.features_to_include = features_to_include if features_to_include is not None else self.FEATURES

        self.module_logger.debug(f"Loading {self.stage} dataset from {self.root}{' + ' + str(self.single_view_ref_root) if self.single_view_ref_root is not None else ''}!")
        self.data = self.load_dataset()
        self.chunk_size = chunk_size
        self.drop_last = drop_last
        self.samples = self.load_samples()
        self.reshuffle()
        self.module_logger.debug(f"Dataset {self.stage} from {self.root} loaded (length={len(self)})!")

    def load_samples(self):
        samples = []
        # Iterate over all samples and collect them
        for idx in range(super().__len__()):
            sample = self._load_sample(idx)
            samples.append(sample)
        samples = Batch.from_batch_list(*samples).map(lambda x: torch.stack(x, dim=0))

        # Reshape the image-level features
        B, _, H, W = samples['rays'].shape
        for feature in self.features_to_include:
            if feature in ("index", "focal"):
                samples[feature] = samples[feature][:, :, None, None].expand(-1, -1, H, W)
            elif feature in ("exposure"):
                samples[feature] = samples[feature][:, None, None, None].expand(-1, -1, H, W)
            elif feature in ("crf"):
                raise NotImplementedError("CRF handling not implemented yet!")

        # Reshape to independent rays
        samples = samples.map(lambda x: einops.rearrange(x, 'b ... c h w -> (b h w) ... c'), in_place=True)

        return samples
    
    def reshuffle(self):
        # resample camera ray batches
        # self.module_logger.debug(f"Reshuffling the dataset: {self.num_rays}!")
        self.idxs = torch.randperm(self.num_rays)

    @property
    def num_rays(self) -> int:
        """Total number of rays in the dataset."""
        # self.module_logger.debug(f"Samples: {self.samples}")
        return self.samples.shape[0].collapse()

    def __len__(self) -> int:
        num_chunks = self.num_rays // self.chunk_size
        if not self.drop_last:
            if self.num_rays % self.chunk_size != 0:
                num_chunks += 1
        return num_chunks

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")

        # Get the chunk
        b0 = index * self.chunk_size
        b1 = b0 + self.chunk_size
        idxs = self.idxs[b0:b1]

        batch = self.samples[idxs]
        return batch
    
