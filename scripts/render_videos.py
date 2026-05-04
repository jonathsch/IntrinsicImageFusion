"""Render multi-modality fly-through videos from indoor_synthetic scenes.

For a chosen scene, this script interpolates the existing keyframe camera
trajectory and renders six modalities per frame with Mitsuba 3:
    rgb, albedo, roughness, metallic, normal, depth

Each run produces its own output folder containing one MP4 per modality
plus a metadata.json describing the camera path and rendering settings.

Usage:
    python scripts/render_videos.py --scene kitchen --num-frames 24 --spp 4
"""

import argparse
import contextlib
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import drjit as dr
import mediapy
import mitsuba as mi
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# The shipped Mitsuba 3.5.0 build in this env exposes only:
#   scalar_rgb, scalar_spectral, cuda_ad_rgb, llvm_ad_rgb
# There's no plain `cuda_rgb`. We use `cuda_ad_rgb` but never enable
# gradients, so the AD machinery is dormant and adds negligible overhead.
mi.set_variant("cuda_ad_rgb")


SCENES = ("kitchen", "bedroom", "livingroom", "bathroom")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--data-root", default="data/indoor_synthetic")
    p.add_argument("--output-root", default="outputs/render_video")
    p.add_argument("--num-frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-depth", type=int, default=16,
                   help="path-tracing max depth (more bounces = better GI, slower)")
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--pose-source", choices=("trajectory", "transforms"),
                   default="trajectory")
    p.add_argument("--seed", type=int, default=0)

    # Quality knobs
    p.add_argument("--denoise", dest="denoise", action="store_true",
                   help="apply OptiX AI denoiser with albedo+normal guidance (default)")
    p.add_argument("--no-denoise", dest="denoise", action="store_false")
    p.set_defaults(denoise=True)
    p.add_argument("--rfilter", choices=("box", "tent", "gaussian"),
                   default="gaussian",
                   help="main-pass reconstruction filter. gaussian gives the best AA for "
                        "the albedo/normal/depth videos; the denoiser still handles "
                        "gaussian-filtered input well because the AOV guides carry most "
                        "of the structure. Use box only if you want razor-sharp denoiser input.")
    p.add_argument("--sampler", choices=("independent", "multijitter", "stratified", "orthogonal"),
                   default="multijitter",
                   help="sampler — multijitter/stratified converge faster than independent at low spp")
    p.add_argument("--exposure", type=float, default=1.0,
                   help="multiplicative scale on linear HDR before tonemap (>1 brightens)")
    p.add_argument("--crf", type=int, default=10,
                   help="x264 constant-rate-factor (lower = better quality, larger file). 10 is near-lossless.")
    return p.parse_args()


def load_keyframe_poses(scene_dir: Path, source: str):
    """Return (K, 4, 4) ndarray of keyframe camera-to-world matrices and the source path."""
    if source == "trajectory":
        path = scene_dir / "trajectory.json"
        with open(path) as f:
            data = json.load(f)
        keys = sorted(data.keys())
        poses = np.stack([np.asarray(data[k], dtype=np.float64) for k in keys], axis=0)
    elif source == "transforms":
        path = scene_dir / "train" / "transforms.json"
        with open(path) as f:
            data = json.load(f)
        frames = data["frames"]
        poses = np.stack(
            [np.asarray(fr["transform_matrix"], dtype=np.float64) for fr in frames],
            axis=0,
        )
    else:
        raise ValueError(f"unknown pose source: {source}")
    return poses, str(path)


def interpolate_poses(key_poses: np.ndarray, num_frames: int) -> np.ndarray:
    """SLERP rotations + lerp translations across keyframes."""
    K = key_poses.shape[0]
    if K < 2:
        raise ValueError(f"need at least 2 keyframes, got {K}")
    R_keys = Rotation.from_matrix(key_poses[:, :3, :3])
    t_keys = key_poses[:, :3, 3]
    times_keys = np.arange(K, dtype=np.float64)
    times_query = np.linspace(0.0, K - 1, num_frames, dtype=np.float64)

    slerp = Slerp(times_keys, R_keys)
    R_query = slerp(times_query).as_matrix()
    t_query = np.stack(
        [np.interp(times_query, times_keys, t_keys[:, i]) for i in range(3)],
        axis=-1,
    )
    out = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
    out[:, :3, :3] = R_query
    out[:, :3, 3] = t_query
    return out


def build_brdf_lookup(scene):
    """Build per-shape (roughness, metallic) lookup arrays.

    Heuristic from BSDF parameter keys (works for the indoor_synthetic scenes
    that use diffuse / roughconductor / roughplastic / thindielectric BSDFs):
        - has `.k` (eta+k pair):   metallic = 1
        - has `.alpha`:            roughness = sqrt(alpha)
        - has `.reflectance` only: diffuse → roughness = 1
        - has `.eta` only:         smooth dielectric → roughness = 0
        - fallback:                roughness = 0.5, metallic = 0

    Returns:
        roughness_lookup: shape (num_shapes + 1,) — index 0 is no-hit (= 0.0)
        metallic_lookup:  shape (num_shapes + 1,) — index 0 is no-hit (= 0.0)

    Mitsuba's `shape_index` AOV is 1-based: shape_index=0 means no intersection,
    shape_index=i (i>0) maps to scene.shapes()[i - 1].
    """
    params = mi.traverse(scene)
    shapes = scene.shapes()

    # Index BSDF param keys by BSDF id for fast lookup
    keys_by_bsdf = {}
    for k in params.keys():
        bsdf_id = k.split(".")[0]
        keys_by_bsdf.setdefault(bsdf_id, []).append(k)

    def attrs_for(bsdf_id):
        ks = keys_by_bsdf.get(bsdf_id, [])
        has_k = any(k.endswith(".k") or k.endswith(".k.value") for k in ks)
        has_alpha = any(k.endswith(".alpha") or k.endswith(".alpha.value") for k in ks)
        has_eta = any(k.endswith(".eta") or k.endswith(".eta.value") for k in ks)
        has_reflectance = any(
            k.endswith(".reflectance.value") or k.endswith(".reflectance.data")
            for k in ks
        )
        has_diffuse_refl = any(
            k.endswith(".diffuse_reflectance.value")
            or k.endswith(".diffuse_reflectance.data")
            for k in ks
        )
        alpha_val = None
        for k in ks:
            if k.endswith(".alpha") or k.endswith(".alpha.value"):
                v = params[k]
                try:
                    alpha_val = float(np.asarray(v).reshape(-1)[0])
                except Exception:
                    alpha_val = float(v)
                break
        return has_k, has_alpha, has_eta, has_reflectance, has_diffuse_refl, alpha_val

    rough = np.zeros(len(shapes) + 1, dtype=np.float32)
    metal = np.zeros(len(shapes) + 1, dtype=np.float32)
    for i, shp in enumerate(shapes):
        bsdf_id = shp.bsdf().id()
        has_k, has_alpha, has_eta, has_refl, has_diff_refl, alpha = attrs_for(bsdf_id)
        if has_k:
            metal[i + 1] = 1.0
        if has_alpha and alpha is not None:
            rough[i + 1] = float(math.sqrt(max(alpha, 0.0)))
        elif has_refl and not has_alpha and not has_eta:
            rough[i + 1] = 1.0  # plain diffuse
        elif has_eta and not has_alpha:
            rough[i + 1] = 0.0  # smooth dielectric / smooth conductor
        elif has_diff_refl and not has_alpha:
            rough[i + 1] = 1.0
        else:
            rough[i + 1] = 0.5  # unknown
    return rough, metal


def adjust_spp_for_sampler(spp: int, sampler: str) -> int:
    """multijitter and stratified samplers need spp = a*b for sub-pixel stratification.

    Mitsuba rounds up internally and prints a warning per render call; we round up
    here once at startup so the warning never fires.
    """
    if sampler not in ("multijitter", "stratified"):
        return spp
    s = int(math.isqrt(spp))
    if s * s >= spp:
        return s * s
    if s * (s + 1) >= spp:
        return s * (s + 1)
    return (s + 1) * (s + 1)


def make_sensor(pose: np.ndarray, fov: float, W: int, H: int, spp: int,
                rfilter: str, sampler: str = "independent"):
    return mi.load_dict({
        "type": "perspective",
        "fov": fov,
        "to_world": mi.ScalarTransform4f(pose.tolist()),
        "film": {
            "type": "hdrfilm", "width": W, "height": H,
            "pixel_format": "rgb",
            "rfilter": {"type": rfilter},
        },
        "sampler": {"type": sampler, "sample_count": spp},
    })


def denoise_rgb(denoiser, rgb: np.ndarray, albedo: np.ndarray, normal: np.ndarray,
                pose: np.ndarray) -> np.ndarray:
    """Run OptiX denoiser. ``rgb`` is HDR-linear; ``normal`` is in world space.

    The denoiser expects normals in the sensor's coordinate frame, so we pass
    ``to_sensor`` = world-to-camera transform (inverse of camera pose).
    """
    rgb_t = mi.TensorXf(np.ascontiguousarray(rgb, dtype=np.float32))
    albedo_t = mi.TensorXf(np.ascontiguousarray(np.clip(albedo, 0, None), dtype=np.float32))
    normal_t = mi.TensorXf(np.ascontiguousarray(normal, dtype=np.float32))
    world_to_cam = np.linalg.inv(pose)
    to_sensor = mi.Transform4f(world_to_cam.tolist())
    out = denoiser(rgb_t, denoise_alpha=False, albedo=albedo_t,
                   normals=normal_t, to_sensor=to_sensor)
    return np.asarray(out)


def to_uint8_rgb(img_rgb01: np.ndarray) -> np.ndarray:
    img = np.clip(img_rgb01, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def open_video_writer(path: str, shape_hw: tuple, fps: int, crf: int):
    """Open a mediapy ffmpeg writer with explicit CRF for near-lossless encoding."""
    return mediapy.VideoWriter(path, shape=shape_hw, fps=fps, crf=crf, codec="h264")


def gamma_srgb(x: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    return np.clip(np.power(np.clip(x * exposure, 0.0, None), 1.0 / 2.2), 0.0, 1.0)


def normalize_depth(depth: np.ndarray):
    """Map a single-channel depth image (H, W) to (H, W, 3) in [0, 1].

    Closer surfaces become brighter. Returns (image, (d_min, d_max)).
    Background pixels (depth == 0) are written as black.
    """
    valid = depth > 0
    if valid.any():
        d_min = float(depth[valid].min())
        d_max = float(depth[valid].max())
    else:
        d_min, d_max = 0.0, 1.0
    if d_max - d_min < 1e-8:
        d_max = d_min + 1e-8
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = 1.0 - (depth[valid] - d_min) / (d_max - d_min)
    return np.repeat(norm[..., None], 3, axis=-1), (d_min, d_max)


def main():
    args = parse_args()
    scene_dir = Path(args.data_root) / args.scene
    xml_path = scene_dir / "test.xml"
    if not xml_path.exists():
        print(f"error: scene XML not found at {xml_path}", file=sys.stderr)
        sys.exit(1)

    requested_spp = args.spp
    args.spp = adjust_spp_for_sampler(args.spp, args.sampler)
    if args.spp != requested_spp:
        print(f"note: rounding spp {requested_spp} → {args.spp} to satisfy "
              f"{args.sampler} sampler's a*b stratification requirement")

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_root) / f"{args.scene}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    print(f"loading scene {xml_path} at {args.width}x{args.height} ...")
    scene = mi.load_file(str(xml_path), resx=args.width, resy=args.height)

    print("building per-shape BRDF lookup ...")
    roughness_lookup, metallic_lookup = build_brdf_lookup(scene)
    print(f"  shapes: {len(roughness_lookup) - 1}, "
          f"metallic shapes: {int(metallic_lookup.sum())}")

    print(f"loading keyframe poses ({args.pose_source}) ...")
    key_poses, pose_path = load_keyframe_poses(scene_dir, args.pose_source)
    print(f"  {key_poses.shape[0]} keyframes from {pose_path}")
    poses = interpolate_poses(key_poses, args.num_frames)
    print(f"  interpolated to {poses.shape[0]} frames")

    aov_integrator = mi.load_dict({
        "type": "aov",
        "aovs": "albedo:albedo,nn:sh_normal,dd:depth",
        "rgb": {"type": "path", "max_depth": args.max_depth},
    })
    idx_integrator = mi.load_dict({
        "type": "aov",
        "aovs": "si:shape_index",
        "rgb": {"type": "path", "max_depth": 1},
    })

    # Reconstruction filter: box when denoising (per OptiX denoiser docs),
    # gaussian otherwise for smooth AA.
    rfilter = args.rfilter or ("box" if args.denoise else "gaussian")

    denoiser = None
    if args.denoise:
        print("initializing OptiX denoiser (albedo + normal guidance) ...")
        denoiser = mi.OptixDenoiser([args.width, args.height],
                                    albedo=True, normals=True)

    modalities = ("rgb", "albedo", "roughness", "metallic", "normal", "depth")
    depth_ranges = []

    print(f"rendering {args.num_frames} frames "
          f"(filter={rfilter}, sampler={args.sampler}, denoise={args.denoise}, crf={args.crf}) ...")

    with contextlib.ExitStack() as stack:
        writers = {
            m: stack.enter_context(open_video_writer(
                str(out_dir / f"{m}.mp4"),
                (args.height, args.width), args.fps, args.crf))
            for m in modalities
        }
        for i, pose in enumerate(poses):
            # Pass A: main RGB + albedo + sh_normal + depth at full spp
            sensor_a = make_sensor(pose, args.fov, args.width, args.height,
                                   args.spp, rfilter, args.sampler)
            img_a = mi.render(scene, sensor=sensor_a, integrator=aov_integrator,
                              spp=args.spp, seed=args.seed + i)
            arr_a = np.asarray(img_a)

            rgb = arr_a[..., 0:3]
            albedo = arr_a[..., 3:6]
            normal = arr_a[..., 6:9]
            depth = arr_a[..., 9]

            if denoiser is not None:
                rgb = denoise_rgb(denoiser, rgb, albedo, normal, pose)

            # Pass B: box-filtered shape_index at 1 spp for clean integer indices
            sensor_b = make_sensor(pose, args.fov, args.width, args.height, 1, "box")
            img_b = mi.render(scene, sensor=sensor_b, integrator=idx_integrator,
                              spp=1, seed=args.seed + i)
            shape_idx = np.asarray(img_b)[..., -1]
            shape_idx = np.rint(shape_idx).astype(np.int32)
            shape_idx = np.clip(shape_idx, 0, len(roughness_lookup) - 1)

            roughness = roughness_lookup[shape_idx]
            metallic = metallic_lookup[shape_idx]

            # Normalize each modality to [0, 1] H×W×3 for video writing
            rgb_v = gamma_srgb(rgb, args.exposure)
            albedo_v = gamma_srgb(albedo)
            normal_v = np.clip((normal + 1.0) * 0.5, 0.0, 1.0)
            rough_v = np.repeat(roughness[..., None], 3, axis=-1)
            metal_v = np.repeat(metallic[..., None], 3, axis=-1)
            depth_v, d_range = normalize_depth(depth)
            depth_ranges.append(d_range)

            writers["rgb"].add_image(to_uint8_rgb(rgb_v))
            writers["albedo"].add_image(to_uint8_rgb(albedo_v))
            writers["roughness"].add_image(to_uint8_rgb(rough_v))
            writers["metallic"].add_image(to_uint8_rgb(metal_v))
            writers["normal"].add_image(to_uint8_rgb(normal_v))
            writers["depth"].add_image(to_uint8_rgb(depth_v))

            if (i + 1) % 10 == 0 or i == 0 or i == args.num_frames - 1:
                print(f"  frame {i + 1}/{args.num_frames}")

    metadata = {
        "scene": args.scene,
        "timestamp": timestamp,
        "mitsuba_version": mi.__version__,
        "mitsuba_variant": mi.variant(),
        "drjit_version": dr.__version__,
        "resolution": [args.width, args.height],
        "fps": args.fps,
        "num_frames": args.num_frames,
        "spp": args.spp,
        "max_depth": args.max_depth,
        "fov_deg": args.fov,
        "rfilter": rfilter,
        "sampler": args.sampler,
        "denoise": bool(args.denoise),
        "exposure": args.exposure,
        "crf": args.crf,
        "pose_source": pose_path,
        "num_keyframes": int(key_poses.shape[0]),
        "interpolation": "slerp+lerp",
        "modalities": list(modalities),
        "depth_normalization_per_frame": [
            {"min": dmin, "max": dmax} for dmin, dmax in depth_ranges
        ],
        "depth_video_convention": "closer = brighter (1 - normalized)",
        "camera_poses": [pose.tolist() for pose in poses],
        "args": vars(args),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote metadata to {out_dir / 'metadata.json'}")
    print("done.")


if __name__ == "__main__":
    main()
