"""Alpha=0 camera coordinates, followed by full-frame resize without a letterbox.

Lens distortion is handled by an image remap, not by inverting every patch ray.
The downstream geometry is always pinhole. Already-rectified inputs bypass remap.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

MODE = "rectified_alpha0_resize_v1"
GEOMETRY = "rectified_pinhole_v1"


def contract(image_hw) -> dict[str, Any]:
    return {
        "mode": MODE, "image_hw": [int(x) for x in image_hw],
        "rectification": "legacy_alpha0_at_source_size_then_scale_K",
        "image_rectification": "when_needed; explicit rectified input bypasses remap",
        "camera_geometry": GEOMETRY, "crop": False, "letterbox": False,
        "alpha": 0.0, "pixel_coordinates": "integer_centres_half_pixel_resize",
        "resampling": "cached_native_alpha0_OpenCV_cubic_remap_then_PIL_bicubic_resize",
        "remap_border": "replicate_interpolation_boundary; no_black_canvas",
        "image_token_policy": "all_output_pixels_latents_tokens_supervised",
    }


def _coefficients(value) -> np.ndarray:
    if value is None:
        return np.zeros(5, dtype=np.float64)
    if isinstance(value, dict):
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y")
        n = 14 if any(k in value for k in names[12:]) else 12 if any(k in value for k in names[8:12]) else 8 if any(k in value for k in names[5:8]) else 5
        value = [value.get(name, 0.0) for name in names[:n]]
    co = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(co) not in (4, 5, 8, 12, 14) or not np.isfinite(co).all():
        raise ValueError("Expected finite OpenCV distortion coefficients of length 4, 5, 8, 12 or 14")
    return co


def _state(value: str | None) -> str:
    if value in (None, "raw", "distorted", "needs_rectification"):
        return "needs_rectification"
    if value in ("rectified", "undistorted", "already_rectified"):
        return "already_rectified"
    raise ValueError(f"Unknown image_state={value!r}; use raw or rectified")


def resize_matrix(source_wh, image_hw) -> np.ndarray:
    w, h = map(int, source_wh)
    oh, ow = map(int, image_hw)
    if min(w, h, oh, ow) <= 0:
        raise ValueError("Source and output image dimensions must be positive")
    sx, sy = ow / w, oh / h
    return np.array([[sx, 0., (sx - 1.) / 2.], [0., sy, (sy - 1.) / 2.], [0., 0., 1.]], dtype=np.float64)


@lru_cache(maxsize=8)
def _maps(source_wh, image_hw, k_values, coefficient_values, image_state):
    """Bounded per-process cache of calibrated remap arrays."""
    w, h = source_wh
    oh, ow = image_hw
    # Match the legacy ONCE K input precision and alpha=0 policy.
    k = np.array(k_values, dtype=np.float32).reshape(3, 3)
    co = np.array(coefficient_values, dtype=np.float32)
    transform = resize_matrix(source_wh, image_hw)
    needs_remap = image_state != "already_rectified" and bool(np.any(np.abs(co) > 1e-12))
    map_x = map_y = None
    roi = [0, 0, w, h]
    rectified_k = k.astype(np.float64)
    if needs_remap:
        import cv2
        rectified_k, roi = cv2.getOptimalNewCameraMatrix(k, co, (w, h), 0.0, (w, h))
    out_k = transform @ rectified_k
    if not np.isfinite(out_k).all() or min(out_k[0, 0], out_k[1, 1]) <= 0:
        raise ValueError("alpha=0 produced invalid camera intrinsics; inspect this camera calibration")
    outside = 0
    excess = 0.0
    if needs_remap:
        # Preserve the established image pipeline, including downsampling antialiasing:
        # native-size rectification first, then PIL bicubic full-image resize.
        map_x, map_y = cv2.initUndistortRectifyMap(k, co, None, rectified_k, (w, h), cv2.CV_32FC1)
        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise ValueError("Non-finite image rectification map")
        outside = int(((map_x < 0) | (map_x > w - 1) | (map_y < 0) | (map_y > h - 1)).sum())
        excess = float(max(0., -float(map_x.min()), float(map_x.max()) - (w - 1),
                           -float(map_y.min()), float(map_y.max()) - (h - 1)))
        map_x.setflags(write=False)
        map_y.setflags(write=False)
    record = {
        "policy": MODE, "geometry": GEOMETRY, "input_state": image_state,
        "rectification_applied": needs_remap, "source_wh": [w, h], "output_hw": [oh, ow],
        "raw_K": k.tolist(), "raw_distortion": co.tolist(), "rectified_K": rectified_k.tolist(),
        "resize_affine": transform.tolist(), "output_K": out_k.tolist(),
        "rectification_rotation": "identity; lidar_to_camera unchanged",
        "alpha0_valid_roi_not_cropped": [int(v) for v in roi],
        "native_map_centres_outside_source": outside, "map_max_outside_source_pixels": excess,
        "remap_border": "BORDER_REPLICATE", "image_supervision": "full", "nonlinear_ray_inverse": False,
    }
    out_k = out_k.astype(np.float32)
    out_k.setflags(write=False)
    return out_k, map_x, map_y, record


def camera_plan(source_wh, K, image_hw, distortion=None, image_state=None):
    k = np.asarray(K, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("K must be a finite pinhole 3x3 matrix with positive focal lengths")
    co = _coefficients(distortion)
    return _maps(tuple(map(int, source_wh)), tuple(map(int, image_hw)), tuple(k.ravel()), tuple(co), _state(image_state))


def prepare_rectified_image(im: Image.Image, K, image_hw, distortion=None, image_state=None, *, return_info=False):
    out_k, map_x, map_y, record = camera_plan(im.size, K, image_hw, distortion, image_state)
    oh, ow = map(int, image_hw)
    if map_x is None:
        output = im.convert("RGB").resize((ow, oh), Image.Resampling.BICUBIC)
    else:
        import cv2
        rgb = np.asarray(im.convert("RGB"))
        corrected = Image.fromarray(cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE))
        output = corrected.resize((ow, oh), Image.Resampling.BICUBIC)
    # Absence of LiDAR support is NOT absence of a target RGB pixel.
    result = output, out_k.copy(), np.ones((oh, ow), dtype=np.bool_)
    return (*result, dict(record)) if return_info else result
