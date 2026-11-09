from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque

import torch
import torch.nn.functional as F


def spherical_indices(points, h=96, w=2048, fov_down=-25.0, fov_up=15.0):
    """Map XYZI points to the periodic sensor-centric spherical lattice."""
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"expected points [N,>=4], got {tuple(points.shape)}")
    xyz = points[..., :3]
    raw_r = xyz.norm(dim=-1)
    r = raw_r.clamp_min(1e-6)
    theta = torch.atan2(xyz[..., 1], xyz[..., 0])
    phi = torch.atan2(xyz[..., 2], xyz[..., :2].norm(dim=-1))
    col = torch.floor((theta + math.pi) / (2 * math.pi) * w).long().remainder(w)
    lo, hi = math.radians(fov_down), math.radians(fov_up)
    row = torch.floor((hi - phi) / (hi - lo) * h).long()
    valid = (
        (row >= 0)
        & (row < h)
        & torch.isfinite(points[..., :4]).all(-1)
        & (raw_r > 1e-6)
    )
    return row.clamp(0, h - 1), col, valid, r


def spherical_raster(points, h=96, w=2048, hits=2):
    """Vectorized nearest-K return layers plus four cell aggregates.

    The returned ID map always contains indices into the original ``points``
    tensor. It must not be replaced by a spherical cell index because that
    identity links the same physical LiDAR measurement across cameras.
    """
    if hits != 2:
        raise ValueError(
            "SweepFlow-MV spherical contract requires exactly two retained returns per cell"
        )
    row, col, valid, radius = spherical_indices(points, h, w)
    key = (row * w + col).long()
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    raster = points.new_zeros((hits, 6, h, w))
    idmap = torch.full((hits, h, w), -1, dtype=torch.long, device=points.device)
    if valid_ids.numel() == 0:
        return raster, idmap, points.new_zeros((4, h, w))

    # Stable lexicographic (cell, range) order and vectorized within-cell rank.
    by_range = valid_ids[torch.argsort(radius[valid_ids], stable=True)]
    order = by_range[torch.argsort(key[by_range], stable=True)]
    sorted_key = key[order]
    _, group_counts = torch.unique_consecutive(sorted_key, return_counts=True)
    group_starts = torch.cumsum(group_counts, 0) - group_counts
    repeated_starts = torch.repeat_interleave(group_starts, group_counts)
    within_rank = torch.arange(order.numel(), device=points.device) - repeated_starts
    keep = within_rank < hits
    kept_ids = order[keep]
    kept_rank = within_rank[keep].long()
    kept_row, kept_col = row[kept_ids], col[kept_ids]
    kept_r = radius[kept_ids]
    directions = points[kept_ids, :3] / kept_r[:, None]
    values = torch.cat(
        (
            torch.log1p(kept_r[:, None]),
            points[kept_ids, 3:4],
            directions,
            points.new_ones((kept_ids.numel(), 1)),
        ),
        dim=1,
    )
    raster[kept_rank, :, kept_row, kept_col] = values
    idmap[kept_rank, kept_row, kept_col] = kept_ids

    flat_size = h * w
    valid_key = key[valid]
    valid_r = radius[valid]
    valid_i = points[valid, 3]
    ones = torch.ones_like(valid_r)
    n = points.new_zeros(flat_size).scatter_add_(0, valid_key, ones)
    sum_r = points.new_zeros(flat_size).scatter_add_(0, valid_key, valid_r)
    sum_r2 = points.new_zeros(flat_size).scatter_add_(0, valid_key, valid_r.square())
    sum_i = points.new_zeros(flat_size).scatter_add_(0, valid_key, valid_i)
    sum_i2 = points.new_zeros(flat_size).scatter_add_(0, valid_key, valid_i.square())
    denom = n.clamp_min(1)
    var_r = (sum_r2 / denom - (sum_r / denom).square()).clamp_min(0)
    var_i = (sum_i2 / denom - (sum_i / denom).square()).clamp_min(0)
    gap = points.new_zeros(h, w)
    both = (idmap[0] >= 0) & (idmap[1] >= 0)
    gap[both] = radius[idmap[1, both]] - radius[idmap[0, both]]
    aggregates = torch.stack(
        (n.reshape(h, w), gap, var_r.reshape(h, w), var_i.reshape(h, w))
    )
    return raster, idmap, aggregates




def project(points_xyz, K, T, image_hw, distortion=None):
    """Project LiDAR-frame points using a LiDAR-to-camera transform."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"expected points_xyz [N,3], got {tuple(points_xyz.shape)}")
    p = torch.cat((points_xyz, torch.ones_like(points_xyz[:, :1])), 1) @ T.T
    z = p[:, 2]
    uvw = p[:, :3] @ K.T
    safe_z = torch.where(
        uvw[:, 2:3].abs() > 1e-8,
        uvw[:, 2:3],
        uvw.new_ones((uvw.shape[0], 1)),
    )
    uv = uvw[:, :2] / safe_z
    domain_valid = torch.ones(len(p), device=p.device, dtype=torch.bool)
    if distortion is not None:
        from .lens import project_camera_points, forward_domain_mask
        uv = project_camera_points(p[:, :3], K, distortion)
        domain_valid = forward_domain_mask(p[:, :3], distortion)
    h, w = image_hw
    valid = (
        (z > 1e-6)
        & domain_valid
        & torch.isfinite(uv).all(1)
        & torch.isfinite(z)
        & torch.isfinite(K).all()
        & torch.isfinite(T).all()
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    return uv, z, valid


def plucker_rays(K, T_lidar_camera, uv, distortion=None, *, return_info=False):
    """LiDAR-frame Pluecker ray ``[direction, camera_center x direction]``."""
    R = T_lidar_camera[:3, :3]
    t = T_lidar_camera[:3, 3]
    center = -(R.T @ t)
    pix = torch.cat((uv, torch.ones_like(uv[:, :1])), 1)
    d_cam = pix @ torch.linalg.inv(K).T
    valid = torch.ones(len(uv), device=uv.device, dtype=torch.bool)
    info = {"total_rays": len(uv), "verified_rays": len(uv), "unverified_rays": 0,
            "max_verified_pixel_error": 0., "rgb_supervision_mask_changed": False}
    if distortion is not None:
        from .lens import unproject_pixels
        if return_info:
            d_cam, valid, info = unproject_pixels(uv, K, distortion, return_validity=True, return_diagnostics=True)
        else:
            d_cam = unproject_pixels(uv, K, distortion)
    d = F.normalize(d_cam @ R, dim=-1)
    moment = center.expand_as(d).cross(d, dim=-1)
    rays = torch.cat((d, moment), -1)
    # A zero vector is an explicit unavailable ray, never a fabricated pinhole ray.
    rays = torch.where(valid[:, None], rays, torch.zeros_like(rays))
    return (rays, valid, info) if return_info else rays


def _continuous_grid_coordinates(uv, source_hw, target_hw):
    """Map source pixel-center coordinates to target pixel-center coordinates."""
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    u = (uv[:, 0] + 0.5) * (target_w / source_w) - 0.5
    v = (uv[:, 1] + 0.5) * (target_h / source_h) - 0.5
    return torch.stack((u, v), dim=1)


def bilinear_neighbors(xy, grid_hw):
    """Return four (row, col) neighbors and bilinear weights."""
    h, w = grid_hw
    x0 = torch.floor(xy[:, 0]).long()
    y0 = torch.floor(xy[:, 1]).long()
    dx = xy[:, 0] - x0
    dy = xy[:, 1] - y0
    cols = torch.stack((x0, x0 + 1, x0, x0 + 1), dim=1)
    rows = torch.stack((y0, y0, y0 + 1, y0 + 1), dim=1)
    weights = torch.stack(
        ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), dim=1
    )
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    weights = weights * inside.to(weights.dtype)
    return torch.stack(
        (rows.clamp(0, h - 1), cols.clamp(0, w - 1)), dim=-1
    ), weights


def dense_zbuffer(points_xyz, K, T, image_hw, distortion=None):
    """Nearest raw-sweep depth at each realized image pixel."""
    uv, depth, valid = project(points_xyz, K, T, image_hw, distortion=distortion)
    h, w = image_hw
    zbuf = depth.new_full((h * w,), float("inf"))
    ids = torch.nonzero(valid, as_tuple=False).flatten()
    if ids.numel():
        col = torch.floor(uv[ids, 0]).long().clamp(0, w - 1)
        row = torch.floor(uv[ids, 1]).long().clamp(0, h - 1)
        flat = row * w + col
        zbuf.scatter_reduce_(0, flat, depth[ids], reduce="amin", include_self=True)
    return zbuf.reshape(h, w), uv, depth, valid


def rasterize_local_pcd(
    points,
    K,
    T,
    image_hw,
    latent_hw,
    image_mask=None,
    z_tolerance_m=0.25,
    z_tolerance_ratio=0.01,
    distortion=None,
):
    """Occlusion-aware six-channel camera-local PCD grid at F8 resolution.

    Channels are log-depth, normalized intensity, unit LiDAR direction xyz,
    and bilinear support mass. Raw points first pass a full realized-image
    z-buffer, then bilinearly splat to the latent grid.
    """
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"expected normalized XYZI points [N,>=4], got {tuple(points.shape)}")
    zbuf, uv, depth, valid = dense_zbuffer(points[:, :3], K, T, image_hw, distortion=distortion)
    h, w = image_hw
    ids = torch.nonzero(valid, as_tuple=False).flatten()
    out_h, out_w = latent_hw
    result = points.new_zeros((6, out_h, out_w))
    if not ids.numel():
        return result, zbuf

    pix_col = torch.floor(uv[ids, 0]).long().clamp(0, w - 1)
    pix_row = torch.floor(uv[ids, 1]).long().clamp(0, h - 1)
    nearest = zbuf[pix_row, pix_col]
    tolerance = torch.maximum(
        depth[ids].new_full(depth[ids].shape, z_tolerance_m),
        nearest * z_tolerance_ratio,
    )
    visible = depth[ids] <= nearest + tolerance
    if image_mask is not None:
        visible &= image_mask[pix_row, pix_col].bool()
    ids = ids[visible]
    if not ids.numel():
        return result, zbuf

    latent_xy = _continuous_grid_coordinates(uv[ids], image_hw, latent_hw)
    rc, weights = bilinear_neighbors(latent_xy, latent_hw)
    flat = rc[..., 0] * out_w + rc[..., 1]
    weights = weights.reshape(-1)
    point_ids = ids[:, None].expand(-1, 4).reshape(-1)
    flat = flat.reshape(-1)
    keep = weights > 0
    weights, point_ids, flat = weights[keep], point_ids[keep], flat[keep]
    if not weights.numel():
        return result, zbuf

    radius = points[point_ids, :3].norm(dim=1).clamp_min(1e-6)
    feature = torch.cat(
        (
            torch.log1p(depth[point_ids, None].clamp_min(0)),
            points[point_ids, 3:4],
            points[point_ids, :3] / radius[:, None],
        ),
        dim=1,
    )
    support = points.new_zeros(out_h * out_w).scatter_add_(0, flat, weights)
    sums = points.new_zeros((5, out_h * out_w))
    sums.scatter_add_(
        1, flat[None].expand(5, -1), (feature * weights[:, None]).T
    )
    means = sums / support.clamp_min(1e-8)[None]
    result[:5] = means.reshape(5, out_h, out_w)
    result[5] = support.reshape(out_h, out_w).clamp_max(1.0)
    return result, zbuf


def patch_camera_condition(
    K,
    T,
    image_hw,
    latent_hw,
    patch_size,
    timestamp_delta,
    distortion=None,
    *,
    return_info=False,
):
    """11-D camera condition; ray validity is separate from RGB/token validity."""
    latent_h, latent_w = latent_hw
    if latent_h % patch_size or latent_w % patch_size:
        raise ValueError(
            f"latent grid {latent_hw} is not divisible by patch_size={patch_size}"
        )
    token_h, token_w = latent_h // patch_size, latent_w // patch_size
    ys = (torch.arange(token_h, device=K.device, dtype=K.dtype) + 0.5) * (
        image_hw[0] / token_h
    ) - 0.5
    xs = (torch.arange(token_w, device=K.device, dtype=K.dtype) + 0.5) * (
        image_hw[1] / token_w
    ) - 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    ray, ray_valid, ray_info = plucker_rays(K, T, uv, distortion=distortion, return_info=True)
    if not ray_valid.any():
        raise ValueError("No verified camera rays for this calibration; refusing geometry-free calibration fallback")
    h, w = image_hw
    normalized_k = torch.stack(
        (K[0, 0] / w, K[1, 1] / h, K[0, 2] / w, K[1, 2] / h)
    )
    dt = K.new_tensor(float(timestamp_delta)).expand(uv.shape[0], 1)
    condition = torch.cat((ray, dt, normalized_k.expand(uv.shape[0], -1)), dim=1)
    return (condition, ray_valid, ray_info) if return_info else condition




def _return_visibility(
    return_xyz,
    K,
    T,
    image_hw,
    image_mask,
    zbuffer,
    z_tolerance_m,
    z_tolerance_ratio,
    distortion=None,
):
    uv, depth, valid = project(return_xyz, K, T, image_hw, distortion=distortion)
    h, w = image_hw
    col = torch.floor(uv[:, 0]).long().clamp(0, w - 1)
    row = torch.floor(uv[:, 1]).long().clamp(0, h - 1)
    nearest = zbuffer[row, col]
    tol = torch.maximum(
        depth.new_full(depth.shape, z_tolerance_m), nearest * z_tolerance_ratio
    )
    valid &= depth <= nearest + tol
    valid &= image_mask[row, col].bool()
    confidence = torch.exp(
        -torch.relu(depth - nearest) / max(z_tolerance_m, 1e-6)
    )
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    return uv, depth, valid, confidence


def _balanced_return_selection(view_valid, confidence, ranges, budget):
    """Deterministic round-robin over camera-pair/range strata."""
    candidate = torch.nonzero(view_valid.sum(1) >= 2, as_tuple=False).flatten()
    if candidate.numel() <= budget:
        return candidate
    # One bulk transfer avoids one CUDA synchronization per candidate inside
    # the deterministic Python round-robin below.
    original_device = view_valid.device
    view_valid = view_valid.detach().cpu()
    confidence = confidence.detach().float().cpu()
    ranges = ranges.detach().float().cpu()
    candidate = candidate.cpu()
    groups = defaultdict(list)
    boundaries = ranges.new_tensor((15.0, 30.0, 50.0, 80.0))
    for rid in candidate.tolist():
        views = torch.nonzero(view_valid[rid], as_tuple=False).flatten().tolist()
        pair_scores = []
        for a, b in itertools.combinations(views, 2):
            pair_scores.append(
                (float(confidence[rid, a] + confidence[rid, b]), a, b)
            )
        _, a, b = max(pair_scores, key=lambda x: (x[0], -x[1], -x[2]))
        range_bin = int(torch.bucketize(ranges[rid], boundaries))
        score = float(confidence[rid].sum()) + 0.05 * len(views)
        groups[(a, b, range_bin)].append((score, rid))
    queues = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda x: (-x[0], x[1]))
        queues.append(deque(rid for _, rid in rows))
    selected = []
    while queues and len(selected) < budget:
        active = []
        for queue in queues:
            if queue and len(selected) < budget:
                selected.append(queue.popleft())
            if queue:
                active.append(queue)
        queues = active
    return torch.tensor(selected, dtype=torch.long, device=original_device)


def build_exact_return_graph(
    points,
    retained_ids,
    return_cells,
    return_ranks,
    Ks,
    Ts,
    image_masks,
    image_hw,
    token_hw,
    timestamp_delta,
    token_masks=None,
    budget=1024,
    tau_time=0.1,
    z_tolerance_m=0.25,
    z_tolerance_ratio=0.01,
    distortions=None,
):
    """Build the budgeted occlusion-aware exact-return multi-view graph.

    Only spherical-retained raw returns seen by at least two views are kept.
    Each projection is expanded into up to four bilinear transformer-token
    edges with bilinear, z-buffer, and timestamp confidence.
    """
    n_retained = retained_ids.numel()
    if (
        retained_ids.ndim != 1
        or return_cells.shape != (n_retained, 2)
        or return_ranks.shape != (n_retained,)
    ):
        raise ValueError(
            "retained_ids, return_cells, return_ranks must be [R], [R,2], [R]"
        )
    n_views = Ks.shape[0]
    xyz = points[retained_ids, :3]
    ranges = xyz.norm(dim=1).clamp_min(1e-6)
    all_uv, all_valid, all_confidence = [], [], []
    for view in range(n_views):
        distortion = None if distortions is None else distortions[view]
        zbuf, _, _, _ = dense_zbuffer(
            points[:, :3], Ks[view], Ts[view], image_hw, distortion=distortion
        )
        uv, _, valid, confidence = _return_visibility(
            xyz,
            Ks[view],
            Ts[view],
            image_hw,
            image_masks[view],
            zbuf,
            z_tolerance_m,
            z_tolerance_ratio,
            distortion=distortion,
        )
        all_uv.append(uv)
        all_valid.append(valid)
        time_conf = math.exp(
            -abs(float(timestamp_delta[view])) / max(tau_time, 1e-8)
        )
        all_confidence.append(confidence * time_conf)
    view_valid = torch.stack(all_valid, dim=1)
    confidence = torch.stack(all_confidence, dim=1)
    chosen = _balanced_return_selection(view_valid, confidence, ranges, int(budget))

    selected_ids = retained_ids[chosen]
    selected_cells = return_cells[chosen]
    selected_points = points[selected_ids]
    selected_ranges = ranges[chosen]
    # 10-D raw return contract: xyz, normalized intensity, log-range,
    # unit direction, retention rank, and support flag.
    return_features = torch.cat(
        (
            selected_points[:, :3],
            selected_points[:, 3:4],
            torch.log1p(selected_ranges[:, None]),
            selected_points[:, :3] / selected_ranges[:, None],
            return_ranks[chosen, None].to(points.dtype),
            points.new_ones((chosen.numel(), 1)),
        ),
        dim=1,
    )
    edge_rows, edge_weights = [], []
    # Vectorize all chosen returns within a view. The previous per-return loop
    # caused thousands of tiny GPU synchronizations for every cached scene.
    all_slots = torch.arange(chosen.numel(), device=points.device)
    for view in range(n_views):
        visible = view_valid[chosen, view]
        slots = all_slots[visible]
        source_indices = chosen[visible]
        if not slots.numel():
            continue
        token_xy = _continuous_grid_coordinates(
            all_uv[view][source_indices], image_hw, token_hw
        )
        neighbors, weights = bilinear_neighbors(token_xy, token_hw)
        token = neighbors[..., 0] * token_hw[1] + neighbors[..., 1]
        valid_edge = weights > 0
        if token_masks is not None:
            valid_edge &= token_masks[view, token].bool()
        weights = weights * valid_edge.to(weights.dtype)
        # Normalize each projection after clipping/padding-mask removal.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        valid_edge &= weights > 0
        if not valid_edge.any():
            continue
        flat_token = token[valid_edge]
        flat_slot = slots[:, None].expand_as(token)[valid_edge]
        edge_rows.append(
            torch.stack(
                (
                    torch.full_like(flat_token, view),
                    flat_token,
                    flat_slot,
                ),
                dim=1,
            )
        )
        per_projection_confidence = confidence[source_indices, view, None]
        edge_weights.append((weights * per_projection_confidence)[valid_edge])
    if edge_rows:
        edge_index = torch.cat(edge_rows, dim=0).long()
        edge_weight = torch.cat(edge_weights, dim=0)
    else:
        edge_index = torch.empty((0, 3), dtype=torch.long, device=points.device)
        edge_weight = points.new_empty((0,))
    # A projection near padding can have no valid patch-token neighbor. Prune
    # such returns after bilinear expansion so every final return still links
    # at least two distinct views, then compact return slots in the graph.
    if chosen.numel():
        linked = torch.zeros(
            (chosen.numel(), n_views), dtype=torch.bool, device=points.device
        )
        if edge_index.numel():
            linked[edge_index[:, 2], edge_index[:, 0]] = True
        keep_return = linked.sum(1) >= 2
        remap = torch.full(
            (chosen.numel(),), -1, dtype=torch.long, device=points.device
        )
        remap[keep_return] = torch.arange(
            int(keep_return.sum()), device=points.device
        )
        if edge_index.numel():
            keep_edge = keep_return[edge_index[:, 2]]
            edge_index = edge_index[keep_edge]
            edge_index[:, 2] = remap[edge_index[:, 2]]
            edge_weight = edge_weight[keep_edge]
        selected_ids = selected_ids[keep_return]
        selected_cells = selected_cells[keep_return]
        return_features = return_features[keep_return]
    return {
        "return_ids": selected_ids.long(),
        "return_cell": selected_cells.long(),
        "return_features": return_features,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "shared_return_count_before_budget": int((view_valid.sum(1) >= 2).sum()),
    }


