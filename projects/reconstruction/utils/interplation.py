# Modified by ADToolbox contributors, 2026: standalone reconstruction/edit/export extensions.
import os
import json
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib

from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear
import numpy as np
import torch
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import defaultdict
def project_point_cloud(point_cloud, other_extrinsic, intrinsic, H, W):
    
    device = point_cloud.device
    dtype = point_cloud.dtype
    N = point_cloud.shape[0]
    
    
    ones = torch.ones((N, 1), device=device, dtype=dtype)
    point_cloud_hom = torch.cat([point_cloud, ones], dim=-1)  # (N, 4)
    
    pts_cam_hom = (other_extrinsic @ point_cloud_hom.T).T  # (N, 4)
    pts_cam = pts_cam_hom[:, :3]  # (N, 3)

    depth = pts_cam[:, 2]  # (N,)
    valid = depth > 0
    if valid.sum() == 0:
        point_map = torch.full((H, W, 3), float('nan'), device=device, dtype=dtype)
        mask = torch.zeros((H, W), device=device, dtype=torch.bool)
        return point_map, mask

    valid_indices = valid.nonzero(as_tuple=True)[0]
    pts_cam_valid = pts_cam[valid_indices]           # (M, 3)
    world_valid   = point_cloud[valid_indices]         # (M, 3) 
    depth_valid   = pts_cam_valid[:, 2]                # (M,)

    pts_norm = pts_cam_valid[:, :2] / pts_cam_valid[:, 2:3]   # (M, 2)

    ones_norm = torch.ones((pts_norm.shape[0], 1), device=device, dtype=dtype)
    pts_norm_hom = torch.cat([pts_norm, ones_norm], dim=-1)     # (M, 3)
    uv_homog = (intrinsic @ pts_norm_hom.T).T  # (M, 3)
    u = uv_homog[:, 0]
    v = uv_homog[:, 1]

    u_int = torch.round(u).long()
    v_int = torch.round(v).long()

    inside = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    if inside.sum() == 0:
        point_map = torch.full((H, W, 3), float('nan'), device=device, dtype=dtype)
        mask = torch.zeros((H, W), device=device, dtype=torch.bool)
        return point_map, mask

    candidate_indices = inside.nonzero(as_tuple=True)[0]
    u_int       = u_int[candidate_indices]
    v_int       = v_int[candidate_indices]
    depth_valid = depth_valid[candidate_indices]
    world_valid = world_valid[candidate_indices]

    flat_pixel_idx = v_int * W + u_int  # (M_valid,)

    perm_depth = torch.argsort(depth_valid, stable=True)
    flat_pixel_idx = flat_pixel_idx[perm_depth]
    depth_valid    = depth_valid[perm_depth]
    world_valid    = world_valid[perm_depth]

    perm_idx = torch.argsort(flat_pixel_idx, stable=True)
    flat_pixel_idx = flat_pixel_idx[perm_idx]
    depth_valid    = depth_valid[perm_idx]
    world_valid    = world_valid[perm_idx]

    unique_flat_idx = torch.unique_consecutive(flat_pixel_idx)
    mask = torch.cat((torch.tensor([True], device=flat_pixel_idx.device), flat_pixel_idx[1:] != flat_pixel_idx[:-1]))
    unique_indices = mask.nonzero(as_tuple=True)[0]

    winner_world = world_valid[unique_indices]  # (num_pixels, 3)

    point_map_flat = torch.full((H * W, 3), float('nan'), device=device, dtype=dtype)
    mask_flat = torch.zeros((H * W,), device=device, dtype=torch.bool)

    point_map_flat[unique_flat_idx] = winner_world
    mask_flat[unique_flat_idx] = True

    point_map = point_map_flat.view(H, W, 3)
    mask = mask_flat.view(H, W)

    return point_map, mask

def filter_dense_points(points, radius=0.2, min_neighbors=10, min_total=50):
    if len(points) < min_total:
        return np.empty((0, 3))

    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    if len(points) < min_total:
        return np.empty((0, 3))
    
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius)
    mask = np.array([len(c) > min_neighbors for c in counts])
    filtered = points[mask]
    
    if len(filtered) < min_total:
        return np.empty((0, 3))
    
    return filtered


def interpolate_poses_intrinsics(extrinsics, intrinsics, interval, views, device=None):
    extrinsics_np = extrinsics.detach().cpu().numpy()
    intrinsics_np = intrinsics.detach().cpu().numpy()
    T = extrinsics.shape[0]
    if views == 1:
        # 单视角逻辑不变
        K = T
        extrinsics_out = []
        intrinsics_out = []
        for i in range(K - 1):
            pose0 = extrinsics_np[i]
            pose1 = extrinsics_np[i + 1]
            intr0 = intrinsics_np[i]
            intr1 = intrinsics_np[i + 1]
            R0 = pose0[:3, :3]
            R1 = pose1[:3, :3]
            t0 = pose0[:3, 3]
            t1 = pose1[:3, 3]
            r0 = scipy.spatial.transform.Rotation.from_matrix(R0)
            r1 = scipy.spatial.transform.Rotation.from_matrix(R1)
            slerp = scipy.spatial.transform.Slerp([0, 1], scipy.spatial.transform.Rotation.concatenate([r0, r1]))
            for j in range(interval):
                alpha = j / interval
                R_interp = slerp([alpha]).as_matrix()[0]
                t_interp = (1 - alpha) * t0 + alpha * t1
                pose_interp = np.eye(4)
                pose_interp[:3, :3] = R_interp
                pose_interp[:3, 3] = t_interp
                extrinsics_out.append(pose_interp)
                intr_interp = (1 - alpha) * intr0 + alpha * intr1
                intrinsics_out.append(intr_interp)
        extrinsics_out.append(extrinsics_np[-1])
        intrinsics_out.append(intrinsics_np[-1])
        extrinsics_interp = torch.tensor(extrinsics_out, dtype=torch.float32, device=device)
        intrinsics_interp = torch.tensor(intrinsics_out, dtype=torch.float32, device=device)
        return extrinsics_interp, intrinsics_interp
    else:
        frames_per_view = T // views
        extrinsics_out_views = [[] for _ in range(views)]
        intrinsics_out_views = [[] for _ in range(views)]
        for v in range(views):
            idxs = [v + i * views for i in range(frames_per_view)]
            ext_v = extrinsics_np[idxs]
            int_v = intrinsics_np[idxs]
            for i in range(frames_per_view - 1):
                pose0 = ext_v[i]
                pose1 = ext_v[i + 1]
                intr0 = int_v[i]
                intr1 = int_v[i + 1]
                R0 = pose0[:3, :3]
                R1 = pose1[:3, :3]
                t0 = pose0[:3, 3]
                t1 = pose1[:3, 3]
                r0 = scipy.spatial.transform.Rotation.from_matrix(R0)
                r1 = scipy.spatial.transform.Rotation.from_matrix(R1)
                slerp = scipy.spatial.transform.Slerp([0, 1], scipy.spatial.transform.Rotation.concatenate([r0, r1]))
                for j in range(interval):
                    alpha = j / interval
                    R_interp = slerp([alpha]).as_matrix()[0]
                    t_interp = (1 - alpha) * t0 + alpha * t1
                    pose_interp = np.eye(4)
                    pose_interp[:3, :3] = R_interp
                    pose_interp[:3, 3] = t_interp
                    extrinsics_out_views[v].append(pose_interp)
                    intr_interp = (1 - alpha) * intr0 + alpha * intr1
                    intrinsics_out_views[v].append(intr_interp)
            extrinsics_out_views[v].append(ext_v[-1])
            intrinsics_out_views[v].append(int_v[-1])
        extrinsics_out = []
        intrinsics_out = []
        for i in range(frames_per_view * interval + 1):
            for v in range(views):
                if i < len(extrinsics_out_views[v]):
                    extrinsics_out.append(extrinsics_out_views[v][i])
                    intrinsics_out.append(intrinsics_out_views[v][i])
        extrinsics_out = np.array(extrinsics_out)
        intrinsics_out = np.array(intrinsics_out)
        extrinsics_interp = torch.tensor(extrinsics_out, dtype=torch.float32, device=device)
        intrinsics_interp = torch.tensor(intrinsics_out, dtype=torch.float32, device=device)
        return extrinsics_interp, intrinsics_interp


def smooth_depth(depth_torch, kernel_size=7, sigma=1.0, iterations=2):
    device = depth_torch.device
    dtype = depth_torch.dtype

    mask = torch.isfinite(depth_torch)
    if mask.all():
        ks = kernel_size
        coords = torch.arange(ks, dtype=dtype, device=device) - (ks - 1) / 2.0
        gauss1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
        gauss1d = gauss1d / gauss1d.sum()
        kernel2d = gauss1d[:, None] * gauss1d[None, :]
        kernel = kernel2d.unsqueeze(0).unsqueeze(0)
        pad = ks // 2
        depth_in = depth_torch.unsqueeze(0).unsqueeze(0)
        sm = F.conv2d(depth_in, kernel.to(device=device, dtype=dtype), padding=pad)
        return sm.squeeze(0).squeeze(0)

    if mask.sum() == 0:
        return depth_torch
    depth0 = depth_torch.clone()
    depth0[~mask] = 0.0
    depth_in = depth0.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    mask_in = mask.float().unsqueeze(0).unsqueeze(0)
    ks = kernel_size
    coords = torch.arange(ks, dtype=dtype, device=device) - (ks - 1) / 2.0
    gauss1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    gauss1d = gauss1d / gauss1d.sum()
    kernel2d = gauss1d[:, None] * gauss1d[None, :]
    kernel = kernel2d.unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(device=device, dtype=dtype)
    pad = ks // 2

    eps = 1e-8
    for _ in range(max(1, iterations)):
        conv_d = F.conv2d(depth_in * mask_in, kernel, padding=pad)
        conv_m = F.conv2d(mask_in, kernel, padding=pad)
        filled = conv_d / (conv_m + eps)
        missing = (mask_in == 0)
        depth_in = depth_in.clone()
        depth_in[missing] = filled[missing]
        mask_in = (mask_in + (conv_m > 0).float()).clamp(max=1.0)
    smoothed = F.conv2d(depth_in, kernel, padding=pad)
    return smoothed.squeeze(0).squeeze(0)

def export_clusters(images, frames_clusters, cluster_assignments, tracked_objects, output_dir, interval):
    os.makedirs(output_dir, exist_ok=True)
    palette = np.asarray([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60]], dtype=np.float32) / 255.0
    records = []
    for t, frame_info in enumerate(frames_clusters):
        image = images[0, t].detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        overlay = image.copy()
        height, width = overlay.shape[:2]
        matched = 0
        for cluster_id, cluster in enumerate(frame_info["clusters"]):
            object_id = cluster_id if t == 0 else cluster_assignments[t].get(cluster_id)
            if object_id is not None:
                color = palette[int(object_id) % len(palette)]
                matched += t > 0
            else:
                color = np.asarray([0.6, 0.6, 0.6], dtype=np.float32)
            pixels = np.asarray(cluster["pixel_idxs"], dtype=np.int64)
            overlay[pixels // width, pixels % width] = 0.4 * overlay[pixels // width, pixels % width] + 0.6 * color
        plt.imsave(os.path.join(output_dir, f"cluster_frame_{t * interval:02d}.png"), overlay)
        records.append({"input_frame_index": t * interval, "clusters_detected": len(frame_info["clusters"]), "clusters_matched_to_frame_0": matched})
    with open(os.path.join(output_dir, "clusters.json"), "w") as handle:
        json.dump({"dynamic_probability_threshold": 0.95, "tracked_object_count_from_frame_0": len(tracked_objects), "frames": records}, handle, indent=2)


def export_track_head_tracks(images, tracks_xy, query_object_ids, output_dir, interval):
    """Save raw 2D Track Head trajectories over each input keyframe."""
    os.makedirs(output_dir, exist_ok=True)
    palette = np.asarray(
        [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
         [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60]],
        dtype=np.float32,
    ) / 255.0
    tracks = tracks_xy.detach().float().cpu().numpy()
    object_ids = np.asarray(query_object_ids, dtype=np.int64)

    for t in range(tracks.shape[0]):
        image = images[0, t].detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(image)
        for point_idx, object_id in enumerate(object_ids):
            color = palette[object_id % len(palette)]
            path = tracks[:t + 1, point_idx]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.8, alpha=0.7)
            ax.scatter(path[-1, 0], path[-1, 1], s=10, color=color)
        ax.set_xlim(0, image.shape[1] - 1)
        ax.set_ylim(image.shape[0] - 1, 0)
        ax.axis('off')
        fig.savefig(os.path.join(output_dir, f'track_frame_{t * interval:02d}.png'), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)

    displacements = np.linalg.norm(tracks - tracks[0:1], axis=-1)
    with open(os.path.join(output_dir, 'track_head_tracks.json'), 'w') as handle:
        json.dump({
            'source': 'SceneModel Track Head raw 2D trajectories',
            'input_frame_indices': [t * interval for t in range(tracks.shape[0])],
            'query_point_count': int(tracks.shape[1]),
            'displacement_from_frame_0_pixels': [
                {
                    'frame_index': t * interval,
                    'mean': float(displacements[t].mean()),
                    'median': float(np.median(displacements[t])),
                    'max': float(displacements[t].max()),
                }
                for t in range(tracks.shape[0])
            ],
            'object_ids_from_frame_0_dbscan': object_ids.tolist(),
        }, handle, indent=2)

def tracked_pixels_to_world(tracks_xy, depth, intrinsics, extrinsics):
    """Lift Track Head 2D pixel trajectories with the existing depth, K and world-to-camera poses."""
    T, N, _ = tracks_xy.shape
    H, W = depth.shape[-2:]
    dtype = tracks_xy.dtype
    tracks_xy = tracks_xy.clone()
    tracks_xy[..., 0].clamp_(0, W - 1)
    tracks_xy[..., 1].clamp_(0, H - 1)
    grid = tracks_xy.clone()
    grid[..., 0] = 2 * grid[..., 0] / max(W - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(H - 1, 1) - 1
    sampled_depth = F.grid_sample(
        depth[:, None].to(dtype), grid.view(T, N, 1, 2), mode="bilinear",
        padding_mode="zeros", align_corners=True
    ).squeeze(1).squeeze(-1)
    u, v = tracks_xy.unbind(dim=-1)
    fx, fy = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None]
    cx, cy = intrinsics[:, 0, 2, None], intrinsics[:, 1, 2, None]
    points_cam = torch.stack(((u - cx) * sampled_depth / fx, (v - cy) * sampled_depth / fy, sampled_depth), dim=-1)
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return torch.einsum("tij,tnj->tni", rotation.transpose(1, 2), points_cam - translation[:, None])


def export_motion_comparison(query_pixels, query_object_ids, tapip_coords, tapip_visibs, track_xy, track_world, waymo_flow, output_dir, interval):
    """Export raw outputs on identical frame-0 queries without changing motion estimation."""
    os.makedirs(output_dir, exist_ok=True)
    query_pixels_np = query_pixels.detach().float().cpu().numpy()
    track_xy_np = track_xy.detach().float().cpu().numpy()
    track_world_np = track_world.detach().float().cpu().numpy()
    tapip_coords_np = tapip_coords.detach().float().cpu().numpy()
    tapip_visibs_np = tapip_visibs.detach().cpu().numpy()

    flow_grid = waymo_flow.detach().float().to(track_xy.device).permute(0, 3, 1, 2)
    height, width = flow_grid.shape[-2:]
    sample_xy = track_xy.clone()
    sample_xy[..., 0].clamp_(0, width - 1)
    sample_xy[..., 1].clamp_(0, height - 1)
    grid = sample_xy.clone()
    grid[..., 0] = 2 * grid[..., 0] / max(width - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(height - 1, 1) - 1
    sampled_waymo_flow = F.grid_sample(
        flow_grid, grid.view(flow_grid.shape[0], -1, 1, 2), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(-1).permute(0, 2, 1).detach().cpu().numpy()

    np.savez_compressed(
        os.path.join(output_dir, "same_query_tracks.npz"),
        query_pixels_frame0=query_pixels_np,
        query_object_ids=np.asarray(query_object_ids, dtype=np.int64),
        input_frame_indices=np.arange(track_xy_np.shape[0], dtype=np.int64) * interval,
        tapip3d_world=tapip_coords_np,
        tapip3d_visible=tapip_visibs_np,
        track_head_pixels=track_xy_np,
        track_head_world=track_world_np,
        waymo_flow_at_track_head_pixels=sampled_waymo_flow,
    )
    with open(os.path.join(output_dir, "README.json"), "w") as handle:
        json.dump({
            "query_definition": "frame-0 GSRecon dynamic-map DBSCAN samples; shared by TAPIP3D and SceneModel Track Head",
            "input_frame_indices": (np.arange(track_xy_np.shape[0]) * interval).tolist(),
            "tapip3d_world": "TAPIP3D predicted world-coordinate trajectories, shape [T,N,3]",
            "track_head_pixels": "SceneModel Track Head predicted image coordinates, shape [T,N,2]",
            "track_head_world": "Track Head pixels lifted with GSRecon predicted depth, intrinsics and world-to-camera extrinsics, shape [T,N,3]",
            "waymo_flow_at_track_head_pixels": "depth_flows_4 sampled at Track Head positions, shape [T,N,3]; raw per-frame Waymo flow labels, not accumulated ground-truth trajectories",
            "no_visibility_or_confidence_filtering": True,
        }, handle, indent=2)


def interp_all(extrinsics, intrinsics, point_map, gs_map, dy_map, gs_conf, bg_mask, images,target_images, depth, track_model, interval=4,views=1, output_length=None, cluster_output_dir=None, track_visualization_dir=None, motion_comparison_dir=None, waymo_flow=None, motion_source='tapip3d', gsrecon_model=None):
    rng = np.random.default_rng(2026)
    depth = depth.squeeze(0).squeeze(-1)
    extrinsics_interp, intrinsics_interp = interpolate_poses_intrinsics(
        extrinsics, intrinsics, interval,views, device=images.device
    )
    depth_interp = None
    
    dy_map_sigmoid = dy_map.sigmoid()
    dy_map_binary = dy_map_sigmoid > 0.95
    dy_map_0 = dy_map_binary[0, 0]

    if dy_map_0.count_nonzero() > 500:
        cluster_eps_local = 0.015
        cluster_min_samples_local = 10
        filter_radius_local = 0.15
        filter_min_neighbors_local = 10
        filter_min_total_local = 50
        sample_per_cluster = 50
        match_max_distance = 0.5

        B, T, H, W, _ = point_map.shape
        assert B == 1, " batch=1"
        device = point_map.device

        frames_clusters = [None] * T
        for i in range(T):
            dynamic_mask = dy_map_binary[0, i, ...]
            if isinstance(dynamic_mask, torch.Tensor):
                dynamic_mask_np = dynamic_mask.detach().cpu().numpy()
            else:
                dynamic_mask_np = np.asarray(dynamic_mask)

            ys, xs = np.nonzero(dynamic_mask_np)
            if ys.size == 0:
                frames_clusters[i] = {'clusters': [], 'pts': np.empty((0,3)), 'labels': np.array([]), 'pixel_idxs': np.array([], dtype=np.int64)}
                continue
            pts_all = point_map[0, i, ys, xs]
            if isinstance(pts_all, torch.Tensor):
                pts_np = pts_all.detach().cpu().numpy().reshape(-1, 3)
            else:
                pts_np = np.asarray(pts_all).reshape(-1, 3)

            flat_idxs = (ys * W + xs).astype(np.int64)

            if pts_np.shape[0] == 0:
                frames_clusters[i] = {'clusters': [], 'pts': np.empty((0,3)), 'labels': np.array([]), 'pixel_idxs': np.array([], dtype=np.int64)}
                continue
            filtered = filter_dense_points(pts_np,
                                           radius=filter_radius_local,
                                           min_neighbors=filter_min_neighbors_local,
                                           min_total=filter_min_total_local)
            to_cluster = filtered if (filtered is not None and filtered.shape[0] > 0) else pts_np
            if to_cluster.shape[0] == pts_np.shape[0]:
                pixel_idxs_for_to_cluster = flat_idxs
            else:
                kd = cKDTree(pts_np)
                dists, nn_idx = kd.query(to_cluster, k=1)
                pixel_idxs_for_to_cluster = flat_idxs[nn_idx]

            if to_cluster.shape[0] == 0:
                frames_clusters[i] = {'clusters': [], 'pts': pts_np, 'labels': np.array([]), 'pixel_idxs': np.array([], dtype=np.int64)}
                continue

            clustering = DBSCAN(eps=cluster_eps_local, min_samples=cluster_min_samples_local).fit(to_cluster)
            labels = clustering.labels_
            unique_labels = sorted(set(labels))
            clusters = []
            for lab in unique_labels:
                if lab == -1:
                    continue
                mask_lab = (labels == lab)
                cluster_pts = to_cluster[mask_lab]
                cluster_pixel_idxs = pixel_idxs_for_to_cluster[mask_lab]
                centroid = cluster_pts.mean(axis=0)
                clusters.append({'pts': cluster_pts, 'centroid': centroid, 'pixel_idxs': cluster_pixel_idxs})
            frames_clusters[i] = {'clusters': clusters, 'pts': to_cluster, 'labels': labels, 'pixel_idxs': pixel_idxs_for_to_cluster}
        tracked_objects = {}

        if (track_model is not None or motion_source == 'track_head') and frames_clusters[0]['clusters']:
            first_frame_clusters = frames_clusters[0]['clusters']
            queries_list = []
            query_pixels_list = []
            cluster_indices = []
            for cidx, cluster in enumerate(first_frame_clusters):
                cluster_pts = cluster['pts']
                k = min(sample_per_cluster, cluster_pts.shape[0])
                if k == 0:
                    continue
                choice = rng.choice(np.arange(cluster_pts.shape[0]), k, replace=False)
                sampled_pts = cluster_pts[choice]
                sampled_pixel_idxs = cluster['pixel_idxs'][choice]
                frame_idx_col = np.zeros((k, 1), dtype=np.float32)
                query = np.concatenate([frame_idx_col, sampled_pts.astype(np.float32)], axis=1)
                queries_list.append(query)
                query_pixels_list.append(np.stack((sampled_pixel_idxs % W, sampled_pixel_idxs // W), axis=1).astype(np.float32))
                cluster_indices.append(cidx)

            if len(queries_list) > 0:
                if motion_source == 'track_head':
                    if gsrecon_model is None:
                        raise ValueError('gsrecon_model is required when motion_source=track_head')
                    query_pixels = torch.tensor(np.concatenate(query_pixels_list, axis=0), dtype=torch.float32, device=device)
                    track_predictions = gsrecon_model(images, query_points=query_pixels)
                    tracks_xy = track_predictions['track'][0]
                    if track_visualization_dir is not None:
                        query_object_ids = np.concatenate([np.full(len(points), cluster_id, dtype=np.int64) for points, cluster_id in zip(query_pixels_list, cluster_indices)])
                        export_track_head_tracks(images, tracks_xy, query_object_ids, track_visualization_dir, interval)
                    coords_out = tracked_pixels_to_world(tracks_xy, depth, intrinsics, extrinsics)
                    visibs = None
                    del track_predictions, tracks_xy, query_pixels
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    queries_np = np.concatenate(queries_list, axis=0)
                    queries_torch = torch.tensor(queries_np, dtype=torch.float32, device=device)

                    device_local = device
                    video_arg = images[0].to(device_local).float()
                    depth_arg = depth.to(device_local).float() if depth is not None else None
                    intrinsics_arg = intrinsics.to(device_local).float()
                    extrinsics_arg = extrinsics.to(device_local).float()
                    queries_arg = queries_torch.to(device_local).float()

                    coords_out = None
                    visibs = None
                    try:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        coords_out, visibs = inference(
                            model=track_model,
                            video=video_arg,
                            depths=depth_arg,
                            intrinsics=intrinsics_arg,
                            extrinsics=extrinsics_arg,
                            query_point=queries_arg,
                            num_iters=6,
                            grid_size=16,
                            bidrectional=False
                        )
                    except Exception as e_float:
                        try:
                            model_device = next(track_model.parameters()).device if any(p.numel() for p in track_model.parameters()) else torch.device("cpu")
                            track_model_cpu = track_model.to("cpu")
                            coords_out, visibs = inference(
                                model=track_model_cpu,
                                video=video_arg.cpu(),
                                depths=(depth_arg.cpu() if depth_arg is not None else None),
                                intrinsics=intrinsics_arg.cpu(),
                                extrinsics=extrinsics_arg.cpu(),
                                query_point=queries_arg.cpu(),
                                num_iters=6,
                                grid_size=16,
                                bidrectional=False
                            )
                            try:
                                track_model.to(model_device)
                            except Exception:
                                pass
                        except Exception as e_cpu:
                            try:
                                if torch.cuda.is_available():
                                    with torch.autocast("cuda", dtype=torch.bfloat16):
                                        coords_out, visibs = inference(
                                            model=track_model,
                                            video=video_arg,
                                            depths=depth_arg,
                                            intrinsics=intrinsics_arg,
                                            extrinsics=extrinsics_arg,
                                            query_point=queries_arg,
                                            num_iters=6,
                                            grid_size=16,
                                            bidrectional=False
                                        )
                            except Exception as e_last:
                                raise
                    finally:
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                if motion_comparison_dir is not None:
                    if motion_source != 'tapip3d':
                        raise ValueError('export_motion_comparison requires motion_source=tapip3d')
                    query_pixels = torch.tensor(np.concatenate(query_pixels_list, axis=0), dtype=torch.float32, device=device)
                    query_object_ids = np.concatenate([np.full(len(points), cluster_id, dtype=np.int64) for points, cluster_id in zip(query_pixels_list, cluster_indices)])
                    track_predictions = gsrecon_model(images, query_points=query_pixels)
                    track_xy = track_predictions['track'][0]
                    track_world = tracked_pixels_to_world(track_xy, depth, intrinsics, extrinsics)
                    export_track_head_tracks(images, track_xy, query_object_ids, os.path.join(motion_comparison_dir, 'track_head'), interval)
                    export_motion_comparison(
                        query_pixels, query_object_ids, coords_out, visibs, track_xy, track_world,
                        waymo_flow, motion_comparison_dir, interval,
                    )
                    del track_predictions, query_pixels, track_xy, track_world
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if coords_out is not None:
                    try:
                        coords_all = coords_out.detach().cpu().numpy()
                    except Exception:
                        coords_all = coords_out
                    visibs_np = visibs.detach().cpu().numpy() if visibs is not None else None
                else:
                    coords_all = None
                    visibs_np = None
                

                if coords_all is not None:
                    acc = 0
                    for cidx in cluster_indices:
                        k = queries_list[cluster_indices.index(cidx)].shape[0]
                        pts_tracked = coords_all[:, acc:acc+k, :]
                        pt_start = pts_tracked[0, :, :]
                        
                        centroids_per_frame = pts_tracked.mean(axis=1)
                        motion_vectors_per_frame = []
                        
                        for t_idx in range(1, T):
                            centroid_t = centroids_per_frame[t_idx - 1]
                            centroid_next = centroids_per_frame[t_idx]
                            motion_vec = centroid_next - centroid_t  # (3,)
                            motion_vectors_per_frame.append(motion_vec)
                        
                        tracked_objects[cidx] = {
                            'pred_centroids': centroids_per_frame,
                            'motion_vectors': motion_vectors_per_frame,
                            'queries_start': pt_start
                        }
                        acc += k
        cluster_assignments = [{} for _ in range(T)]
        obj_to_cluster = {oid: {0: 0} for oid in tracked_objects.keys()} 
        if tracked_objects:
            for t in range(1, T):
                frame_info = frames_clusters[t]
                if not frame_info['clusters']:
                    continue
                centroids_t = np.stack([c['centroid'] for c in frame_info['clusters']])
                obj_pred_centroids = {obj_id: obj_info['pred_centroids'][t] for obj_id, obj_info in tracked_objects.items() if not np.any(np.isnan(obj_info['pred_centroids'][t]))}
                if not obj_pred_centroids:
                    continue
                obj_ids = list(obj_pred_centroids.keys())
                preds = np.stack([obj_pred_centroids[oid] for oid in obj_ids])
                cost = np.linalg.norm(preds[:, None, :] - centroids_t[None, :, :], axis=2)
                assigned_obj = set()
                for cidx in range(centroids_t.shape[0]):
                    obj_dist = cost[:, cidx]
                    min_idx = np.argmin(obj_dist)
                    min_dist = obj_dist[min_idx]
                    if min_dist < match_max_distance:
                        oid = obj_ids[min_idx]
                        if cidx not in cluster_assignments[t]:
                            cluster_assignments[t][cidx] = oid
                            assigned_obj.add(oid)
                            obj_to_cluster[oid][t] = cidx
            if cluster_output_dir is not None:
                export_clusters(images, frames_clusters, cluster_assignments, tracked_objects, cluster_output_dir, interval)

            B_new = point_map.shape[0]
            T_new = (T - 1) * interval + 1
            H_new = point_map.shape[2]
            W_new = point_map.shape[3]
            point_map_interp = torch.full((B_new, T_new, H_new, W_new, 3), float('nan'),
                                           dtype=point_map.dtype, device=point_map.device)
            dy_map_interp = torch.full((B_new, T_new, H_new, W_new), float('nan'),
                                        dtype=dy_map.dtype, device=dy_map.device)
            if output_length is not None:
                last_motion = torch.zeros_like(point_map[:, -1])
                last_motion_mask = torch.zeros(point_map.shape[:1] + point_map.shape[2:4], dtype=torch.bool, device=point_map.device)
            point_obj_id = [dict() for _ in range(T)]

        for t in range(T):
            frame_info = frames_clusters[t]
            if not frame_info['clusters']:
                continue
            for cidx, cluster in enumerate(frame_info['clusters']):
                if t == 0:
                    oid = cidx
                else:
                    oid = cluster_assignments[t].get(cidx, None)
                if oid is None:
                    continue
                cluster_pixel_idxs = cluster.get('pixel_idxs', None)
                if cluster_pixel_idxs is None or cluster_pixel_idxs.size == 0:
                    continue
                for flat_idx in cluster_pixel_idxs:
                    point_obj_id[t][int(flat_idx)] = oid
                    if output_length is not None and t == T - 1 and oid in tracked_objects:
                        v_src = int(flat_idx) // W
                        u_src = int(flat_idx) % W
                        last_motion[0, v_src, u_src] = torch.as_tensor(tracked_objects[oid]['motion_vectors'][t - 1], dtype=point_map.dtype, device=device)
                        last_motion_mask[0, v_src, u_src] = True
        
        for t in range(T):
            frame_info = frames_clusters[t]
            dynamic_mask_t = dy_map_binary[0, t, ...]  # (H, W)
            point_map_interp[0, t*interval, :, :, :] = point_map[0, t, :, :, :]
            dy_map_interp[0, t*interval, :, :] = dy_map[0, t, :, :]
            
            if t < T - 1:
                frame_info_next = frames_clusters[t + 1]
                
                for j in range(1, interval):
                    alpha = float(j) / float(interval)
                    t_interp = t * interval + j
                    
                    point_map_interp[0, t_interp, :, :, :] = float('nan')
                    dy_map_interp[0, t_interp, :, :] = 0
                    src_pts = point_map[0, t, :, :, :]
                    valid_src = torch.isfinite(src_pts).all(dim=-1) 
                    if valid_src.any():
                        point_map_interp[0, t_interp][valid_src] = src_pts[valid_src]
                    if frame_info['clusters']:
                        for cidx, cluster_t in enumerate(frame_info['clusters']):
                            if t == 0:
                                oid = cidx
                            else:
                                oid = cluster_assignments[t].get(cidx, None)
                            
                            if oid is None or oid not in tracked_objects:
                                continue
                            
                            motion_vec_t_to_next = tracked_objects[oid]['motion_vectors'][t]
                            
                            cluster_pixel_idxs = cluster_t.get('pixel_idxs', np.array([], dtype=np.int64))
                            if cluster_pixel_idxs is None or cluster_pixel_idxs.size == 0:
                                continue

                            for flat_idx in cluster_pixel_idxs:
                                flat_idx = int(flat_idx)
                                v_src = flat_idx // W
                                u_src = flat_idx % W
                                
                                pt_world_t = point_map[0, t, v_src, u_src]
                                if not torch.isfinite(pt_world_t).all():
                                    continue

                                mv = torch.tensor(motion_vec_t_to_next, dtype=pt_world_t.dtype, device=device)
                                pt_interp = pt_world_t + alpha * mv  # (3,)

                                point_map_interp[0, t_interp, v_src, u_src, :] = pt_interp
                                dy_map_interp[0, t_interp, v_src, u_src] = 1.0
        point_map = point_map_interp
        dy_map = dy_map_interp
        
        pred_flows = None
        flow_masks = None
    else:
        T_orig = dy_map.shape[1]
        T_new_local = (T_orig - 1) * interval + 1
        dy_map = dy_map.repeat_interleave(interval, dim=1)[:, :T_new_local, ...]
        point_map = point_map.repeat_interleave(interval, dim=1)[:, :T_new_local, ...]
        pred_flows = None
        flow_masks = None
    B_new, T_new, H_new, W_new, _ = point_map.shape
    if output_length is not None and output_length > T_new:
        extra = output_length - T_new
        last_points = point_map[:, -1:, ...]
        last_dy_map = dy_map[:, -1:, ...]
        point_map = torch.cat([point_map, last_points.repeat(1, extra, 1, 1, 1)], dim=1)
        dy_map = torch.cat([dy_map, last_dy_map.repeat(1, extra, 1, 1)], dim=1)

        if 'last_motion_mask' in locals():
            for step in range(1, extra + 1):
                point_map[:, T_new + step - 1][last_motion_mask] = (
                    last_points[:, 0][last_motion_mask] + step * last_motion[last_motion_mask] / interval
                )

        translation_step = extrinsics_interp[-1, :3, 3] - extrinsics_interp[-2, :3, 3]
        rotation_step = extrinsics_interp[-1, :3, :3] @ extrinsics_interp[-2, :3, :3].transpose(0, 1)
        intrinsic_step = intrinsics_interp[-1] - intrinsics_interp[-2]
        extra_extrinsics, extra_intrinsics = [], []
        prev_extrinsic, prev_intrinsic = extrinsics_interp[-1], intrinsics_interp[-1]
        for _ in range(extra):
            next_extrinsic = prev_extrinsic.clone()
            next_extrinsic[:3, :3] = rotation_step @ prev_extrinsic[:3, :3]
            next_extrinsic[:3, 3] = prev_extrinsic[:3, 3] + translation_step
            prev_extrinsic = next_extrinsic
            prev_intrinsic = prev_intrinsic + intrinsic_step
            extra_extrinsics.append(prev_extrinsic)
            extra_intrinsics.append(prev_intrinsic)
        extrinsics_interp = torch.cat([extrinsics_interp, torch.stack(extra_extrinsics)], dim=0)
        intrinsics_interp = torch.cat([intrinsics_interp, torch.stack(extra_intrinsics)], dim=0)
        T_new = output_length

    depth_interp = torch.full((B_new, T_new, H_new, W_new), float('nan'),
                              device=point_map.device, dtype=point_map.dtype)

    for b in range(B_new):
        for t in range(T_new):
            pts_flat = point_map[b, t].reshape(-1, 3)
            extr_t = extrinsics_interp[t].to(device=point_map.device, dtype=point_map.dtype)
            intr_t = intrinsics_interp[t].to(device=point_map.device, dtype=point_map.dtype)

            depth_map_frame, _ = project_point_cloud(pts_flat, extr_t, intr_t, H_new, W_new)
            depth_frame = depth_map_frame[:, :, 2]

            depth_filled = smooth_depth(depth_frame, kernel_size=5, sigma=1.0, iterations=2)
            depth_interp[b, t] = depth_filled

    final_T = point_map.shape[1]
    gs_map = gs_map.repeat_interleave(interval, dim=1)[:, :final_T, ...]

    return (
        extrinsics_interp,
        intrinsics_interp,
        point_map,
        gs_map,
        dy_map,
        gs_conf.repeat_interleave(interval, dim=1)[:, :point_map.shape[1], ...],
        bg_mask.repeat_interleave(interval, dim=1)[:, :point_map.shape[1], ...],
        target_images,
        pred_flows,
        flow_masks,
        depth_interp
    )


def interp_all_3views(
    extrinsics, intrinsics, point_map, gs_map, dy_map, gs_conf, bg_mask,
    images, target_images, depth, track_model, interval=4,
    views=3, output_length=None,
    cluster_output_dir=None, track_visualization_dir=None,
    motion_comparison_dir=None, waymo_flow=None, motion_source='tapip3d',
    gsrecon_model=None,
):
    """Interpolate each camera independently and restore time-major view order."""
    per_view = []
    for view in range(3):
        view_dir = lambda root: os.path.join(root, f'view_{view}') if root is not None else None
        per_view.append(interp_all(
            extrinsics[view::3], intrinsics[view::3],
            point_map[:, view::3], gs_map[:, view::3], dy_map[:, view::3],
            gs_conf[:, view::3], bg_mask[:, view::3], images[:, view::3],
            target_images[:, view::3], depth[:, view::3], track_model,
            interval=interval, views=1,
            output_length=output_length,
            cluster_output_dir=view_dir(cluster_output_dir),
            track_visualization_dir=view_dir(track_visualization_dir),
            motion_comparison_dir=view_dir(motion_comparison_dir),
            waymo_flow=waymo_flow[:, view::3] if waymo_flow is not None else None,
            motion_source=motion_source, gsrecon_model=gsrecon_model,
        ))

    def interleave(values):
        return torch.stack([values[view][:, index]
                            for index in range(values[0].shape[1])
                            for view in range(3)], dim=1)

    extrinsics_out = torch.stack([
        per_view[view][0][index]
        for index in range(per_view[0][0].shape[0])
        for view in range(3)
    ])
    intrinsics_out = torch.stack([
        per_view[view][1][index]
        for index in range(per_view[0][1].shape[0])
        for view in range(3)
    ])
    return (
        extrinsics_out,
        intrinsics_out,
        interleave([result[2] for result in per_view]),
        interleave([result[3] for result in per_view]),
        interleave([result[4] for result in per_view]),
        interleave([result[5] for result in per_view]),
        interleave([result[6] for result in per_view]),
        target_images,
        None,
        None,
        interleave([result[10] for result in per_view]),
    )
