# Modified by ADToolbox contributors, 2026: standalone reconstruction/edit/export extensions.
import argparse
import json
import os
import random
import time
import numpy as np
from scipy.spatial import cKDTree
import scipy.spatial.transform
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import open3d as o3d
from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear
from gsrecon.models.scene_model import SceneModel
from gsrecon.utils.pose_enc import pose_encoding_to_extri_intri
from utils.ego_trajectory_offsets import apply_ego_trajectory_offsets, load_ego_trajectory_offsets
from gsrecon.utils.geometry import unproject_depth_map_to_point_map
from gsrecon.utils.rotation import mat_to_quat
from gsrecon.utils.gs import concat_list, get_masked_gs, get_split_gs
from gsrecon.utils.gs import alpha_t
from gsrecon.utils.visual_track import visualize_tracks_on_images
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
from utils.interplation import interp_all, interp_all_3views, interpolate_poses_intrinsics
from utils.video_maker import make_comparison_video_quad, _to_uint8_rgb, _apply_colormap_gray
from utils.misc import export_gaussians_to_3dgs_ply, load_gaussians_from_3dgs_ply

def compute_metrics(img1, img2, loss_fn):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)
        lpips_val = loss_fn(img1[i].unsqueeze(0) * 2 - 1, img2[i].unsqueeze(0) * 2 - 1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val.item())
    return sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list), sum(lpips_list) / len(lpips_list)

def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def _masked_rgb_metrics(prediction, target, mask):
    """Compute macro-averaged RGB metrics on a boolean [N,H,W] evaluation mask."""
    prediction = prediction.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    target = target.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != prediction.shape[:3]:
        raise ValueError(f"RGB mask shape {mask.shape} does not match images {prediction.shape[:3]}")

    psnr_values, ssim_values, pixels = [], [], 0
    for predicted_image, target_image, image_mask in zip(prediction, target, mask):
        count = int(image_mask.sum())
        if count == 0:
            continue
        mse = np.square(predicted_image - target_image)[image_mask].mean()
        psnr_values.append(float(10.0 * np.log10(1.0 / max(float(mse), 1e-10))))
        _, ssim_map = structural_similarity(
            predicted_image, target_image, channel_axis=2, data_range=1.0, full=True,
        )
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)
        ssim_values.append(float(ssim_map[image_mask].mean()))
        pixels += count
    return {
        'psnr': _mean_or_none(psnr_values),
        'ssim': _mean_or_none(ssim_values),
        'images': len(psnr_values),
        'pixels': pixels,
        'psnr_values': psnr_values,
        'ssim_values': ssim_values,
    }


def _dynamic_evaluation_mask(non_sky_mask, gt_dynamic_mask):
    non_sky_mask = np.asarray(non_sky_mask, dtype=bool)
    gt_dynamic_mask = np.asarray(gt_dynamic_mask, dtype=bool)
    if non_sky_mask.shape != gt_dynamic_mask.shape:
        raise ValueError("Non-sky and GT dynamic masks must have identical [N,H,W] shapes")
    return non_sky_mask & gt_dynamic_mask


def _dynamic_mask_metrics(pred_dynamic_mask, gt_dynamic_mask, non_sky_mask):
    pred_dynamic_mask = np.asarray(pred_dynamic_mask, dtype=bool)
    gt_dynamic_mask = np.asarray(gt_dynamic_mask, dtype=bool)
    non_sky_mask = np.asarray(non_sky_mask, dtype=bool)
    pred_dynamic = pred_dynamic_mask & non_sky_mask
    gt_dynamic = gt_dynamic_mask & non_sky_mask
    tp = int((pred_dynamic & gt_dynamic).sum())
    fp = int((pred_dynamic & ~gt_dynamic).sum())
    fn = int((~pred_dynamic & gt_dynamic).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'iou': iou, 'f1': f1,
    }


def _metric_evaluation_masks(metric_sky_mask, metric_dynamic_mask, expected_shape):
    metric_sky_mask = np.asarray(metric_sky_mask)
    if metric_sky_mask.shape != expected_shape:
        raise ValueError(f"Metric sky mask shape {metric_sky_mask.shape} does not match renders {expected_shape}")
    sky_mask = metric_sky_mask > 0.5
    non_sky_mask = ~sky_mask
    if metric_dynamic_mask is None:
        return sky_mask, non_sky_mask, None
    metric_dynamic_mask = np.asarray(metric_dynamic_mask)
    if metric_dynamic_mask.shape != expected_shape:
        raise ValueError(f"Metric dynamic mask shape {metric_dynamic_mask.shape} does not match renders {expected_shape}")
    dynamic_mask = metric_dynamic_mask > 0.5
    return sky_mask, non_sky_mask, _dynamic_evaluation_mask(non_sky_mask, dynamic_mask)


def _three_view_slices(num_images):
    if num_images % 3 != 0:
        raise ValueError(f"Three-view metrics require image count divisible by 3, got {num_images}")
    return {
        'front': slice(0, num_images, 3),
        'front_left': slice(1, num_images, 3),
        'front_right': slice(2, num_images, 3),
    }


def _depth_range_stats():
    return {'values': [], 'images': 0, 'pixels': 0}


def _finalize_depth_stats(stats):
    return {
        name: {'rmse': _mean_or_none(value['values']), 'images': value['images'], 'pixels': value['pixels']}
        for name, value in stats.items()
    }


def _aligned_depth_metrics(prediction, target, non_sky_mask, dynamic_mask=None):
    """Fit one affine depth alignment per scene and macro-average frame RMSE values."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    non_sky_mask = np.asarray(non_sky_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != non_sky_mask.shape:
        raise ValueError("Depth prediction, target, and non-sky mask must have identical [N,H,W] shapes")
    if dynamic_mask is not None:
        dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
        if dynamic_mask.shape != prediction.shape:
            raise ValueError("Dynamic mask must have the same [N,H,W] shape as depth")

    ranges = ('0_100m', '100_200m', '0_200m')
    stats = {name: _depth_range_stats() for name in ranges}
    dynamic_stats = {name: _depth_range_stats() for name in ranges}
    valid_base = (
        np.isfinite(prediction) & np.isfinite(target) & non_sky_mask
        & (target >= 0.01) & (target <= 200.0)
    )
    if int(valid_base.sum()) < 2:
        return {'valid': False, 'scale': None, 'offset': None,
                'depth': _finalize_depth_stats(stats), 'dynamic_depth': _finalize_depth_stats(dynamic_stats),
                'depth_values': stats, 'dynamic_depth_values': dynamic_stats}

    x = prediction[valid_base]
    y = target[valid_base]
    x_mean = x.mean()
    y_mean = y.mean()
    variance = np.square(x - x_mean).sum()
    tolerance = np.finfo(np.float64).eps * max(1.0, np.square(x).sum())
    if not np.isfinite(variance) or variance <= tolerance:
        return {'valid': False, 'scale': None, 'offset': None,
                'depth': _finalize_depth_stats(stats), 'dynamic_depth': _finalize_depth_stats(dynamic_stats),
                'depth_values': stats, 'dynamic_depth_values': dynamic_stats}
    covariance = ((x - x_mean) * (y - y_mean)).sum()
    scale = covariance / variance
    offset = y_mean - scale * x_mean
    if not np.isfinite(scale) or not np.isfinite(offset):
        return {'valid': False, 'scale': None, 'offset': None,
                'depth': _finalize_depth_stats(stats), 'dynamic_depth': _finalize_depth_stats(dynamic_stats),
                'depth_values': stats, 'dynamic_depth_values': dynamic_stats}
    aligned = scale * prediction + offset
    for index in range(prediction.shape[0]):
        frame_ranges = {
            '0_100m': valid_base[index] & (target[index] < 100.0),
            '100_200m': valid_base[index] & (target[index] >= 100.0),
            '0_200m': valid_base[index],
        }
        for name, range_mask in frame_ranges.items():
            count = int(range_mask.sum())
            if count:
                stats[name]['values'].append(float(np.sqrt(np.square(aligned[index][range_mask] - target[index][range_mask]).mean())))
                stats[name]['images'] += 1
                stats[name]['pixels'] += count
            if dynamic_mask is not None:
                dynamic_range_mask = range_mask & dynamic_mask[index]
                dynamic_count = int(dynamic_range_mask.sum())
                if dynamic_count:
                    dynamic_stats[name]['values'].append(float(np.sqrt(np.square(aligned[index][dynamic_range_mask] - target[index][dynamic_range_mask]).mean())))
                    dynamic_stats[name]['images'] += 1
                    dynamic_stats[name]['pixels'] += dynamic_count
    return {'valid': True, 'scale': float(scale), 'offset': float(offset),
            'depth': _finalize_depth_stats(stats), 'dynamic_depth': _finalize_depth_stats(dynamic_stats),
            'depth_values': stats, 'dynamic_depth_values': dynamic_stats}

def _empty_depth_metrics():
    ranges = ('0_100m', '100_200m', '0_200m')
    depth = {name: _depth_range_stats() for name in ranges}
    dynamic_depth = {name: _depth_range_stats() for name in ranges}
    return {'valid': False, 'scale': None, 'offset': None,
            'depth': _finalize_depth_stats(depth), 'dynamic_depth': _finalize_depth_stats(dynamic_depth),
            'depth_values': depth, 'dynamic_depth_values': dynamic_depth}



def _new_mode2_metric_aggregate():
    ranges = ('0_100m', '100_200m', '0_200m')
    aggregate = {
        'evaluated_scenes': 0, 'evaluated_images': 0,
        'sky_psnr': [], 'sky_ssim': [],
        'non_sky_psnr': [], 'non_sky_ssim': [], 'dynamic_psnr': [], 'dynamic_ssim': [],
        'valid_sky_images': 0, 'valid_sky_pixels': 0,
        'valid_non_sky_images': 0, 'valid_non_sky_pixels': 0,
        'valid_dynamic_images': 0, 'valid_dynamic_pixels': 0,
        'dynamic_mask_tp': 0, 'dynamic_mask_fp': 0, 'dynamic_mask_fn': 0,
        'depth': {name: _depth_range_stats() for name in ranges},
        'dynamic_depth': {name: _depth_range_stats() for name in ranges},
        'depth_alignment_valid_scenes': 0, 'depth_alignment_invalid_scenes': 0,
    }
    for view_name in ('front', 'front_left', 'front_right'):
        aggregate[f'{view_name}_psnr'] = []
        aggregate[f'{view_name}_ssim'] = []
        aggregate[f'{view_name}_lpips'] = []
        for region in ('sky', 'non_sky', 'dynamic'):
            aggregate[f'{view_name}_{region}_psnr'] = []
            aggregate[f'{view_name}_{region}_ssim'] = []
    return aggregate


def _accumulate_depth_stats(total, scene_stats):
    for name, values in scene_stats.items():
        total[name]['values'].extend(values['values'])
        total[name]['images'] += values['images']
        total[name]['pixels'] += values['pixels']


def _accumulate_masked_rgb_metrics(total, prefix, metrics):
    total[f'{prefix}_psnr'].extend(metrics['psnr_values'])
    total[f'{prefix}_ssim'].extend(metrics['ssim_values'])


def _empty_masked_rgb_metrics():
    return {'psnr': None, 'ssim': None, 'images': 0, 'pixels': 0,
            'psnr_values': [], 'ssim_values': []}


def _metric_depth_array(gt_depth_batch):
    """Extract GT depth with the same channel-0 convention used by Mode 2."""
    if gt_depth_batch.ndim == 4:
        return gt_depth_batch[0].detach().cpu().numpy()
    if gt_depth_batch.ndim == 5:
        if gt_depth_batch.shape[2] == 1:
            return gt_depth_batch[0, :, 0].detach().cpu().numpy()
        return gt_depth_batch[0, ..., 0].detach().cpu().numpy()
    return None


def _region_scene_metrics(
    rendered_image,
    target_image,
    rendered_depth,
    gt_depth,
    metric_sky_mask,
    metric_dynamic_mask,
    loss_fn,
    input_views,
):
    sky_eval_mask, non_sky_mask, dynamic_eval_mask = _metric_evaluation_masks(
        metric_sky_mask, metric_dynamic_mask, tuple(rendered_depth.shape)
    )
    sky_rgb = _masked_rgb_metrics(rendered_image, target_image, sky_eval_mask)
    non_sky_rgb = _masked_rgb_metrics(rendered_image, target_image, non_sky_mask)
    dynamic_rgb = (
        _masked_rgb_metrics(rendered_image, target_image, dynamic_eval_mask)
        if dynamic_eval_mask is not None else _empty_masked_rgb_metrics()
    )
    view_rgb_metrics = {}
    if input_views == 3:
        for view_name, view_slice in _three_view_slices(rendered_image.shape[0]).items():
            view_psnr, view_ssim, view_lpips = compute_metrics(
                rendered_image[view_slice], target_image[view_slice], loss_fn
            )
            view_rgb_metrics[view_name] = {
                'psnr': view_psnr,
                'ssim': view_ssim,
                'lpips': view_lpips,
                'sky': _masked_rgb_metrics(
                    rendered_image[view_slice], target_image[view_slice], sky_eval_mask[view_slice]
                ),
                'non_sky': _masked_rgb_metrics(
                    rendered_image[view_slice], target_image[view_slice], non_sky_mask[view_slice]
                ),
                'dynamic': (
                    _masked_rgb_metrics(
                        rendered_image[view_slice], target_image[view_slice], dynamic_eval_mask[view_slice]
                    )
                    if dynamic_eval_mask is not None else _empty_masked_rgb_metrics()
                ),
            }
    depth_metrics = (
        _aligned_depth_metrics(rendered_depth, gt_depth, non_sky_mask, dynamic_eval_mask)
        if gt_depth is not None else _empty_depth_metrics()
    )
    return {
        'sky': sky_rgb,
        'non_sky': non_sky_rgb,
        'dynamic': dynamic_rgb,
        'views': view_rgb_metrics,
        'depth': depth_metrics,
    }


def _accumulate_region_scene_metrics(total, scene_metrics, image_count):
    total['evaluated_scenes'] += 1
    total['evaluated_images'] += image_count
    for region in ('sky', 'non_sky', 'dynamic'):
        metrics = scene_metrics[region]
        _accumulate_masked_rgb_metrics(total, region, metrics)
        total[f'valid_{region}_images'] += metrics['images']
        total[f'valid_{region}_pixels'] += metrics['pixels']
    for view_name, view_metrics in scene_metrics['views'].items():
        total[f'{view_name}_psnr'].append(view_metrics['psnr'])
        total[f'{view_name}_ssim'].append(view_metrics['ssim'])
        total[f'{view_name}_lpips'].append(view_metrics['lpips'])
        for region in ('sky', 'non_sky', 'dynamic'):
            _accumulate_masked_rgb_metrics(
                total, f'{view_name}_{region}', view_metrics[region]
            )
    depth_metrics = scene_metrics['depth']
    if depth_metrics['valid']:
        total['depth_alignment_valid_scenes'] += 1
    else:
        total['depth_alignment_invalid_scenes'] += 1
    _accumulate_depth_stats(total['depth'], depth_metrics['depth_values'])
    _accumulate_depth_stats(total['dynamic_depth'], depth_metrics['dynamic_depth_values'])


def _finalize_region_metric_aggregate(total, input_views):
    depth_summary = _finalize_depth_stats(total['depth'])
    dynamic_depth_summary = _finalize_depth_stats(total['dynamic_depth'])
    summary = {
        'sky_psnr': _mean_or_none(total['sky_psnr']),
        'sky_ssim': _mean_or_none(total['sky_ssim']),
        'non_sky_psnr': _mean_or_none(total['non_sky_psnr']),
        'non_sky_ssim': _mean_or_none(total['non_sky_ssim']),
        'dynamic_psnr': _mean_or_none(total['dynamic_psnr']),
        'dynamic_ssim': _mean_or_none(total['dynamic_ssim']),
        'aligned_depth_rmse_0_100m': depth_summary['0_100m']['rmse'],
        'aligned_depth_rmse_100_200m': depth_summary['100_200m']['rmse'],
        'aligned_depth_rmse_0_200m': depth_summary['0_200m']['rmse'],
        'dynamic_aligned_depth_rmse_0_100m': dynamic_depth_summary['0_100m']['rmse'],
        'dynamic_aligned_depth_rmse_100_200m': dynamic_depth_summary['100_200m']['rmse'],
        'dynamic_aligned_depth_rmse_0_200m': dynamic_depth_summary['0_200m']['rmse'],
        'evaluated_scenes': total['evaluated_scenes'],
        'evaluated_images': total['evaluated_images'],
        'valid_sky_images': total['valid_sky_images'],
        'valid_sky_pixels': total['valid_sky_pixels'],
        'valid_non_sky_images': total['valid_non_sky_images'],
        'valid_non_sky_pixels': total['valid_non_sky_pixels'],
        'valid_dynamic_images': total['valid_dynamic_images'],
        'valid_dynamic_pixels': total['valid_dynamic_pixels'],
        'depth_alignment_valid_scenes': total['depth_alignment_valid_scenes'],
        'depth_alignment_invalid_scenes': total['depth_alignment_invalid_scenes'],
    }
    for prefix, stats in (('valid_depth', depth_summary), ('valid_dynamic_depth', dynamic_depth_summary)):
        for range_name, values in stats.items():
            summary[f'{prefix}_images_{range_name}'] = values['images']
            summary[f'{prefix}_pixels_{range_name}'] = values['pixels']
    if input_views == 3:
        for view_name in ('front', 'front_left', 'front_right'):
            summary[f'{view_name}_psnr'] = _mean_or_none(total[f'{view_name}_psnr'])
            summary[f'{view_name}_ssim'] = _mean_or_none(total[f'{view_name}_ssim'])
            summary[f'{view_name}_lpips'] = _mean_or_none(total[f'{view_name}_lpips'])
            for region in ('sky', 'non_sky', 'dynamic'):
                summary[f'{view_name}_{region}_psnr'] = _mean_or_none(
                    total[f'{view_name}_{region}_psnr']
                )
                summary[f'{view_name}_{region}_ssim'] = _mean_or_none(
                    total[f'{view_name}_{region}_ssim']
                )
    return summary, depth_summary, dynamic_depth_summary


def calculate_scale_factor(P1, P2):
    distances_P1 = torch.norm(P1[1:], dim=1)
    distances_P2 = torch.norm(P2[1:], dim=1)
    avg_distance_P1 = torch.mean(distances_P1)
    if avg_distance_P1 < 0.1: #almost not move
        return 1
    avg_distance_P2 = torch.mean(distances_P2)
    scale_factor = avg_distance_P2 / avg_distance_P1
    return scale_factor

def save_video(images, path, fps=8):
    images = images.detach().cpu()  # Ensure it's on CPU
    if images.max() <= 1.0:
        images = images * 255.0
    images = images.byte().permute(0, 2, 3, 1).numpy()  # [S, H, W, 3]

    imageio.mimwrite(path, images, fps=fps, codec='libx264')

def load_checkpoint_compat(model, checkpoint):
    """Load legacy checkpoints while allowing newly added optional heads."""
    result = model.load_state_dict(checkpoint, strict=False)
    unexpected = list(result.unexpected_keys)
    allowed_missing_prefixes = ("dense_motion_head.",)
    invalid_missing = [
        key for key in result.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    dense_missing = [key for key in result.missing_keys if key.startswith("dense_motion_head.")]
    if unexpected or invalid_missing:
        raise RuntimeError(f"Checkpoint mismatch: unexpected={unexpected}, missing={invalid_missing}")
    if dense_missing:
        print("Checkpoint has no Dense Motion Head weights; continuing because the selected legacy path does not use it.")



def parse_scene_names(scene_names_str):
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        return [str(i).zfill(3) for i in range(int(start), int(end)+1)]
    else:
        return [str(int(x)).zfill(3) for x in scene_names_str.split()]

def load_instance_bboxes(instance_info_path):
    with open(instance_info_path, encoding='utf-8') as handle:
        config = json.load(handle)
    instances = config.get('instances')
    if not isinstance(instances, list) or not instances:
        raise ValueError('instance_info requires a non-empty instances list')
    targets = []
    seen = set()
    for instance in instances:
        instance_id = instance['instance_id']
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id < 0:
            raise ValueError('instance_id must be a non-negative integer')
        frames = instance['frames']
        if not isinstance(frames, list) or not frames:
            raise ValueError('each instance requires a non-empty frames list')
        for frame in frames:
            frame_index = frame['frame_index']
            bbox = frame['bbox']
            if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
                raise ValueError('frame_index must be a non-negative integer')
            if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox):
                raise ValueError('bbox must be four integers: [x1, y1, x2, y2]')
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError('bbox must satisfy 0 <= x1 < x2 and 0 <= y1 < y2')
            key = (instance_id, frame_index)
            if key in seen:
                raise ValueError(f'duplicate instance/frame bbox: {key}')
            seen.add(key)
            targets.append({'instance_id': instance_id, 'frame_index': frame_index, 'bbox': bbox})
    return targets

def interpolate_instance_trajectory(waypoints, method, frame_indices):
    points = np.asarray(waypoints, dtype=np.float64)
    if len(points) == 1:
        values = np.repeat(points, len(frame_indices), axis=0)
    else:
        positions = np.linspace(0.0, len(points) - 1, len(frame_indices))
        if method == 'linear':
            values = np.stack([
                np.interp(positions, np.arange(len(points)), points[:, axis])
                for axis in range(3)
            ], axis=-1)
        else:
            values = []
            for position in positions:
                segment = min(int(position), len(points) - 2)
                local = position - segment
                p0 = points[max(segment - 1, 0)]
                p1 = points[segment]
                p2 = points[segment + 1]
                p3 = points[min(segment + 2, len(points) - 1)]
                values.append(0.5 * (
                    2 * p1
                    + (-p0 + p2) * local
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * local ** 2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * local ** 3
                ))
            values = np.asarray(values, dtype=np.float64)
    return {frame_index: value.tolist() for frame_index, value in zip(frame_indices, values)}

def load_instance_edit(edit_config_path):
    edit_config_path = os.path.abspath(edit_config_path)
    with open(edit_config_path, encoding='utf-8') as handle:
        config = json.load(handle)
    rigid_edit = config['Nodes']['RigidNodes']
    if len(rigid_edit) != 1 or any(
        name in rigid_edit and len(rigid_edit[name].get("instance_id", [])) > 1
        for name in ("trajectory_edit", "rotate", "scale", "remove")
    ):
        return load_instance_edit_operations(edit_config_path, config, rigid_edit)
    operations = [name for name in ('trajectory_edit', 'rotate', 'scale', 'remove', 'add', 'replace') if name in rigid_edit]
    if len(operations) != 1 or len(rigid_edit) != 1:
        raise ValueError('edit_config requires exactly one of trajectory_edit, rotate, scale, remove, add, or replace')
    edit = rigid_edit[operations[0]]
    if operations[0] == 'add':
        if len(edit['add_obj']) != 1 or len(edit['anchor_pixel']) != 1 or len(edit['anchor_pixel'][0]) != 2 or len(edit['scale']) != 1:
            raise ValueError('add requires one asset, one anchor pixel, and one scale')
        return {
            'operation': 'add',
            'asset_path': os.path.join(os.path.dirname(edit_config_path), edit['add_obj'][0]),
            'anchor_pixel': edit['anchor_pixel'][0],
            'scale': edit['scale'][0],
        }
    if operations[0] == 'replace':
        if len(edit['target_id']) != 1 or len(edit['replace_obj']) != 1 or len(edit['scale']) != 1:
            raise ValueError('replace requires one target_id, one asset, and one scale')
        instance_id = edit['target_id'][0]
        instance_info_path = os.path.join(os.path.dirname(edit_config_path), config['instance_info_path'])
        targets = [target for target in load_instance_bboxes(instance_info_path) if target['instance_id'] == instance_id]
        if not targets:
            raise ValueError(f'instance_id {instance_id} is not present in instance_info')
        return {
            'instance_id': instance_id,
            'targets': targets,
            'operation': 'replace',
            'asset_path': os.path.join(os.path.dirname(edit_config_path), edit['replace_obj'][0]),
            'scale': edit['scale'][0],
            'value': None,
        }
    if operations[0] == 'trajectory_edit':
        if len(edit['instance_id']) != 1 or len(edit['waypoints']) != 1 or not edit['waypoints'][0] or len(edit['method']) != 1:
            raise ValueError('translation requires one instance_id, one or more waypoints, and one method')
        if edit['method'][0] not in ('linear', 'cubic'):
            raise ValueError('translation method must be linear or cubic')
        if any(len(point) != 3 for point in edit['waypoints'][0]):
            raise ValueError('translation waypoints must contain [x, y, z] points')
        operation = 'translation'
        waypoints = edit['waypoints'][0]
        method = edit['method'][0]
    elif operations[0] == 'rotate':
        if len(edit['instance_id']) != 1 or len(edit['rotation']) != 1 or len(edit['rotation'][0]) != 3:
            raise ValueError('rotation requires one instance_id and one XYZ rotation')
        operation = 'rotation'
        value = edit['rotation'][0]
    elif operations[0] == 'scale':
        if len(edit['instance_id']) != 1 or len(edit['scale']) != 1:
            raise ValueError('scale requires one instance_id and one scale')
        operation = 'scale'
        value = edit['scale'][0]
    else:
        if len(edit['instance_id']) != 1:
            raise ValueError('remove requires one instance_id')
        operation = 'remove'
        value = None
    instance_info_path = os.path.join(os.path.dirname(edit_config_path), config['instance_info_path'])
    instance_id = edit['instance_id'][0]
    targets = [target for target in load_instance_bboxes(instance_info_path) if target['instance_id'] == instance_id]
    if not targets:
        raise ValueError(f'instance_id {instance_id} is not present in instance_info')
    value_by_frame = None
    if operation == 'translation':
        value_by_frame = interpolate_instance_trajectory(
            waypoints, method, [target['frame_index'] for target in targets]
        )
        value = value_by_frame[targets[0]['frame_index']] if len(waypoints) == 1 else None
    return {
        'instance_id': instance_id,
        'targets': targets,
        'operation': operation,
        'value': value,
        'value_by_frame': value_by_frame,
    }

def load_instance_edit_operations(edit_config_path, config, rigid_edit):
    base_dir = os.path.dirname(edit_config_path)
    all_targets = load_instance_bboxes(os.path.join(base_dir, config['instance_info_path']))

    def targets_for(instance_id):
        targets = [target for target in all_targets if target['instance_id'] == instance_id]
        if not targets:
            raise ValueError(f'instance_id {instance_id} is not present in instance_info')
        return targets

    operations = []
    for instance_id, waypoints, method in zip(*[
        rigid_edit.get('trajectory_edit', {}).get(key, [])
        for key in ('instance_id', 'waypoints', 'method')
    ]):
        targets = targets_for(instance_id)
        value_by_frame = interpolate_instance_trajectory(waypoints, method, [target['frame_index'] for target in targets])
        operations.append({'operation': 'translation', 'instance_id': instance_id, 'targets': targets,
                           'value': value_by_frame[targets[0]['frame_index']] if len(waypoints) == 1 else None,
                           'value_by_frame': value_by_frame})
    for instance_id, value in zip(*[
        rigid_edit.get('rotate', {}).get(key, []) for key in ('instance_id', 'rotation')
    ]):
        operations.append({'operation': 'rotation', 'instance_id': instance_id, 'targets': targets_for(instance_id),
                           'value': value, 'value_by_frame': None})
    for instance_id, value in zip(*[
        rigid_edit.get('scale', {}).get(key, []) for key in ('instance_id', 'scale')
    ]):
        operations.append({'operation': 'scale', 'instance_id': instance_id, 'targets': targets_for(instance_id),
                           'value': value, 'value_by_frame': None})
    for instance_id in rigid_edit.get('remove', {}).get('instance_id', []):
        operations.append({'operation': 'remove', 'instance_id': instance_id, 'targets': targets_for(instance_id),
                           'value': None, 'value_by_frame': None})
    if 'replace' in rigid_edit:
        edit = rigid_edit['replace']
        instance_id = edit['target_id'][0]
        operations.append({'operation': 'replace', 'instance_id': instance_id, 'targets': targets_for(instance_id),
                           'asset_path': os.path.join(base_dir, edit['replace_obj'][0]), 'scale': edit['scale'][0],
                           'value': None, 'value_by_frame': None})
    if 'add' in rigid_edit:
        edit = rigid_edit['add']
        operations.append({'operation': 'add', 'instance_id': None, 'targets': [],
                           'asset_path': os.path.join(base_dir, edit['add_obj'][0]),
                           'anchor_pixel': edit['anchor_pixel'][0], 'scale': edit['scale'][0],
                           'value': None, 'value_by_frame': None})
    return {'operation': 'combined', 'operations': operations,
            'targets': [target for operation in operations for target in operation['targets']],
            'instance_id': None, 'value': None, 'value_by_frame': None}
def build_instance_masks(targets, bg_mask, dynamic_probability, threshold):
    _, _, height, width = bg_mask.shape
    masks = {}
    for target in targets:
        frame_index = target['frame_index']
        x1, y1, x2, y2 = target['bbox']
        if frame_index >= bg_mask.shape[1] or x2 > width or y2 > height:
            raise ValueError('instance frame or bbox is outside the model input')
        roi_mask = torch.zeros_like(bg_mask[:, frame_index])
        roi_mask[..., y1:y2, x1:x2] = True
        mask = (
            roi_mask
            & bg_mask[:, frame_index]
            & (dynamic_probability[:, frame_index] >= threshold)
        )
        masks[frame_index] = masks.get(frame_index, torch.zeros_like(mask)) | mask
    return masks

def apply_instance_translation(points, selection, translation):
    edited = points.clone()
    edited[selection] += torch.as_tensor(translation, dtype=points.dtype, device=points.device)
    return edited

def left_multiply_quaternions(delta, quaternions):
    delta = delta.to(dtype=quaternions.dtype, device=quaternions.device)
    dw, dx, dy, dz = delta
    ow, ox, oy, oz = quaternions.unbind(dim=-1)
    return F.normalize(torch.stack([
        dw * ow - dx * ox - dy * oy - dz * oz,
        dw * ox + dx * ow + dy * oz - dz * oy,
        dw * oy - dx * oz + dy * ow + dz * ox,
        dw * oz + dx * oy - dy * ox + dz * ow,
    ], dim=-1), dim=-1)

def apply_instance_rotation(points, quaternions, selection, rotation_degrees, center):
    angles = torch.deg2rad(torch.as_tensor(rotation_degrees, dtype=points.dtype, device=points.device))
    rx, ry, rz = angles
    cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
    sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)
    rotation = torch.stack([
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx,
    ]).reshape(3, 3)

    center = torch.as_tensor(center, dtype=points.dtype, device=points.device)
    edited_points = points.clone()
    edited_points[selection] = (points[selection] - center) @ rotation.T + center

    half_rx, half_ry, half_rz = angles / 2
    chx, chy, chz = torch.cos(half_rx), torch.cos(half_ry), torch.cos(half_rz)
    shx, shy, shz = torch.sin(half_rx), torch.sin(half_ry), torch.sin(half_rz)
    delta = torch.stack([
        chz * chy * chx + shz * shy * shx,
        chz * chy * shx - shz * shy * chx,
        chz * shy * chx + shz * chy * shx,
        shz * chy * chx - chz * shy * shx,
    ])
    edited_quaternions = quaternions.clone()
    edited_quaternions[selection] = left_multiply_quaternions(delta, quaternions[selection])
    return edited_points, edited_quaternions

def apply_instance_scale(points, scales, selection, factor, center):
    center = torch.as_tensor(center, dtype=points.dtype, device=points.device)
    point_factor = torch.as_tensor(factor, dtype=points.dtype, device=points.device)
    scale_factor = torch.as_tensor(factor, dtype=scales.dtype, device=scales.device)
    edited_points = points.clone()
    edited_scales = scales.clone()
    edited_points[selection] = center + point_factor * (points[selection] - center)
    edited_scales[selection] = scales[selection] * scale_factor
    return edited_points, edited_scales

def remove_selected_gaussians(selection, *attributes):
    keep = ~selection
    return tuple(attribute[keep] for attribute in attributes)

def place_rigid_asset(asset, first_extrinsic, anchor_point, scale_factor):
    rotation = first_extrinsic[:3, :3]
    asset_to_camera = asset['means'].new_tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    asset_to_world = rotation.T @ asset_to_camera
    world_rotation_xyzw = mat_to_quat(asset_to_world)
    world_rotation_wxyz = world_rotation_xyzw[[3, 0, 1, 2]]
    asset_contact = asset['means'].mean(dim=0)
    asset_contact[2] = asset['means'][:, 2].min()
    anchor_point = torch.as_tensor(anchor_point, dtype=asset['means'].dtype, device=asset['means'].device)
    factor = torch.as_tensor(scale_factor, dtype=asset['means'].dtype, device=asset['means'].device)
    return {
        'means': anchor_point + (asset['means'] - asset_contact) @ asset_to_world.T * factor,
        'colors': asset['colors'],
        'opacities': asset['opacities'],
        'scales': asset['scales'] * factor,
        'quaternions': left_multiply_quaternions(world_rotation_wxyz, asset['quaternions']),
    }

def place_replacement_asset(asset, first_extrinsic, center, scale_factor):
    placed = place_rigid_asset(asset, first_extrinsic, center, scale_factor)
    center = torch.as_tensor(center, dtype=asset['means'].dtype, device=asset['means'].device)
    placed['means'] = placed['means'] + center - placed['means'].mean(dim=0)
    return placed

@torch.inference_mode()
def main(argv=None, model=None, loss_fn=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the input images')
    parser.add_argument('--scene_names', type=str, nargs='+', required=True, help='Scene names, supports formats like 3 5 7 or (3,7)')
    parser.add_argument('--input_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index')
    parser.add_argument('--mode', type=int, choices=[1,2,3], required=True, help='Processing mode')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for results')
    parser.add_argument('-images', action='store_true', help='Whether to output each frame image')
    parser.add_argument('-depth', action='store_true', help='Whether to output each frame depth as .npy')
    parser.add_argument('-metrics', action='store_true', help='Whether to output evaluation metrics')
    parser.add_argument('--metrics_only', action='store_true', help='Only compute metrics; do not export render files')
    parser.add_argument('--per_scene_metrics_file', type=str, default=None, help='Path to append per-scene metrics as JSONL')
    parser.add_argument('--summary_file', type=str, default=None, help='Path to write average metrics as JSON')
    parser.add_argument('--export_camera_params', action='store_true', help='Save predicted camera parameters for every output frame')
    parser.add_argument('--export_opacity_dir', type=str, default=None, help='Directory for per-frame Gaussian opacity PNGs')
    parser.add_argument('--export_input_frames_dir', type=str, default=None, help='Directory for model input RGB PNGs')
    parser.add_argument('--tracking_ckpt', type=str, default='pretrained/tracking_model.pth', help='Path to the TAPIP3D checkpoint required by mode=3')
    parser.add_argument('--motion_source', choices=['tapip3d', 'track_head'], default='tapip3d', help='Source of mode=3 dynamic 3D tracks')
    parser.add_argument('--track_head_ckpt', type=str, default=None, help='Optional SceneModel checkpoint used to replace only Track Head weights')
    parser.add_argument('--aggregator_ckpt', type=str, default=None, help='Optional SceneModel checkpoint used to replace only Aggregator weights')
    parser.add_argument('--intervals', type=int, default=2, help='Interval for mode=3')
    parser.add_argument('--export_clusters', action='store_true', help='Export TAPIP3D dynamic-cluster overlays for mode=3')
    parser.add_argument('--export_track_head_tracks', action='store_true', help='Export SceneModel Track Head 2D trajectories for mode=3')
    parser.add_argument('--export_motion_comparison', action='store_true', help='Export same-query TAPIP3D, SceneModel Track Head and Waymo flow samples for mode=3')
    parser.add_argument('--lateral_shifts', type=float, nargs='*', default=[], help='实验性横向平移量，仅mode=2')
    parser.add_argument('--ego_trajectory_offsets', type=str, default=None, help='Mode2 自车渲染轨迹相对偏移 JSON；每项为相机局部 [前, 左, 上]，GSRecon 原生单位')
    parser.add_argument('--output_fps', type=float, default=8, help='输出视频帧率')
    parser.add_argument('-export_ply', action='store_true', help='导出 static、dynamic、combined Gaussian PLY')
    parser.add_argument('--ply_only', action='store_true', help='仅导出 PLY 及配套相机位姿，不执行 2D 渲染')
    parser.add_argument('--export_dynamic_frame_ply', action='store_true', help='将每帧动态高斯分别导出为 PLY')
    parser.add_argument('--export_render_state', action='store_true', help='保存未裁剪的 Mode2 时序高斯，供仿真引擎保真渲染')
    parser.add_argument('--ply_opacity_threshold', type=float, default=0.005, help='静态和合并 PLY 的透明度阈值')
    parser.add_argument('--ply_dynamic_threshold', type=float, default=0.5, help='导出动态 Gaussian 的动态概率阈值')
    parser.add_argument('--save_dynamic_mask_vis', action='store_true', help='保存单相机 Mode2 诊断图动态概率与二值掩码')
    parser.add_argument('--ply_dynamic_opacity_threshold', type=float, default=0.001, help='动态 PLY 的透明度阈值')
    parser.add_argument('--ply_max_points', type=int, default=500000, help='静态和合并 PLY 的最大点数，-1 表示不限制')
    parser.add_argument('--ply_dynamic_max_points', type=int, default=-1, help='动态 PLY 的最大点数，-1 表示不限制')
    parser.add_argument('--ply_voxel_size', type=float, default=0.002, help='静态和合并 PLY 的体素去重尺寸，0 表示关闭')
    parser.add_argument('--ply_include_timestamp', action='store_true', help='在动态 PLY 中保存时间戳')
    parser.add_argument('--ply_no_filter', action='store_true', help='关闭透明度过滤、体素去重和点数限制，导出全部 Gaussian')
    parser.add_argument('--ply_no_normalize', action='store_true', help='使用原始 GSRecon 世界坐标导出 PLY，适配 SuperSplat 的低精度相机输入')
    parser.add_argument('--edit_config', type=str, default=None, help='单相机 Mode2 bbox 实例编辑 JSON')
    args = parser.parse_args(argv)
    if args.export_ply and args.mode != 2:
        raise ValueError('-export_ply currently supports mode=2')
    if args.export_render_state and (args.mode != 2 or args.input_views != 1 or not args.export_ply or args.edit_config):
        raise ValueError('--export_render_state requires unedited single-view Mode2 with -export_ply')
    if args.ego_trajectory_offsets and (args.mode != 2 or args.input_views != 1 or len(parse_scene_names(' '.join(args.scene_names))) != 1):
        raise ValueError('--ego_trajectory_offsets requires one scene with --mode 2 --input_views 1')
    if args.ego_trajectory_offsets and (args.lateral_shifts or args.metrics_only):
        raise ValueError('--ego_trajectory_offsets cannot be combined with --lateral_shifts or --metrics_only')
    ego_trajectory_offsets = load_ego_trajectory_offsets(args.ego_trajectory_offsets) if args.ego_trajectory_offsets else None
    per_scene_metrics_path = args.per_scene_metrics_file or os.path.join(args.output_path, 'per_scene_metrics.jsonl')
    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if loss_fn is None:
        loss_fn = lpips.LPIPS(net='alex').to(device)

    scene_names_str = ' '.join(args.scene_names)
    scene_names = parse_scene_names(scene_names_str)
    instance_edit = None
    if args.edit_config is not None:
        if args.mode != 2 or args.input_views != 1 or len(scene_names) != 1:
            raise ValueError('--edit_config requires one scene with --mode 2 --input_views 1')
        instance_edit = load_instance_edit(args.edit_config)
    if args.mode == 3:
        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=scene_names,
            sequence_length=args.sequence_length,
            start_idx=args.start_idx,
            mode=args.mode,
            views=args.input_views,
            intervals=args.intervals,
        )
    else:
        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=scene_names,
            sequence_length=args.sequence_length,
            start_idx=args.start_idx,
            mode=args.mode,
            views=args.input_views
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if model is None:
        model = SceneModel().to(device)
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        load_checkpoint_compat(model, checkpoint)
        if args.aggregator_ckpt is not None:
            aggregator_checkpoint = torch.load(args.aggregator_ckpt, map_location="cpu")
            aggregator_state = {
                key.removeprefix("aggregator."): value
                for key, value in aggregator_checkpoint.items()
                if key.startswith("aggregator.")
            }
            model.aggregator.load_state_dict(aggregator_state, strict=True)
            print(f"Loaded Aggregator from {args.aggregator_ckpt}")
        if args.track_head_ckpt is not None:
            track_checkpoint = torch.load(args.track_head_ckpt, map_location="cpu")
            track_state = {
                key.removeprefix("track_head."): value
                for key, value in track_checkpoint.items()
                if key.startswith("track_head.")
            }
            model.track_head.load_state_dict(track_state, strict=True)
            print(f"Loaded Track Head from {args.track_head_ckpt}")
    track_model = None
    if args.mode == 3 and (args.motion_source == 'tapip3d' or args.export_motion_comparison):
        track_ckpt = args.tracking_ckpt
        track_model = load_model(track_ckpt)
        track_model.to(device)
        track_model.seq_len = 2
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []
    reconstruction_metrics, non_reconstruction_metrics = [], []
    inference_time_list = []
    mode2_metrics = _new_mode2_metric_aggregate() if args.mode == 2 else None
    warned_missing_dynamic_mask = False
    warned_missing_depth = False
    lateral_renders = {float(shift): [] for shift in args.lateral_shifts}
    lateral_alphas = {float(shift): [] for shift in args.lateral_shifts}
    lateral_sky_masks = {float(shift): [] for shift in args.lateral_shifts}
    scene_idx = 1

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)

            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            gt_depth = batch['gt_depth'].to(device)

            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            if args.mode == 3:
                target_images = batch['targets'].to(device)
                target_sky_masks = batch['target_masks'].to(device)

            start_time = time.time()
            dynamic = False
            if 'dynamic_mask' in batch:
                dynamic = True
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]

            with torch.cuda.amp.autocast(dtype=dtype):
                model_input_images = images[0].detach()
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]
                intervals=args.intervals
                views=args.input_views

                use_depth = True
                if use_depth:
                    depth_map = predictions["depth"][0]
                    point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None,...]
                    point_map = torch.from_numpy(point_map).to(device).float()
                else:
                    point_map = predictions["world_points"]
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1) #B,H,W,1

                edit_masks = {}
                edit_operations = [] if instance_edit is None else instance_edit.get("operations", [instance_edit])
                edit_operation = None if instance_edit is None else instance_edit["operation"]
                edit_value = None if instance_edit is None else instance_edit.get("value")
                if args.mode == 2:
                    dy_prob = dy_map.sigmoid()
                    if instance_edit is not None:
                        edit_operation = instance_edit['operation']
                        if edit_operation != 'add':
                            edit_masks = build_instance_masks(
                                instance_edit['targets'], bg_mask, dy_prob, args.ply_dynamic_threshold
                            )
                            edit_value = instance_edit['value']
                    if args.save_dynamic_mask_vis and args.input_views == 1:
                        scene_out_dir = os.path.join(args.output_path, str(scene_idx).zfill(3))
                        vis_dir = os.path.join(scene_out_dir, 'dynamic_mask_vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        for frame_index, input_rgb in enumerate(model_input_images):
                            rgb = _to_uint8_rgb(input_rgb)
                            probability = dy_prob[0, frame_index].detach().float().cpu().numpy()
                            heatmap = _apply_colormap_gray((probability * 255).astype(np.uint8))
                            mask = (bg_mask[0, frame_index] & (dy_prob[0, frame_index] >= args.ply_dynamic_threshold)).cpu().numpy()
                            overlay = rgb.copy()
                            overlay[mask] = (0.5 * rgb[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                            Image.fromarray(np.concatenate([rgb, heatmap, overlay], axis=1)).save(
                                os.path.join(vis_dir, f'{frame_index:04d}.png'))
                    # Match Mode1 soft static/dynamic opacity decomposition while excluding sky Gaussians.
                    static_mask = bg_mask
                    static_points = point_map[static_mask].reshape(-1, 3)
                    gs_dynamic_list = dy_map[static_mask].sigmoid()
                    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                    static_opacity = static_opacity * (1 - gs_dynamic_list)
                    static_gs_conf = gs_conf[static_mask]
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                    gs_timestamps = timestamps[frame_idx]
                    if instance_edit is not None and edit_operation != 'add':
                        static_edit_selection = torch.zeros_like(static_opacity, dtype=torch.bool)
                        static_edit_selections = {}
                        for frame_index, edit_mask in edit_masks.items():
                            static_edit_mask = torch.zeros_like(static_mask)
                            static_edit_mask[:, frame_index] = edit_mask
                            static_edit_selections[frame_index] = static_edit_mask[static_mask]
                            static_edit_selection |= static_edit_selections[frame_index]
                operation_edit_masks = [
                    build_instance_masks(operation["targets"], bg_mask, dy_prob, args.ply_dynamic_threshold)
                    if operation["targets"] else {}
                    for operation in edit_operations
                ]

                if args.mode == 3:
                    depth_map = depth_map.unsqueeze(0)
                    interp_fn = interp_all_3views if args.input_views == 3 else interp_all
                    (extrinsic, intrinsic, point_map, gs_map, dy_map,
                        gs_conf, bg_mask, images, pred_flows, flow_masks,depth_interp) = interp_fn(extrinsic, intrinsic, point_map, gs_map, dy_map,
                                                                                gs_conf, bg_mask, images, target_images, depth_map, track_model,intervals,views,
                                                                                output_length=None,
                                                                                cluster_output_dir=os.path.join(args.output_path, str(scene_idx).zfill(3) + "/clusters") if args.export_clusters else None,
                                                                                track_visualization_dir=os.path.join(args.output_path, str(scene_idx).zfill(3) + "/track_head") if args.export_track_head_tracks else None,
                                                                                motion_comparison_dir=os.path.join(args.output_path, str(scene_idx).zfill(3) + "/motion_comparison") if args.export_motion_comparison else None,
                                                                                waymo_flow=gt_depth[0, ::intervals, ..., 1:4] if args.export_motion_comparison else None,
                                                                                motion_source=args.motion_source,
                                                                                gsrecon_model=model)
                    I = intervals
                    if args.input_views == 3:
                        endpoint_indices = list(range(3)) + list(range(point_map.shape[1] - 3, point_map.shape[1]))
                        render_source_times = timestamps[::3]
                        render_timestamps = torch.stack([
                            torch.repeat_interleave(render_source_times, I)[:point_map.shape[1] // 3]
                            for _ in range(3)
                        ], dim=1).reshape(-1)
                    else:
                        endpoint_indices = list(range(0, point_map.shape[1], I))
                        render_timestamps = torch.repeat_interleave(timestamps, I)[:point_map.shape[1]]
                    bg_point_map = point_map[:, endpoint_indices, ...]
                    bg_bg_mask = bg_mask[:, endpoint_indices, ...]
                    bg_gs_map = gs_map[:, endpoint_indices, ...]
                    bg_dy_map = dy_map[:, endpoint_indices, ...]
                    bg_gs_conf = gs_conf[:, endpoint_indices, ...]

                    static_mask = (bg_bg_mask & (bg_dy_map < 0.5))
                    gs_conf = bg_gs_conf[static_mask]
                    static_points = bg_point_map[static_mask].reshape(-1, 3)
                    gs_dynamic_list = bg_dy_map[static_mask].sigmoid()
                    static_rgbs, static_opacity, static_scales, static_rotation = get_split_gs(bg_gs_map, static_mask)
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                    gs_timestamps = timestamps[frame_idx]
                    static_opacity = static_opacity * (1 - gs_dynamic_list)


                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                dynamic_edit_selections = {}
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    dy_conf_i = dy_map[:, i].sigmoid()

                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i], bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i

                    dynamic_points.append(dynamic_point)
                    dynamic_rgbs.append(dynamic_rgb)
                    dynamic_opacitys.append(dynamic_opacity)
                    dynamic_scales.append(dynamic_scale)
                    dynamic_rotations.append(dynamic_rotation)
                    if i in edit_masks:
                        dynamic_edit_selections[i] = edit_masks[i][bg_mask_i]
                operation_static_edit_selections = []
                operation_static_union_selections = []
                operation_dynamic_edit_selections = []
                for operation_masks in operation_edit_masks:
                    static_by_frame = {}
                    static_union = torch.zeros_like(static_opacity, dtype=torch.bool)
                    dynamic_by_frame = {}
                    for frame_index, edit_mask in operation_masks.items():
                        static_edit_mask = torch.zeros_like(static_mask)
                        static_edit_mask[:, frame_index] = edit_mask
                        static_by_frame[frame_index] = static_edit_mask[static_mask]
                        static_union |= static_by_frame[frame_index]
                        dynamic_by_frame[frame_index] = edit_mask[bg_mask[:, frame_index]]
                    operation_static_edit_selections.append(static_by_frame)
                    operation_static_union_selections.append(static_union)
                    operation_dynamic_edit_selections.append(dynamic_by_frame)
                operation_edit_centers = [{} for _ in edit_operations]
                operation_added_gaussians = {}
                operation_replacement_gaussians = {}
                for operation_index, operation in enumerate(edit_operations):
                    if operation["operation"] in ("rotation", "scale", "replace"):
                        for frame_index, selection in operation_dynamic_edit_selections[operation_index].items():
                            if not selection.any():
                                raise ValueError(f"{operation['operation']} instance has no dynamic Gaussian in frame {frame_index}")
                            operation_edit_centers[operation_index][frame_index] = dynamic_points[frame_index][selection].mean(dim=0)
                    if operation["operation"] in ("add", "replace"):
                        asset = load_gaussians_from_3dgs_ply(operation["asset_path"])
                        asset = {key: value.to(device=point_map.device, dtype=point_map.dtype) for key, value in asset.items()}
                        if operation["operation"] == "add":
                            anchor_x, anchor_y = operation["anchor_pixel"]
                            operation_added_gaussians[operation_index] = place_rigid_asset(asset, extrinsic[0], point_map[0, 0, anchor_y, anchor_x], operation["scale"])
                        else:
                            operation_replacement_gaussians[operation_index] = place_replacement_asset(asset, extrinsic[0], torch.zeros(3, device=point_map.device), operation["scale"])
                edit_centers = {}
                if instance_edit is not None and edit_operation in ('rotation', 'scale', 'replace'):
                    for frame_index, selection in dynamic_edit_selections.items():
                        if not selection.any():
                            raise ValueError(f'{edit_operation} instance has no dynamic Gaussian in frame {frame_index}')
                        edit_centers[frame_index] = dynamic_points[frame_index][selection].mean(dim=0)
                added_gaussians = None
                replacement_gaussians = None
                if instance_edit is not None and edit_operation in ('add', 'replace'):
                    asset = load_gaussians_from_3dgs_ply(instance_edit['asset_path'])
                    asset = {key: value.to(device=point_map.device, dtype=point_map.dtype) for key, value in asset.items()}
                    if edit_operation == 'add':
                        anchor_x, anchor_y = instance_edit['anchor_pixel']
                        added_gaussians = place_rigid_asset(
                            asset, extrinsic[0], point_map[0, 0, anchor_y, anchor_x], instance_edit['scale']
                        )
                        print(
                            f"Added asset {instance_edit['asset_path']} at pixel {instance_edit['anchor_pixel']} "
                            f"with {len(added_gaussians['means'])} Gaussians"
                        )
                    else:
                        replacement_gaussians = place_replacement_asset(
                            asset, extrinsic[0], torch.zeros(3, device=point_map.device), instance_edit['scale']
                        )
                        print(
                            f"Loaded replacement asset {instance_edit['asset_path']} "
                            f"with {len(replacement_gaussians['means'])} Gaussians"
                        )

                if args.export_ply and args.mode == 2:
                    export_static_mask = bg_mask & (dy_prob < args.ply_dynamic_threshold)
                    export_static_points = point_map[export_static_mask].reshape(-1, 3)
                    export_static_rgbs, export_static_opacity, export_static_scales, export_static_rotations = get_split_gs(gs_map, export_static_mask)
                    export_static_opacity = export_static_opacity * (1 - dy_prob[export_static_mask])
                    export_dynamic = {'positions': [], 'colors': [], 'opacity': [], 'scales': [], 'rotation': [], 'dynamic_score': [], 'timestamp': []}
                    export_dynamic_frames = []
                    for frame_index in range(dy_map.shape[1]):
                        export_dynamic_mask = bg_mask[:, frame_index] & (dy_prob[:, frame_index] >= args.ply_dynamic_threshold)
                        if not export_dynamic_mask.any():
                            continue
                        rgb, opacity, scales, rotation = get_split_gs(gs_map[:, frame_index], export_dynamic_mask)
                        score = dy_prob[:, frame_index][export_dynamic_mask]
                        count = score.numel()
                        frame_storage = {
                            'positions': point_map[:, frame_index][export_dynamic_mask].reshape(-1, 3).detach().cpu(),
                            'colors': rgb.detach().cpu(),
                            'opacity': (opacity * score).detach().cpu(),
                            'scales': scales.detach().cpu(),
                            'rotation': rotation.detach().cpu(),
                            'dynamic_score': score.detach().cpu(),
                            'timestamp': torch.full((count,), timestamps[frame_index].item()),
                        }
                        export_dynamic_frames.append((frame_index, frame_storage))
                        for key, values in export_dynamic.items():
                            values.append(frame_storage[key])
                    sky_intrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(intrinsic.shape[0], 1, 1)
                    sky_intrinsics[:, :3, :3] = intrinsic
                    sky_rgb, sky_visible, sky_scale_residual = model.sky_model._get_background_color(
                        images, extrinsic, sky_intrinsics
                    )
                    export_sky = {
                        'positions': model.sky_model.bg_pcd[sky_visible].detach().cpu(),
                        'colors': sky_rgb.detach().cpu(),
                        'opacity': model.sky_model.bg_opacity[sky_visible, 0].detach().cpu(),
                        'scales': (torch.exp(model.sky_model.bg_scales)[sky_visible] + sky_scale_residual).detach().cpu(),
                        'rotation': model.sky_model.bg_quat[sky_visible].detach().cpu(),
                    }

            scene_name = str(scene_idx).zfill(3)
            ply_normalization = None
            if args.export_ply:
                ply_dir = os.path.join(args.output_path, 'point_clouds')
                max_points = None if args.ply_no_filter or args.ply_max_points < 0 else args.ply_max_points
                dynamic_max_points = None if args.ply_no_filter or args.ply_dynamic_max_points < 0 else args.ply_dynamic_max_points
                static_opacity_threshold = 0.0 if args.ply_no_filter else args.ply_opacity_threshold
                dynamic_opacity_threshold = 0.0 if args.ply_no_filter else args.ply_dynamic_opacity_threshold
                voxel_size = 0.0 if args.ply_no_filter else args.ply_voxel_size
                static_storage = {
                    'positions': export_static_points.detach().cpu(),
                    'colors': export_static_rgbs.detach().cpu(),
                    'opacity': export_static_opacity.detach().cpu(),
                    'scales': export_static_scales.detach().cpu(),
                    'rotation': export_static_rotations.detach().cpu(),
                }
                dynamic_storage = None
                if export_dynamic['positions']:
                    dynamic_storage = {key: torch.cat(values, dim=0) for key, values in export_dynamic.items()}

                combined_positions = [static_storage['positions']]
                combined_colors = [static_storage['colors']]
                combined_opacity = [static_storage['opacity']]
                combined_scales = [static_storage['scales']]
                combined_rotation = [static_storage['rotation']]
                combined_score = [torch.zeros_like(static_storage['opacity'])]
                combined_positions.append(export_sky['positions'])
                combined_colors.append(export_sky['colors'])
                combined_opacity.append(export_sky['opacity'])
                combined_scales.append(export_sky['scales'])
                combined_rotation.append(export_sky['rotation'])
                combined_score.append(torch.zeros_like(export_sky['opacity']))
                if dynamic_storage is not None:
                    combined_positions.append(dynamic_storage['positions'])
                    combined_colors.append(dynamic_storage['colors'])
                    combined_opacity.append(dynamic_storage['opacity'])
                    combined_scales.append(dynamic_storage['scales'])
                    combined_rotation.append(dynamic_storage['rotation'])
                    combined_score.append(dynamic_storage['dynamic_score'])
                combined_storage = {
                    'positions': torch.cat(combined_positions),
                    'colors': torch.cat(combined_colors),
                    'opacity': torch.cat(combined_opacity),
                    'scales': torch.cat(combined_scales),
                    'rotation': torch.cat(combined_rotation),
                    'dynamic_score': torch.cat(combined_score),
                }
                static_sky_storage = {
                    'positions': torch.cat([static_storage['positions'], export_sky['positions']]),
                    'colors': torch.cat([static_storage['colors'], export_sky['colors']]),
                    'opacity': torch.cat([static_storage['opacity'], export_sky['opacity']]),
                    'scales': torch.cat([static_storage['scales'], export_sky['scales']]),
                    'rotation': torch.cat([static_storage['rotation'], export_sky['rotation']]),
                }
                if args.ply_no_normalize:
                    center = np.zeros(3, dtype=np.float32)
                    scale = 1.0
                    ply_coordinate_convention = 'OpenCV world-to-camera for original GSRecon world coordinates'
                    ply_world_formula = 'ply_point = original_world_point (no normalization)'
                else:
                    center = combined_storage['positions'].mean(dim=0).numpy()
                    scale = float(np.abs(combined_storage['positions'].numpy() - center).max() * 1.1)
                    ply_coordinate_convention = 'OpenCV world-to-camera for normalized PLY coordinates'
                    ply_world_formula = 'ply_point = (original_world_point - center) / scale'
                ply_normalization = {'center': center, 'scale': scale, 'formula': ply_world_formula, 'coordinate_convention': ply_coordinate_convention}
                export_gaussians_to_3dgs_ply(
                    static_storage, ply_dir, f'static_{scene_name}.ply',
                    opacity_threshold=static_opacity_threshold, max_points=max_points,
                    voxel_size=voxel_size, normalization_center=center, normalization_scale=scale,
                )
                export_gaussians_to_3dgs_ply(
                    export_sky, ply_dir, f'sky_{scene_name}.ply',
                    opacity_threshold=0.0, max_points=None, voxel_size=0,
                    normalization_center=center, normalization_scale=scale,
                )
                export_gaussians_to_3dgs_ply(
                    static_sky_storage, ply_dir, f'static_sky_{scene_name}.ply',
                    opacity_threshold=static_opacity_threshold, max_points=max_points,
                    voxel_size=voxel_size, normalization_center=center, normalization_scale=scale,
                )
                if dynamic_storage is not None:
                    if not args.ply_include_timestamp:
                        dynamic_storage.pop('timestamp')
                    export_gaussians_to_3dgs_ply(
                        dynamic_storage, ply_dir, f'dynamic_{scene_name}.ply',
                        opacity_threshold=dynamic_opacity_threshold, max_points=dynamic_max_points,
                        voxel_size=0, normalization_center=center, normalization_scale=scale,
                    )
                export_gaussians_to_3dgs_ply(
                    combined_storage, ply_dir, f'combined_{scene_name}.ply',
                    opacity_threshold=static_opacity_threshold, max_points=max_points,
                    voxel_size=voxel_size, normalization_center=center, normalization_scale=scale,
                )
                if args.export_dynamic_frame_ply:
                    dynamic_frame_dir = os.path.join(ply_dir, f'dynamic_frames_{scene_name}')
                    dynamic_frame_manifest = []
                    frame_lookup = dict(export_dynamic_frames)
                    for frame_index in range(dy_map.shape[1]):
                        frame_storage = frame_lookup.get(frame_index)
                        if (frame_storage is not None and edit_operation == 'translation'
                                and frame_index in edit_masks):
                            selection = edit_masks[frame_index][
                                bg_mask[:, frame_index]
                                & (dy_prob[:, frame_index] >= args.ply_dynamic_threshold)
                            ].detach().cpu()
                            frame_storage = dict(frame_storage)
                            frame_storage['positions'] = apply_instance_translation(
                                frame_storage['positions'], selection,
                                instance_edit['value_by_frame'][frame_index]
                                if instance_edit.get('value_by_frame') else edit_value,
                            )
                        filename = f'dynamic_{frame_index:03d}.ply'
                        point_count = 0 if frame_storage is None else int(frame_storage['positions'].shape[0])
                        if point_count:
                            export_gaussians_to_3dgs_ply(
                                frame_storage, dynamic_frame_dir, filename,
                                opacity_threshold=dynamic_opacity_threshold, max_points=dynamic_max_points,
                                voxel_size=0, normalization_center=center, normalization_scale=scale,
                            )
                        dynamic_frame_manifest.append({
                            'frame_index': frame_index,
                            'timestamp': float(timestamps[frame_index]),
                            'point_count_before_opacity_filter': point_count,
                            'file': filename if point_count else None,
                        })
                    with open(os.path.join(ply_dir, f'dynamic_frames_{scene_name}.json'), 'w') as manifest_handle:
                        json.dump(dynamic_frame_manifest, manifest_handle, indent=2)
                if instance_edit is not None and len(edit_operations) == 1 and edit_operation not in ('remove', 'add', 'replace'):
                    frame_lookup = dict(export_dynamic_frames)
                    instance_dir = os.path.join(
                        args.output_path, scene_name, 'target_object',
                        f"instance_{instance_edit['instance_id']:04d}",
                    )
                    os.makedirs(instance_dir, exist_ok=True)
                    for frame_index, edit_mask in edit_masks.items():
                        export_dynamic_mask = bg_mask[:, frame_index] & (dy_prob[:, frame_index] >= args.ply_dynamic_threshold)
                        target_selection = edit_mask[export_dynamic_mask].detach().cpu()
                        frame_storage = frame_lookup.get(frame_index)
                        target_storage = None if frame_storage is None else {
                            key: values[target_selection] for key, values in frame_storage.items()
                        }
                        if target_storage is not None and len(target_storage['positions']):
                            selection = torch.ones(len(target_storage['positions']), dtype=torch.bool)
                            if edit_operation == 'translation':
                                target_storage['positions'] = apply_instance_translation(
                                    target_storage['positions'], selection,
                                    instance_edit['value_by_frame'][frame_index]
                                    if instance_edit.get('value_by_frame') else edit_value,
                                )
                            elif edit_operation == 'rotation':
                                target_storage['positions'], target_storage['rotation'] = apply_instance_rotation(
                                    target_storage['positions'], target_storage['rotation'], selection,
                                    edit_value, edit_centers[frame_index].cpu(),
                                )
                            else:
                                target_storage['positions'], target_storage['scales'] = apply_instance_scale(
                                    target_storage['positions'], target_storage['scales'], selection,
                                    edit_value, edit_centers[frame_index].cpu(),
                                )
                            export_gaussians_to_3dgs_ply(
                                target_storage, instance_dir, f'target_{frame_index:04d}.ply',
                                opacity_threshold=dynamic_opacity_threshold, max_points=dynamic_max_points,
                                voxel_size=0, normalization_center=center, normalization_scale=scale,
                            )
                        print(
                            f"Edited instance {instance_edit['instance_id']}, frame {frame_index}: "
                            f"exported {int(target_selection.sum())} selected Gaussians to {instance_dir}"
                        )
            if ply_normalization is not None:
                scene_out_dir = os.path.join(args.output_path, scene_name)
                os.makedirs(scene_out_dir, exist_ok=True)
                center = torch.as_tensor(ply_normalization['center'], dtype=extrinsic.dtype, device=extrinsic.device)
                scale = ply_normalization['scale']
                ply_extrinsic = extrinsic.clone()
                ply_extrinsic[:, :3, 3] = (
                    extrinsic[:, :3, 3] + extrinsic[:, :3, :3] @ center
                ) / scale
                ply_camera_params = {
                    'coordinate_convention': ply_normalization['coordinate_convention'],
                    'ply_world_transform': {
                        'formula': ply_normalization['formula'],
                        'center': ply_normalization['center'].tolist(),
                        'scale': scale,
                    },
                    'frames': [
                        {
                            'frame_index': i,
                            'intrinsics': intrinsic[i].detach().float().cpu().tolist(),
                            'world_to_camera_opencv': ply_extrinsic[i].detach().float().cpu().tolist(),
                        }
                        for i in range(ply_extrinsic.shape[0])
                    ],
                }
                with open(os.path.join(scene_out_dir, 'ply_camera_params.json'), 'w') as camera_handle:
                    json.dump(ply_camera_params, camera_handle, indent=2)

                def write_supersplat_poses(filename, matrices):
                    rotations = matrices[:, :3, :3]
                    translations = matrices[:, :3, 3]
                    positions = -(rotations.transpose(1, 2) @ translations.unsqueeze(-1)).squeeze(-1)
                    targets = positions + rotations.transpose(1, 2)[:, :, 2]
                    supersplat_positions = positions.clone()
                    supersplat_targets = targets.clone()
                    supersplat_positions[:, 0] *= -1
                    supersplat_targets[:, 0] *= -1
                    payload = {
                        'coordinate_convention': 'SuperSplat position/target convention; X is mirrored from GSRecon OpenCV PLY coordinates',
                        'frames': [
                            {
                                'frame_index': i,
                                'position': supersplat_positions[i].detach().float().cpu().tolist(),
                                'target': supersplat_targets[i].detach().float().cpu().tolist(),
                                'intrinsics': intrinsic[i].detach().float().cpu().tolist(),
                            }
                            for i in range(matrices.shape[0])
                        ],
                    }
                    with open(os.path.join(scene_out_dir, filename), 'w') as camera_handle:
                        json.dump(payload, camera_handle, indent=2)

                write_supersplat_poses('supersplat_camera_poses.json', ply_extrinsic)
                for shift in args.lateral_shifts:
                    rotation_wc = extrinsic[:, :3, :3]
                    center_wc = -(rotation_wc.transpose(1, 2) @ extrinsic[:, :3, 3:]).squeeze(-1)
                    left_world = -rotation_wc.transpose(1, 2)[:, :, 0]
                    shifted_center = center_wc + left_world * float(shift)
                    shifted_t = -(rotation_wc @ shifted_center.unsqueeze(-1)).squeeze(-1)
                    ply_shifted_extrinsic = extrinsic.clone()
                    ply_shifted_extrinsic[:, :3, 3] = (shifted_t + rotation_wc @ center) / scale
                    shift_name = str(float(shift)).replace('-', 'm').replace('.', 'p')
                    ply_shifted_camera_params = {
                        'coordinate_convention': ply_normalization['coordinate_convention'],
                        'lateral_shift_left_model_units': float(shift),
                        'matching_video': f'lateral_shift_{shift_name}.mp4',
                        'ply_world_transform': ply_camera_params['ply_world_transform'],
                        'frames': [
                            {
                                'frame_index': i,
                                'intrinsics': intrinsic[i].detach().float().cpu().tolist(),
                                'world_to_camera_opencv': ply_shifted_extrinsic[i].detach().float().cpu().tolist(),
                            }
                            for i in range(ply_shifted_extrinsic.shape[0])
                        ],
                    }
                    shifted_pose_path = os.path.join(scene_out_dir, f'ply_camera_params_lateral_shift_{shift_name}.json')
                    with open(shifted_pose_path, 'w') as camera_handle:
                        json.dump(ply_shifted_camera_params, camera_handle, indent=2)
                    write_supersplat_poses(f'supersplat_camera_poses_lateral_shift_{shift_name}.json', ply_shifted_extrinsic)

                if args.ply_only:
                    scene_idx += 1
                    continue

            if args.export_render_state:
                def render_cloud(points, colors, opacity, scales, quats):
                    return {key: value.detach().float().cpu() for key, value in zip(
                        ('means', 'colors', 'opacities', 'scales', 'quats'),
                        (points, colors, opacity, scales, quats))}
                state = {
                    'format': 'gsrecon_mode2_render_state_v1',
                    'static': render_cloud(static_points, static_rgbs, static_opacity, static_scales, static_rotations),
                    'static_times': gs_timestamps.detach().float().cpu(),
                    'static_confidence': static_gs_conf.detach().float().cpu(),
                    'static_pixel_indices': torch.nonzero(static_mask, as_tuple=False)[:, 1:].cpu(),
                    'dynamic': [render_cloud(*cloud) for cloud in zip(dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations)],
                    'dynamic_pixel_indices': [torch.nonzero(bg_mask[:, i], as_tuple=False)[:, 1:].cpu() for i in range(len(dynamic_points))],
                    'sky': {('means' if key == 'positions' else 'opacities' if key == 'opacity' else 'quats' if key == 'rotation' else key): value.detach().float().cpu() for key, value in export_sky.items()},
                    'times': timestamps.detach().float().cpu(),
                    'frame_seconds': torch.arange(len(dynamic_points), dtype=torch.float64) / args.output_fps,
                    'world_to_camera': extrinsic.detach().float().cpu(),
                    'intrinsics': intrinsic.detach().float().cpu(), 'height': H, 'width': W,
                }
                state_dir = os.path.join(args.output_path, scene_name)
                os.makedirs(state_dir, exist_ok=True)
                torch.save(state, os.path.join(state_dir, 'render_state.pt'))
                del state
                print(f'Saved lossless temporal render state to {state_dir}/render_state.pt', flush=True)

            render_extrinsic = extrinsic
            if ego_trajectory_offsets is not None:
                shifted = apply_ego_trajectory_offsets(extrinsic.detach().cpu().numpy(), ego_trajectory_offsets)
                render_extrinsic = torch.as_tensor(shifted, device=extrinsic.device, dtype=extrinsic.dtype)
            chunked_renders, chunked_alphas = [], []
            edit_before_chunks = {}
            if args.mode == 3:
                origin_extrinsic = extrinsic
                origin_intrinsic = intrinsic
            for idx in range(dy_map.shape[1]):
                if args.mode == 3:
                    t0 = render_timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = gs_conf , gamma1 = 0.1)###

                    world_points, rgbs, opacity, scales, rotation = concat_list(
                        [static_points, static_rgbs, static_opacity_, static_scales, static_rotation],
                        [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                    )
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points,
                        quats=rotation,
                        scales=scales,
                        opacities=opacity,
                        colors=rgbs,
                        viewmats=render_extrinsic[idx:idx+1],
                        Ks=intrinsic[idx:idx+1],
                        width=W,
                        height=H,
                        render_mode='RGB+ED',
                    )
                if args.mode == 2:
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    if dynamic_points:
                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            static_gs_list,
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                        )
                    else:
                        world_points, rgbs, opacity, scales, rotation = static_gs_list
                    height_offset = 0
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points,
                        quats=rotation,
                        scales=scales,
                        opacities=opacity,
                        colors=rgbs,
                        viewmats=render_extrinsic[idx:idx+1],
                        Ks=intrinsic[idx:idx+1],
                        width=W,
                        height=H,
                        render_mode='RGB+ED',
                    )
                    if idx in edit_masks or (instance_edit is not None and any(operation['operation'] == 'add' for operation in edit_operations)):
                        edit_before_chunks[idx] = (renders_chunk, alphas_chunk)
                        frame_edit_value = (instance_edit.get('value_by_frame') or {}).get(idx, edit_value) if edit_operation != 'add' else None
                        if len(edit_operations) > 1:
                            edited_static_points = static_points
                            edited_static_rgbs = static_rgbs
                            edited_static_opacity = static_opacity_
                            edited_static_scales = static_scales
                            edited_static_rotations = static_rotations
                            edited_dynamic_points = dynamic_points[idx]
                            edited_dynamic_rgbs = dynamic_rgbs[idx]
                            edited_dynamic_opacity = dynamic_opacitys[idx]
                            edited_dynamic_scales = dynamic_scales[idx]
                            edited_dynamic_rotations = dynamic_rotations[idx]
                            for operation_index, operation in enumerate(edit_operations):
                                operation_name = operation['operation']
                                dynamic_selection = operation_dynamic_edit_selections[operation_index].get(
                                    idx, torch.zeros_like(edited_dynamic_points[:, 0], dtype=torch.bool)
                                )
                                if operation_name == 'translation':
                                    value = operation.get('value_by_frame', {}).get(idx, operation.get('value'))
                                    if value is not None:
                                        edited_static_points = apply_instance_translation(
                                            edited_static_points, operation_static_union_selections[operation_index], value
                                        )
                                        edited_dynamic_points = apply_instance_translation(
                                            edited_dynamic_points, dynamic_selection, value
                                        )
                                elif operation_name == 'rotation' and idx in operation_edit_centers[operation_index]:
                                    center = edited_dynamic_points[dynamic_selection].mean(dim=0)
                                    edited_static_points, edited_static_rotations = apply_instance_rotation(
                                        edited_static_points, edited_static_rotations,
                                        operation_static_edit_selections[operation_index].get(idx, torch.zeros_like(edited_static_points[:, 0], dtype=torch.bool)),
                                        operation['value'], center,
                                    )
                                    edited_dynamic_points, edited_dynamic_rotations = apply_instance_rotation(
                                        edited_dynamic_points, edited_dynamic_rotations, dynamic_selection, operation['value'], center
                                    )
                                elif operation_name == 'scale' and idx in operation_edit_centers[operation_index]:
                                    center = edited_dynamic_points[dynamic_selection].mean(dim=0)
                                    static_selection = operation_static_edit_selections[operation_index].get(idx, torch.zeros_like(edited_static_points[:, 0], dtype=torch.bool))
                                    edited_static_points, edited_static_scales = apply_instance_scale(edited_static_points, edited_static_scales, static_selection, operation['value'], center)
                                    edited_dynamic_points, edited_dynamic_scales = apply_instance_scale(edited_dynamic_points, edited_dynamic_scales, dynamic_selection, operation['value'], center)
                            static_remove = torch.zeros_like(edited_static_points[:, 0], dtype=torch.bool)
                            dynamic_remove = torch.zeros_like(edited_dynamic_points[:, 0], dtype=torch.bool)
                            for operation_index, operation in enumerate(edit_operations):
                                if operation['operation'] in ('remove', 'replace'):
                                    static_remove |= operation_static_union_selections[operation_index]
                                    dynamic_remove |= operation_dynamic_edit_selections[operation_index].get(idx, torch.zeros_like(dynamic_remove))
                            if static_remove.any():
                                edited_static_points, edited_static_rgbs, edited_static_opacity, edited_static_scales, edited_static_rotations = remove_selected_gaussians(static_remove, edited_static_points, edited_static_rgbs, edited_static_opacity, edited_static_scales, edited_static_rotations)
                            if dynamic_remove.any():
                                edited_dynamic_points, edited_dynamic_rgbs, edited_dynamic_opacity, edited_dynamic_scales, edited_dynamic_rotations = remove_selected_gaussians(dynamic_remove, edited_dynamic_points, edited_dynamic_rgbs, edited_dynamic_opacity, edited_dynamic_scales, edited_dynamic_rotations)
                            world_points, rgbs, opacity, scales, rotation = concat_list(
                                [edited_static_points, edited_static_rgbs, edited_static_opacity, edited_static_scales, edited_static_rotations],
                                [edited_dynamic_points, edited_dynamic_rgbs, edited_dynamic_opacity, edited_dynamic_scales, edited_dynamic_rotations]
                            )
                            for operation_index, operation in enumerate(edit_operations):
                                if operation['operation'] == 'replace' and idx in operation_edit_centers[operation_index]:
                                    replacement = operation_replacement_gaussians[operation_index]
                                    world_points, rgbs, opacity, scales, rotation = concat_list(
                                        [world_points, rgbs, opacity, scales, rotation],
                                        [replacement['means'] + operation_edit_centers[operation_index][idx], replacement['colors'], replacement['opacities'], replacement['scales'], replacement['quaternions']]
                                    )
                                elif operation['operation'] == 'add':
                                    added = operation_added_gaussians[operation_index]
                                    world_points, rgbs, opacity, scales, rotation = concat_list(
                                        [world_points, rgbs, opacity, scales, rotation],
                                        [added["means"], added["colors"], added["opacities"], added["scales"], added["quaternions"]]
                                    )
                        else:
                            edited_static_rgbs = static_rgbs
                            edited_static_opacity = static_opacity_
                            edited_dynamic_rgbs = dynamic_rgbs[idx]
                            edited_dynamic_opacity = dynamic_opacitys[idx]
                            if edit_operation == 'translation':
                                edited_static_points = apply_instance_translation(
                                    static_points, static_edit_selection, frame_edit_value
                                )
                                edited_dynamic_points = apply_instance_translation(
                                    dynamic_points[idx], dynamic_edit_selections[idx], frame_edit_value
                                )
                                edited_static_rotations = static_rotations
                                edited_dynamic_rotations = dynamic_rotations[idx]
                                edited_static_scales = static_scales
                                edited_dynamic_scales = dynamic_scales[idx]
                            elif edit_operation == 'rotation':
                                edited_static_points, edited_static_rotations = apply_instance_rotation(
                                    static_points, static_rotations, static_edit_selections[idx],
                                    edit_value, edit_centers[idx],
                                )
                                edited_dynamic_points, edited_dynamic_rotations = apply_instance_rotation(
                                    dynamic_points[idx], dynamic_rotations[idx], dynamic_edit_selections[idx],
                                    edit_value, edit_centers[idx],
                                )
                                edited_static_scales = static_scales
                                edited_dynamic_scales = dynamic_scales[idx]
                            elif edit_operation == 'scale':
                                edited_static_points, edited_static_scales = apply_instance_scale(
                                    static_points, static_scales, static_edit_selections[idx],
                                    edit_value, edit_centers[idx],
                                )
                                edited_dynamic_points, edited_dynamic_scales = apply_instance_scale(
                                    dynamic_points[idx], dynamic_scales[idx], dynamic_edit_selections[idx],
                                    edit_value, edit_centers[idx],
                                )
                                edited_static_rotations = static_rotations
                                edited_dynamic_rotations = dynamic_rotations[idx]
                            elif edit_operation in ('remove', 'replace'):
                                (
                                    edited_static_points, edited_static_rgbs, edited_static_opacity,
                                    edited_static_scales, edited_static_rotations,
                                ) = remove_selected_gaussians(
                                    static_edit_selection, static_points, static_rgbs, static_opacity_,
                                    static_scales, static_rotations,
                                )
                                (
                                    edited_dynamic_points, edited_dynamic_rgbs, edited_dynamic_opacity,
                                    edited_dynamic_scales, edited_dynamic_rotations,
                                ) = remove_selected_gaussians(
                                    dynamic_edit_selections[idx], dynamic_points[idx], dynamic_rgbs[idx],
                                    dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx],
                                )
                            world_points, rgbs, opacity, scales, rotation = concat_list(
                                [edited_static_points, edited_static_rgbs, edited_static_opacity, edited_static_scales, edited_static_rotations],
                                [edited_dynamic_points, edited_dynamic_rgbs, edited_dynamic_opacity, edited_dynamic_scales, edited_dynamic_rotations]
                            )
                            if edit_operation == 'replace':
                                replacement_means = replacement_gaussians['means'] + edit_centers[idx]
                                world_points, rgbs, opacity, scales, rotation = concat_list(
                                    [world_points, rgbs, opacity, scales, rotation],
                                    [
                                        replacement_means, replacement_gaussians['colors'],
                                        replacement_gaussians['opacities'], replacement_gaussians['scales'],
                                        replacement_gaussians['quaternions'],
                                    ],
                                )
                        renders_chunk, alphas_chunk, _ = rasterization(
                            means=world_points, quats=rotation, scales=scales, opacities=opacity, colors=rgbs,
                            viewmats=render_extrinsic[idx:idx + 1], Ks=intrinsic[idx:idx + 1], width=W, height=H,
                            render_mode='RGB+ED',
                        )
                chunked_renders.append(renders_chunk)
                chunked_alphas.append(alphas_chunk)
                if args.mode == 2 and args.lateral_shifts:
                    rotation_wc = extrinsic[idx:idx + 1, :3, :3]
                    center_wc = -(rotation_wc.transpose(1, 2) @ extrinsic[idx:idx + 1, :3, 3:]).squeeze(-1)
                    left_world = -rotation_wc.transpose(1, 2)[:, :, 0]
                    for shift in args.lateral_shifts:
                        shifted_center = center_wc + left_world * float(shift)
                        shifted_t = -(rotation_wc @ shifted_center.unsqueeze(-1)).squeeze(-1)
                        shifted_extrinsic = extrinsic[idx:idx + 1].clone()
                        shifted_extrinsic[:, :3, 3] = shifted_t
                        shifted_chunk, shifted_alpha, _ = rasterization(means=world_points, quats=rotation, scales=scales, opacities=opacity, colors=rgbs, viewmats=shifted_extrinsic, Ks=intrinsic[idx:idx + 1], width=W, height=H, render_mode='RGB+ED')
                        shifted_bg = model.sky_model(images[:, idx:idx + 1], shifted_extrinsic, intrinsic[idx:idx + 1])
                        lateral_renders[float(shift)].append(shifted_chunk[..., :-1] * shifted_alpha + (1 - shifted_alpha) * shifted_bg)
                        lateral_alphas[float(shift)].append(shifted_alpha)
                        lateral_sky_masks[float(shift)].append((shifted_alpha < 0.5).float())
            renders = torch.cat(chunked_renders, dim=0)
            depth_maps = renders[..., -1]
            renders = renders[..., :-1]
            alphas = torch.cat(chunked_alphas, dim=0)
            if args.mode == 3:
                bg_render = model.sky_model.forward_with_new_pose(images,origin_extrinsic,origin_intrinsic, extrinsic, intrinsic)
            if args.mode == 2:
                bg_render = model.sky_model(images, render_extrinsic, intrinsic)
            renders = alphas * renders + (1 - alphas) * bg_render
            rendered_image = renders.permute(0, 3, 1, 2)
            if instance_edit is not None and edit_operation == 'add':
                target_dir = os.path.join(args.output_path, scene_name, 'target_object', 'added_asset')
                os.makedirs(target_dir, exist_ok=True)
                for edit_frame in range(dy_map.shape[1]):
                    edit_before_chunk, edit_before_alpha = edit_before_chunks[edit_frame]
                    before = edit_before_alpha * edit_before_chunk[..., :-1] + (1 - edit_before_alpha) * bg_render[edit_frame:edit_frame + 1]
                    T.ToPILImage()(before[0].permute(2, 0, 1).cpu().clamp(0, 1)).save(
                        os.path.join(target_dir, f'render_before_{edit_frame:04d}.png'))
                    T.ToPILImage()(rendered_image[edit_frame].cpu().clamp(0, 1)).save(
                        os.path.join(target_dir, f'render_after_{edit_frame:04d}.png'))
            elif instance_edit is not None:
                target_dir = os.path.join(
                    args.output_path, scene_name, "target_object",
                    f"instance_{instance_edit['instance_id']:04d}" if len(edit_operations) == 1 else "combined_edit",
                )
                os.makedirs(target_dir, exist_ok=True)
                for edit_frame, edit_mask in edit_masks.items():
                    edit_before_chunk, edit_before_alpha = edit_before_chunks[edit_frame]
                    before = edit_before_alpha * edit_before_chunk[..., :-1] + (1 - edit_before_alpha) * bg_render[edit_frame:edit_frame + 1]
                    T.ToPILImage()(before[0].permute(2, 0, 1).cpu().clamp(0, 1)).save(
                        os.path.join(target_dir, f'render_before_{edit_frame:04d}.png'))
                    T.ToPILImage()(rendered_image[edit_frame].cpu().clamp(0, 1)).save(
                        os.path.join(target_dir, f'render_after_{edit_frame:04d}.png'))
                    target_mask = edit_mask[0].detach().cpu().numpy()
                    Image.fromarray(target_mask.astype(np.uint8) * 255).save(
                        os.path.join(target_dir, f'mask_{edit_frame:04d}.png'))
                    rgb = _to_uint8_rgb(model_input_images[edit_frame])
                    overlay = rgb.copy()
                    overlay[target_mask] = (0.5 * rgb[target_mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                    Image.fromarray(overlay).save(os.path.join(target_dir, f'overlay_{edit_frame:04d}.png'))
                    if len(edit_operations) == 1 and edit_operation == 'replace':
                        print(
                            f"Replaced instance {instance_edit['instance_id']} in frame {edit_frame}: "
                            f"dynamic_removed={int(dynamic_edit_selections[edit_frame].sum())}, "
                            f"static_overlap_removed={int(static_edit_selections[edit_frame].sum())}, "
                            f"replacement={len(replacement_gaussians['means'])}"
                        )
                    elif len(edit_operations) == 1:
                        print(
                            f"Edited instance {instance_edit['instance_id']} in frame {edit_frame}: "
                            f"dynamic={int(dynamic_edit_selections[edit_frame].sum())}, "
                            f"static_overlap={int(static_edit_selections[edit_frame].sum())}, "
                            f"{edit_operation}={frame_edit_value}"
                        )
            target_image = images[0]
            if not args.metrics_only:
                os.makedirs(os.path.join(args.output_path, str(scene_idx).zfill(3)), exist_ok=True)
            for shift, frames in lateral_renders.items():
                shifted_video = (torch.cat(frames, dim=0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                shift_name = str(shift).replace('-', 'm').replace('.', 'p')
                shift_dir = os.path.join(args.output_path, str(scene_idx).zfill(3), f'lateral_shift_{shift_name}')
                os.makedirs(shift_dir, exist_ok=True)
                for frame_index, frame in enumerate(shifted_video):
                    Image.fromarray(frame).save(os.path.join(shift_dir, f'{frame_index:05d}.png'))
                alpha_dir = os.path.join(shift_dir, 'opacity')
                sky_dir = os.path.join(shift_dir, 'sky_masks')
                os.makedirs(alpha_dir, exist_ok=True)
                os.makedirs(sky_dir, exist_ok=True)
                alpha_video = torch.cat(lateral_alphas[float(shift)], dim=0).detach().cpu().clamp(0, 1).numpy()
                sky_video = torch.cat(lateral_sky_masks[float(shift)], dim=0).detach().cpu().numpy()
                for frame_index, (alpha_frame, sky_frame) in enumerate(zip(alpha_video, sky_video)):
                    Image.fromarray((alpha_frame[..., 0] * 255).astype(np.uint8), mode='L').save(os.path.join(alpha_dir, f'{frame_index:05d}.png'))
                    Image.fromarray((sky_frame[..., 0] * 255).astype(np.uint8), mode='L').save(os.path.join(sky_dir, f'{frame_index:05d}.png'))
                imageio.mimwrite(os.path.join(args.output_path, str(scene_idx).zfill(3), f'lateral_shift_{shift_name}.mp4'), shifted_video, fps=args.output_fps, codec='libx264')

            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)

            psnr, ssim, lpip = compute_metrics(rendered_image, target_image, loss_fn)
            metric_record = {
                'scene_id': scene_names[scene_idx - 1], 'output_scene_id': scene_name,
                'psnr': psnr, 'ssim': ssim, 'lpips': lpip, 'inference_time_s': inference_time,
                'mode': args.mode, 'input_views': args.input_views,
                'sequence_length': args.sequence_length, 'intervals': args.intervals,
            }
            if ego_trajectory_offsets is not None:
                metric_record['source_view_metrics_invalid'] = True
                metric_record['ego_trajectory_offsets'] = args.ego_trajectory_offsets
            if args.mode == 2:
                metric_sky_mask = batch['metric_sky_mask']
                if metric_sky_mask.ndim != 5 or metric_sky_mask.shape[0] != 1 or metric_sky_mask.shape[2] != 1:
                    raise ValueError(f"Expected metric_sky_mask [1,N,1,H,W], got {tuple(metric_sky_mask.shape)}")
                metric_dynamic_mask = None
                if 'metric_dynamic_mask' in batch:
                    metric_dynamic_mask = batch['metric_dynamic_mask']
                    if metric_dynamic_mask.ndim != 5 or metric_dynamic_mask.shape[0] != 1 or metric_dynamic_mask.shape[2] != 1:
                        raise ValueError(f"Expected metric_dynamic_mask [1,N,1,H,W], got {tuple(metric_dynamic_mask.shape)}")
                    metric_dynamic_mask = metric_dynamic_mask[0, :, 0].detach().cpu().numpy()
                elif not warned_missing_dynamic_mask:
                    print('[Mode2 metrics] GT dynamic mask is unavailable; dynamic metrics will be N/A.')
                    warned_missing_dynamic_mask = True
                sky_eval_mask, non_sky_mask, dynamic_eval_mask = _metric_evaluation_masks(
                    metric_sky_mask[0, :, 0].detach().cpu().numpy(),
                    metric_dynamic_mask,
                    tuple(depth_maps.shape),
                )

                dynamic_mask_metrics = (
                    _dynamic_mask_metrics(
                        dy_prob[0].detach().float().cpu().numpy() >= 0.5,
                        metric_dynamic_mask > 0.5,
                        non_sky_mask,
                    )
                    if metric_dynamic_mask is not None else
                    {
                        'tp': 0, 'fp': 0, 'fn': 0,
                        'precision': None, 'recall': None, 'iou': None, 'f1': None,
                    }
                )

                sky_rgb = _masked_rgb_metrics(rendered_image, target_image, sky_eval_mask)
                non_sky_rgb = _masked_rgb_metrics(rendered_image, target_image, non_sky_mask)
                dynamic_rgb = (
                    _masked_rgb_metrics(rendered_image, target_image, dynamic_eval_mask)
                    if dynamic_eval_mask is not None else
                    {'psnr': None, 'ssim': None, 'images': 0, 'pixels': 0, 'psnr_values': [], 'ssim_values': []}
                )
                view_rgb_metrics = {}
                if args.input_views == 3:
                    for view_name, view_slice in _three_view_slices(rendered_image.shape[0]).items():
                        view_psnr, view_ssim, view_lpips = compute_metrics(
                            rendered_image[view_slice], target_image[view_slice], loss_fn
                        )
                        view_rgb_metrics[view_name] = {
                            'psnr': view_psnr,
                            'ssim': view_ssim,
                            'lpips': view_lpips,
                            'sky': _masked_rgb_metrics(
                                rendered_image[view_slice], target_image[view_slice], sky_eval_mask[view_slice]
                            ),
                            'non_sky': _masked_rgb_metrics(
                                rendered_image[view_slice], target_image[view_slice], non_sky_mask[view_slice]
                            ),
                            'dynamic': (
                                _masked_rgb_metrics(
                                    rendered_image[view_slice], target_image[view_slice], dynamic_eval_mask[view_slice]
                                )
                                if dynamic_eval_mask is not None else
                                {'psnr': None, 'ssim': None, 'images': 0, 'pixels': 0,
                                 'psnr_values': [], 'ssim_values': []}
                            ),
                        }
                rendered_depth = depth_maps.detach().cpu().numpy()
                if gt_depth.ndim == 5:
                    gt_depth_mode2 = gt_depth[0, ..., 0].detach().cpu().numpy()
                elif gt_depth.ndim == 4:
                    gt_depth_mode2 = gt_depth[0].detach().cpu().numpy()
                else:
                    gt_depth_mode2 = None
                depth_metrics = (
                    _aligned_depth_metrics(rendered_depth, gt_depth_mode2, non_sky_mask, dynamic_eval_mask)
                    if gt_depth_mode2 is not None else _empty_depth_metrics()
                )
                if not depth_metrics['valid'] and not warned_missing_depth:
                    print('[Mode2 metrics] No usable GT depth for scene-level alignment; depth metrics will be N/A.')
                    warned_missing_depth = True

                mode2_metrics['evaluated_scenes'] += 1
                mode2_metrics['evaluated_images'] += rendered_image.shape[0]
                _accumulate_masked_rgb_metrics(mode2_metrics, 'sky', sky_rgb)
                _accumulate_masked_rgb_metrics(mode2_metrics, 'non_sky', non_sky_rgb)
                _accumulate_masked_rgb_metrics(mode2_metrics, 'dynamic', dynamic_rgb)
                mode2_metrics['valid_sky_images'] += sky_rgb['images']
                mode2_metrics['valid_sky_pixels'] += sky_rgb['pixels']
                mode2_metrics['valid_non_sky_images'] += non_sky_rgb['images']
                mode2_metrics['valid_non_sky_pixels'] += non_sky_rgb['pixels']
                mode2_metrics['valid_dynamic_images'] += dynamic_rgb['images']
                mode2_metrics['valid_dynamic_pixels'] += dynamic_rgb['pixels']
                mode2_metrics['dynamic_mask_tp'] += dynamic_mask_metrics['tp']
                mode2_metrics['dynamic_mask_fp'] += dynamic_mask_metrics['fp']
                mode2_metrics['dynamic_mask_fn'] += dynamic_mask_metrics['fn']
                for view_name, view_metrics in view_rgb_metrics.items():
                    mode2_metrics[f'{view_name}_psnr'].append(view_metrics['psnr'])
                    mode2_metrics[f'{view_name}_ssim'].append(view_metrics['ssim'])
                    mode2_metrics[f'{view_name}_lpips'].append(view_metrics['lpips'])
                    for region in ('sky', 'non_sky', 'dynamic'):
                        _accumulate_masked_rgb_metrics(
                            mode2_metrics, f'{view_name}_{region}', view_metrics[region]
                        )
                if depth_metrics['valid']:
                    mode2_metrics['depth_alignment_valid_scenes'] += 1
                else:
                    mode2_metrics['depth_alignment_invalid_scenes'] += 1
                _accumulate_depth_stats(mode2_metrics['depth'], depth_metrics['depth_values'])
                _accumulate_depth_stats(mode2_metrics['dynamic_depth'], depth_metrics['dynamic_depth_values'])

                metric_record.update({
                    'sky_psnr': sky_rgb['psnr'], 'sky_ssim': sky_rgb['ssim'],
                    'non_sky_psnr': non_sky_rgb['psnr'], 'non_sky_ssim': non_sky_rgb['ssim'],
                    'dynamic_psnr': dynamic_rgb['psnr'], 'dynamic_ssim': dynamic_rgb['ssim'],
                    'dynamic_mask_iou': dynamic_mask_metrics['iou'],
                    'dynamic_mask_precision': dynamic_mask_metrics['precision'],
                    'dynamic_mask_recall': dynamic_mask_metrics['recall'],
                    'dynamic_mask_f1': dynamic_mask_metrics['f1'],
                    'dynamic_mask_tp': dynamic_mask_metrics['tp'],
                    'dynamic_mask_fp': dynamic_mask_metrics['fp'],
                    'dynamic_mask_fn': dynamic_mask_metrics['fn'],
                    'aligned_depth_rmse_0_100m': depth_metrics['depth']['0_100m']['rmse'],
                    'aligned_depth_rmse_100_200m': depth_metrics['depth']['100_200m']['rmse'],
                    'aligned_depth_rmse_0_200m': depth_metrics['depth']['0_200m']['rmse'],
                    'dynamic_aligned_depth_rmse_0_100m': depth_metrics['dynamic_depth']['0_100m']['rmse'],
                    'dynamic_aligned_depth_rmse_100_200m': depth_metrics['dynamic_depth']['100_200m']['rmse'],
                    'dynamic_aligned_depth_rmse_0_200m': depth_metrics['dynamic_depth']['0_200m']['rmse'],
                    'depth_alignment_scale': depth_metrics['scale'],
                    'depth_alignment_offset': depth_metrics['offset'],
                    'valid_sky_images': sky_rgb['images'], 'valid_sky_pixels': sky_rgb['pixels'],
                    'valid_non_sky_images': non_sky_rgb['images'], 'valid_non_sky_pixels': non_sky_rgb['pixels'],
                    'valid_dynamic_images': dynamic_rgb['images'], 'valid_dynamic_pixels': dynamic_rgb['pixels'],
                    'depth_alignment_valid': depth_metrics['valid'],
                })
                for view_name, view_metrics in view_rgb_metrics.items():
                    metric_record.update({
                        f'{view_name}_psnr': view_metrics['psnr'],
                        f'{view_name}_ssim': view_metrics['ssim'],
                        f'{view_name}_lpips': view_metrics['lpips'],
                    })
                    for region in ('sky', 'non_sky', 'dynamic'):
                        metric_record[f'{view_name}_{region}_psnr'] = view_metrics[region]['psnr']
                        metric_record[f'{view_name}_{region}_ssim'] = view_metrics[region]['ssim']
                for prefix, stats in (('valid_depth', depth_metrics['depth']),
                                      ('valid_dynamic_depth', depth_metrics['dynamic_depth'])):
                    for range_name, values in stats.items():
                        metric_record[f'{prefix}_images_{range_name}'] = values['images']
                        metric_record[f'{prefix}_pixels_{range_name}'] = values['pixels']
            with open(per_scene_metrics_path, 'a') as metrics_handle:
                metrics_handle.write(json.dumps(metric_record, allow_nan=False, default=float) + '\n')
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpip)
            scene_idx += 1

            if args.metrics_only:
                continue

            scene_out_dir = os.path.join(args.output_path, scene_name)
            os.makedirs(scene_out_dir, exist_ok=True)

            if args.export_camera_params:
                camera_params = {
                    'coordinate_convention': 'OpenCV world-to-camera (x right, y down, z forward)',
                    'frames': [
                        {
                            'frame_index': i,
                            'intrinsics': intrinsic[i].detach().float().cpu().tolist(),
                            'world_to_camera_opencv': render_extrinsic[i].detach().float().cpu().tolist(),
                        }
                        for i in range(extrinsic.shape[0])
                    ],
                }
                with open(os.path.join(scene_out_dir, 'camera_params.json'), 'w') as camera_handle:
                    json.dump(camera_params, camera_handle, indent=2)

            if args.export_opacity_dir:
                opacity_dir = os.path.join(args.export_opacity_dir, scene_name, 'opacity')
                os.makedirs(opacity_dir, exist_ok=True)
                for i, alpha in enumerate(alphas):
                    alpha_u8 = np.round(alpha.squeeze(-1).detach().float().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
                    Image.fromarray(alpha_u8, mode='L').save(os.path.join(opacity_dir, f'{i:05d}.png'))

            if args.export_input_frames_dir:
                input_frames_dir = os.path.join(args.export_input_frames_dir, scene_name, 'images')
                os.makedirs(input_frames_dir, exist_ok=True)
                for i, image in enumerate(model_input_images):
                    T.ToPILImage()(image.cpu().clamp(0, 1)).save(os.path.join(input_frames_dir, f'{i:05d}.png'))

            if args.images:
                if args.input_views == 1:
                    image_list = []
                    for i in range(rendered_image.shape[0]):
                        rendered = rendered_image[i].detach().cpu().clamp(0, 1)
                        image_path = os.path.join(scene_out_dir, f"view_{i}.png")
                        T.ToPILImage()(rendered).save(image_path)
                        image_list.append(rendered.permute(1, 2, 0).numpy())
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, (np.array(image_list) * 255).astype(np.uint8), fps=args.output_fps, codec="libx264")
                if args.input_views == 3:
                    T_total = rendered_image.shape[0]
                    groups = T_total // 3
                    video_list = []
                    for g in range(groups):
                        idx_center = 3 * g + 0
                        idx_left = 3 * g + 1
                        idx_right = 3 * g + 2
                        center = rendered_image[idx_center].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        left = rendered_image[idx_left].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        right = rendered_image[idx_right].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        H, W = center.shape[0], center.shape[1]
                        # convert to uint8 HWC
                        def to_uint8(arr):
                            a = (arr * 255.0).astype(np.uint8)
                            if a.ndim == 2:
                                a = np.stack([a] * 3, axis=-1)
                            if a.shape[2] == 4:
                                a = a[:, :, :3]
                            return a
                        left_u = to_uint8(left)
                        center_u = to_uint8(center)
                        right_u = to_uint8(right)
                        white = np.ones((H,10, 3), dtype=np.uint8) * 255
                        composed = np.concatenate([left_u, white, center_u, white, right_u], axis=1)
                        # save image
                        image_path = os.path.join(scene_out_dir, f"view_{g:04d}.png")
                        Image.fromarray(composed).save(image_path)
                        video_list.append(composed)
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, np.array(video_list), fps=args.output_fps, codec="libx264")

            gt_frames = target_image.detach().cpu()
            pred_frames = rendered_image.detach().cpu()
            dyn_frames = (dy_prob[0] if args.mode == 2 else dy_map[0].sigmoid()).detach().cpu()
            if "dynamic_mask" in batch:
                gt_dy_map = batch["dynamic_mask"].to(device).mean(dim=2)
                gt_dy_map = gt_dy_map[0].sigmoid().detach().cpu()
            else:
                # 自定义视频没有真值动态掩码时，仅在对比视频中显示全黑图。
                gt_dy_map = torch.zeros_like(dyn_frames)
            if args.mode == 2:
                depth_frames = predictions["depth"][0].detach().cpu()
                gt_depth = gt_depth[..., 0:1]
                gt_depth = gt_depth[0].squeeze(-1).detach().cpu()
                sky_mask = sky_mask.detach().cpu()
            if args.mode == 3:
                depth_frames = depth_interp[0].detach().cpu()
                gt_depth = gt_depth[..., 0:1]
                gt_depth = gt_depth[0].squeeze(-1).detach().cpu()
                sky_mask = target_sky_masks.permute(0, 1, 3, 4, 2).detach().cpu()
            out_video = os.path.join(scene_out_dir, "comparison.mp4")
            make_comparison_video_quad(gt_frames, pred_frames, gt_dy_map, dyn_frames, gt_depth ,depth_frames, sky_mask, out_video, fps=args.output_fps, views = args.input_views )  #
            print("Saved comparison video:", out_video)

            if args.depth:
                S = depth_frames.shape[0]

                if args.input_views == 1:
                    for i in range(S):
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{i}.npy"), depth_i)
                elif args.input_views == 3:
                    for i in range(S):
                        view_id = i % 3
                        frame_id = i // 3
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{frame_id:04d}_{view_id}.npy"), depth_i)
    if args.metrics:
        summary = {
            'psnr': sum(psnr_list) / len(psnr_list),
            'ssim': sum(ssim_list) / len(ssim_list),
            'lpips': sum(lpips_list) / len(lpips_list),
            'avg_inference_time_s': sum(inference_time_list) / len(inference_time_list),
        }
        if args.mode == 2:
            depth_summary = _finalize_depth_stats(mode2_metrics['depth'])
            dynamic_depth_summary = _finalize_depth_stats(mode2_metrics['dynamic_depth'])
            dynamic_mask_tp = mode2_metrics['dynamic_mask_tp']
            dynamic_mask_fp = mode2_metrics['dynamic_mask_fp']
            dynamic_mask_fn = mode2_metrics['dynamic_mask_fn']
            dynamic_mask_precision = dynamic_mask_tp / (dynamic_mask_tp + dynamic_mask_fp) if dynamic_mask_tp + dynamic_mask_fp else None
            dynamic_mask_recall = dynamic_mask_tp / (dynamic_mask_tp + dynamic_mask_fn) if dynamic_mask_tp + dynamic_mask_fn else None
            dynamic_mask_iou = dynamic_mask_tp / (dynamic_mask_tp + dynamic_mask_fp + dynamic_mask_fn) if dynamic_mask_tp + dynamic_mask_fp + dynamic_mask_fn else None
            dynamic_mask_f1 = 2 * dynamic_mask_tp / (2 * dynamic_mask_tp + dynamic_mask_fp + dynamic_mask_fn) if 2 * dynamic_mask_tp + dynamic_mask_fp + dynamic_mask_fn else None
            summary.update({
                'sky_psnr': _mean_or_none(mode2_metrics['sky_psnr']),
                'sky_ssim': _mean_or_none(mode2_metrics['sky_ssim']),
                'non_sky_psnr': _mean_or_none(mode2_metrics['non_sky_psnr']),
                'non_sky_ssim': _mean_or_none(mode2_metrics['non_sky_ssim']),
                'dynamic_psnr': _mean_or_none(mode2_metrics['dynamic_psnr']),
                'dynamic_ssim': _mean_or_none(mode2_metrics['dynamic_ssim']),
                'dynamic_mask_iou': dynamic_mask_iou,
                'dynamic_mask_precision': dynamic_mask_precision,
                'dynamic_mask_recall': dynamic_mask_recall,
                'dynamic_mask_f1': dynamic_mask_f1,
                'dynamic_mask_tp': dynamic_mask_tp,
                'dynamic_mask_fp': dynamic_mask_fp,
                'dynamic_mask_fn': dynamic_mask_fn,
                'aligned_depth_rmse_0_100m': depth_summary['0_100m']['rmse'],
                'aligned_depth_rmse_100_200m': depth_summary['100_200m']['rmse'],
                'aligned_depth_rmse_0_200m': depth_summary['0_200m']['rmse'],
                'dynamic_aligned_depth_rmse_0_100m': dynamic_depth_summary['0_100m']['rmse'],
                'dynamic_aligned_depth_rmse_100_200m': dynamic_depth_summary['100_200m']['rmse'],
                'dynamic_aligned_depth_rmse_0_200m': dynamic_depth_summary['0_200m']['rmse'],
                'evaluated_scenes': mode2_metrics['evaluated_scenes'],
                'evaluated_images': mode2_metrics['evaluated_images'],
                'valid_sky_images': mode2_metrics['valid_sky_images'],
                'valid_sky_pixels': mode2_metrics['valid_sky_pixels'],
                'valid_non_sky_images': mode2_metrics['valid_non_sky_images'],
                'valid_non_sky_pixels': mode2_metrics['valid_non_sky_pixels'],
                'valid_dynamic_images': mode2_metrics['valid_dynamic_images'],
                'valid_dynamic_pixels': mode2_metrics['valid_dynamic_pixels'],
                'depth_alignment_valid_scenes': mode2_metrics['depth_alignment_valid_scenes'],
                'depth_alignment_invalid_scenes': mode2_metrics['depth_alignment_invalid_scenes'],
            })
            for prefix, stats in (('valid_depth', depth_summary), ('valid_dynamic_depth', dynamic_depth_summary)):
                for range_name, values in stats.items():
                    summary[f'{prefix}_images_{range_name}'] = values['images']
                    summary[f'{prefix}_pixels_{range_name}'] = values['pixels']
            if args.input_views == 3:
                for view_name in ('front', 'front_left', 'front_right'):
                    summary[f'{view_name}_psnr'] = _mean_or_none(mode2_metrics[f'{view_name}_psnr'])
                    summary[f'{view_name}_ssim'] = _mean_or_none(mode2_metrics[f'{view_name}_ssim'])
                    summary[f'{view_name}_lpips'] = _mean_or_none(mode2_metrics[f'{view_name}_lpips'])
                    for region in ('sky', 'non_sky', 'dynamic'):
                        summary[f'{view_name}_{region}_psnr'] = _mean_or_none(
                            mode2_metrics[f'{view_name}_{region}_psnr']
                        )
                        summary[f'{view_name}_{region}_ssim'] = _mean_or_none(
                            mode2_metrics[f'{view_name}_{region}_ssim']
                        )

            def format_metric(value):
                return 'N/A' if value is None else f'{value:.6f}'

            print('Average PSNR:', format_metric(summary['psnr']))
            print('Average SSIM:', format_metric(summary['ssim']))
            print('Average LPIPS:', format_metric(summary['lpips']))
            print()
            print('Average Sky PSNR:', format_metric(summary['sky_psnr']))
            print('Average Sky SSIM:', format_metric(summary['sky_ssim']))
            print('Average Non-sky PSNR:', format_metric(summary['non_sky_psnr']))
            print('Average Non-sky SSIM:', format_metric(summary['non_sky_ssim']))
            print('Average Dynamic PSNR:', format_metric(summary['dynamic_psnr']))
            print('Average Dynamic SSIM:', format_metric(summary['dynamic_ssim']))
            print('Dynamic Mask IoU:', format_metric(summary['dynamic_mask_iou']))
            print('Dynamic Mask Precision:', format_metric(summary['dynamic_mask_precision']))
            print('Dynamic Mask Recall:', format_metric(summary['dynamic_mask_recall']))
            print('Dynamic Mask F1:', format_metric(summary['dynamic_mask_f1']))
            if args.input_views == 3:
                print()
                for view_name, label in (
                    ('front', 'Front'),
                    ('front_left', 'Front-left'),
                    ('front_right', 'Front-right'),
                ):
                    print(
                        f'{label} PSNR/SSIM/LPIPS:',
                        format_metric(summary[f'{view_name}_psnr']),
                        '/',
                        format_metric(summary[f'{view_name}_ssim']),
                        '/',
                        format_metric(summary[f'{view_name}_lpips']),
                    )
                    print(
                        f'{label} Sky PSNR/SSIM:',
                        format_metric(summary[f'{view_name}_sky_psnr']),
                        '/',
                        format_metric(summary[f'{view_name}_sky_ssim']),
                    )
                    print(
                        f'{label} Non-sky PSNR/SSIM:',
                        format_metric(summary[f'{view_name}_non_sky_psnr']),
                        '/',
                        format_metric(summary[f'{view_name}_non_sky_ssim']),
                    )
                    print(
                        f'{label} Dynamic PSNR/SSIM:',
                        format_metric(summary[f'{view_name}_dynamic_psnr']),
                        '/',
                        format_metric(summary[f'{view_name}_dynamic_ssim']),
                    )
            print()
            for range_name, label in (('0_100m', '0.01-100m'), ('100_200m', '100-200m'), ('0_200m', '0.01-200m')):
                print(f'Average Aligned Depth RMSE ({label}):', format_metric(depth_summary[range_name]['rmse']))
            print()
            for range_name, label in (('0_100m', '0.01-100m'), ('100_200m', '100-200m'), ('0_200m', '0.01-200m')):
                print(f'Average Dynamic Aligned Depth RMSE ({label}):', format_metric(dynamic_depth_summary[range_name]['rmse']))
            print()
            print('Average Inference Time:', format_metric(summary['avg_inference_time_s']), 's')
            print('Evaluated scenes:', summary['evaluated_scenes'])
            print('Evaluated images:', summary['evaluated_images'])
            print('Valid sky images/pixels:', summary['valid_sky_images'], '/', summary['valid_sky_pixels'])
            print('Valid non-sky images/pixels:', summary['valid_non_sky_images'], '/', summary['valid_non_sky_pixels'])
            print('Valid dynamic images/pixels:', summary['valid_dynamic_images'], '/', summary['valid_dynamic_pixels'])
            for range_name, label in (('0_100m', '0.01-100m'), ('100_200m', '100-200m'), ('0_200m', '0.01-200m')):
                print(f'Valid depth images/pixels ({label}):', depth_summary[range_name]['images'], '/', depth_summary[range_name]['pixels'])
            for range_name, label in (('0_100m', '0.01-100m'), ('100_200m', '100-200m'), ('0_200m', '0.01-200m')):
                print(f'Valid dynamic depth images/pixels ({label}):', dynamic_depth_summary[range_name]['images'], '/', dynamic_depth_summary[range_name]['pixels'])
            print('Depth alignment valid/invalid scenes:', summary['depth_alignment_valid_scenes'], '/', summary['depth_alignment_invalid_scenes'])
        else:
            print("PSNR:", summary['psnr'])
            print("SSIM:", summary['ssim'])
            print("LPIPS:", summary['lpips'])
            print("Avg Inference Time (s):", summary['avg_inference_time_s'])
        if args.summary_file:
            with open(args.summary_file, 'w') as summary_handle:
                json.dump(summary, summary_handle, indent=2, allow_nan=False, default=float)
        return summary

if __name__ == "__main__":
    main()
