# Modified by ADToolbox contributors, 2026: standalone reconstruction/edit/export extensions.
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
import os
import json
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF
import numpy as np

def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=target_size, mode="nearest")
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    return flow.squeeze()

    
def load_and_preprocess_flow(flow_path_list, extrinsic_paths, intrinsic_path, height, width):
    if len(flow_path_list) == 0:
        raise ValueError("At least 1 image is required")

    flows = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # print(f"[DEBUG] flow_path_list: {flow_path_list}")
    for i, flow_path in enumerate(flow_path_list):
        # print(f"[DEBUG] Processing flow_path[{i}]: {flow_path}")
        # print(f"[DEBUG] Is file: {os.path.isfile(flow_path) if flow_path else 'Empty/None'}")
        # print(f"[DEBUG] Is dir: {os.path.isdir(flow_path) if flow_path else 'Empty/None'}")
        depth_and_flow = np.load(flow_path)
        flow = depth_and_flow
        flow = torch.tensor(flow).float()
        flow = resize_flow(flow, (height, width))
        flows.append(flow)
    
    return torch.stack(flows)


def load_and_preprocess_images(image_path_list, mode="crop"):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    images = torch.stack(images)  # concatenate images
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_and_preprocess_masks(mask_path_list):
    """Load binary sky masks with image-matched crop geometry and nearest resize."""
    if len(mask_path_list) == 0:
        raise ValueError("At least 1 sky mask is required")

    masks = []
    shapes = set()
    target_size = 518
    for mask_path in mask_path_list:
        mask = Image.open(mask_path).convert("L")
        width, height = mask.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        mask = TF.ToTensor()(mask)
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            mask = mask[:, start_y : start_y + target_size, :]
        shapes.add((mask.shape[1], mask.shape[2]))
        masks.append(mask)

    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_masks = []
        for mask in masks:
            height_padding = max_height - mask.shape[1]
            width_padding = max_width - mask.shape[2]
            padded_masks.append(torch.nn.functional.pad(
                mask,
                (width_padding // 2, width_padding - width_padding // 2,
                 height_padding // 2, height_padding - height_padding // 2),
                mode="constant",
                value=1.0,
            ))
        masks = padded_masks
    return torch.stack(masks)


WAYMO_OPENCV_TO_DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _waymo_box_corners(size):
    length, width, height = size
    return np.array([
        [-length / 2, -length / 2, length / 2, length / 2,
         -length / 2, -length / 2, length / 2, length / 2],
        [width / 2, -width / 2, -width / 2, width / 2,
         width / 2, -width / 2, -width / 2, width / 2],
        [height / 2, height / 2, height / 2, height / 2,
         -height / 2, -height / 2, -height / 2, -height / 2],
    ])


def _project_waymo_instance_boxes(image_path_list, scene_dir, instances_info, frame_instances):
    """Project tracked Waymo 3D boxes with the repository's existing camera convention."""
    by_frame_camera = {}
    calibration = {}
    for image_path in image_path_list:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        frame_index, camera_index = (int(value) for value in stem.rsplit("_", 1))
        key = (frame_index, camera_index)
        if key in by_frame_camera:
            continue
        if camera_index not in calibration:
            values = np.loadtxt(
                os.path.join(scene_dir, "intrinsics", f"{camera_index}.txt")
            ).reshape(-1)
            intrinsic = np.array([
                [values[0], 0, values[2]],
                [0, values[1], values[3]],
                [0, 0, 1],
            ], dtype=np.float64)
            camera_to_ego = (
                np.loadtxt(os.path.join(scene_dir, "extrinsics", f"{camera_index}.txt"))
                @ WAYMO_OPENCV_TO_DATASET
            )
            calibration[camera_index] = (intrinsic, camera_to_ego)
        intrinsic, camera_to_ego = calibration[camera_index]
        camera_to_world = (
            np.loadtxt(os.path.join(scene_dir, "ego_pose", f"{frame_index:03d}.txt"))
            @ camera_to_ego
        )
        world_to_camera = np.linalg.inv(camera_to_world)
        projected = []
        for instance_id in frame_instances.get(str(frame_index), []):
            annotation = instances_info[str(instance_id)]
            annotation_frames = annotation["frame_annotations"]["frame_idx"]
            if frame_index not in annotation_frames:
                continue
            annotation_index = annotation_frames.index(frame_index)
            object_to_world = np.asarray(
                annotation["frame_annotations"]["obj_to_world"][annotation_index]
            )
            box_size = annotation["frame_annotations"]["box_size"][annotation_index]
            box_world = (
                object_to_world[:3, :3] @ _waymo_box_corners(box_size)
                + object_to_world[:3, 3:4]
            ).T
            box_camera = (
                world_to_camera[:3, :3] @ box_world.T
                + world_to_camera[:3, 3:4]
            ).T
            visible = box_camera[:, 2] > 1e-6
            if not visible.any():
                continue
            pixels = (intrinsic @ box_camera[visible].T).T
            pixels = pixels[:, :2] / pixels[:, 2:3]
            x0, y0 = pixels.min(axis=0)
            x1, y1 = pixels.max(axis=0)
            projected.append((int(instance_id), [x0, y0, x1, y1]))
        by_frame_camera[key] = projected
    return by_frame_camera


class WaymoOpenDataset(Dataset):
    def __init__(self, image_dir, scene_names = None, sequence_length= None, start_idx = -1, mode=1, views=1, intervals=2, continuous_frames=False, metric_data=False):
        #mode 1 : train
        #mode 2 : pure reconstruction
        #mode 3 : interplation
        #mode 4 : 20-frame interpolation and extrapolation
        
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.continuous_frames = continuous_frames
        self.metric_data = metric_data
        if mode == 1:
            interval = 1
        elif mode == 2:
            interval = 1
        elif mode in (3, 4, 5, 6):
            interval = intervals
        else:
            interval = 1
        self.interval =  interval
        self.mode = mode
        if mode == 1:
            test_mode = False
            load_flow = False
        elif mode == 2:
            test_mode = True
            load_flow = False
        elif mode in (3, 4):
            test_mode = True
            load_flow = True
        elif mode == 5:
            test_mode = False
            load_flow = False
        elif mode == 6:
            if views not in (1, 3) or sequence_length not in (2, 4) or intervals != 5:
                raise ValueError("mode=6 requires views=1 or 3, sequence_length=2/4, and intervals=5")
            test_mode = False
            load_flow = False
        else:
            pass
        self.test_mode = test_mode
        self.load_flow = load_flow
        self.views = views

        # Scan all scene folders and collect image paths
        if scene_names is None:
            scene_names = [] 
            scene_names_ = [f"{i:03d}" for i in range(0, 99)]
            scene_names = scene_names + scene_names_
        self.scenes = scene_names
        self.image_paths = []
        self.sky_mask_paths = []
        self.dynamic_mask_path = []
        self.extrinsic_paths = []
        self.intrinsic_paths = []
        self.semantic_mask_path = []
        self.depth_flow_paths = []
        self.ego_paths = []

        self.start_idx = start_idx

        for scene_name in scene_names:
            scene_path = os.path.join(image_dir, scene_name, "images")
            if os.path.isdir(scene_path):
                # image
                if self.views == 1:
                    image_paths = sorted(
                        [
                            os.path.join(scene_path, f)
                            for f in os.listdir(scene_path)
                            if f.endswith(("_0.jpg", "_0.png"))
                        ]
                    )
                    self.image_paths.append(image_paths)
                elif self.views == 3:
                    views_image_lists = []
                    for v in range(3):
                        suffixes = (f"_{v}.jpg", f"_{v}.png")
                        files_v = sorted(
                            [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith(suffixes)]
                        )
                        views_image_lists.append(files_v)
                    lengths = [len(l) for l in views_image_lists]
                    if len(set(lengths)) != 1:
                        raise RuntimeError(f"Inconsistent number of images across views in scene {scene_name}, lengths: {lengths}")
                    self.image_paths.append(views_image_lists)

                # sky_mask
                sky_mask_path = os.path.join(image_dir, scene_name, "sky_masks")
                if os.path.isdir(sky_mask_path):
                    if self.views == 1:
                        sky_mask_paths = sorted(
                            [os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.sky_mask_paths.append(sky_mask_paths)
                    elif self.views == 3:
                        views_sky_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(suffixes)])
                            views_sky_lists.append(files_v)
                        self.sky_mask_paths.append(views_sky_lists)
                else:
                    self.sky_mask_paths.append([] if self.views == 1 else [[] for _ in range(3)])

                # extrinsic
                extrinsic_path = os.path.join(image_dir, scene_name, "ego_pose")
                if os.path.isdir(extrinsic_path):
                    extrinsic_paths = sorted([
                        os.path.join(extrinsic_path, f)
                        for f in os.listdir(extrinsic_path)
                        if f.endswith(".txt")
                    ])
                    self.extrinsic_paths.append(extrinsic_paths)
                else:
                    self.extrinsic_paths.append([])

                # extrinsic
                ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                # ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                if os.path.isdir(ego_path):
                    ego_path =  os.path.join(ego_path, "0.txt")
                    self.ego_paths.append(ego_path)

                # intrinsic
                intrinsic_path = os.path.join(image_dir, scene_name, "intrinsics")
                if os.path.isdir(intrinsic_path):
                    if self.views == 1:
                        intrinsic_paths = os.path.join(intrinsic_path, "0.txt")
                        self.intrinsic_paths.append(intrinsic_paths)
                    elif self.views == 3:
                        intrinsics_views = []
                        for v in range(3):
                            p = os.path.join(intrinsic_path, f"{v}.txt")
                            intrinsics_views.append(p if os.path.exists(p) else "")
                        self.intrinsic_paths.append(intrinsics_views)
                else:
                    self.intrinsic_paths.append("" if self.views == 1 else ["" for _ in range(3)])

                # dynamic mask
                dynamic_mask_path = os.path.join(image_dir, scene_name, "fine_dynamic_masks/all")
                if os.path.isdir(dynamic_mask_path):
                    if self.views == 1:
                        dynamic_mask_paths = sorted(
                            [os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.dynamic_mask_path.append(dynamic_mask_paths)
                    elif self.views == 3:
                        views_dyn_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(suffixes)])
                            views_dyn_lists.append(files_v)
                        self.dynamic_mask_path.append(views_dyn_lists)
                else:
                    self.dynamic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])
                # depth
                depth_path = os.path.join(image_dir, scene_name, "depth_flows_4")
                if os.path.isdir(depth_path):
                    if self.views == 1:
                        depth_paths = sorted(
                            [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith("_0.npy")]
                        )
                        self.depth_flow_paths.append(depth_paths)
                    elif self.views == 3:
                        views_depth_lists = []
                        for v in range(3):
                            suffix = f"_{v}.npy"
                            files_v = sorted(
                                [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith(suffix)]
                            )
                            views_depth_lists.append(files_v)
                        self.depth_flow_paths.append(views_depth_lists)
                else:
                    self.depth_flow_paths.append([] if self.views == 1 else [[] for _ in range(3)])
                # semantic mask
                semantic_mask_path = os.path.join(image_dir, scene_name, "custom_masks")
                if os.path.isdir(semantic_mask_path):
                    if self.views == 1:
                        semantic_mask_paths = sorted(
                            [os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.semantic_mask_path.append(semantic_mask_paths)
                    elif self.views == 3:
                        views_sem_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(suffixes)])
                            views_sem_lists.append(files_v)
                        self.semantic_mask_path.append(views_sem_lists)
                else:
                    self.semantic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        sky_mask_paths = self.sky_mask_paths[idx]

        if self.mode == 6:
            frame_count = len(image_paths[0] if self.views == 3 else image_paths)
            required_frames = 6 if self.sequence_length == 2 else 16
            if frame_count < required_frames:
                raise ValueError(
                    f"mode=6 requires at least {required_frames} images in scene {self.scenes[idx]}, found {frame_count}"
                )
            max_start = frame_count - required_frames
            if self.start_idx > max_start:
                raise ValueError(
                    f"mode=6 start_idx {self.start_idx} exceeds the last valid start {max_start} "
                    f"for scene {self.scenes[idx]}"
                )
            start_idx = self.start_idx if self.start_idx >= 0 else random.randint(0, max_start)
            source_indices = [start_idx + 5 * i for i in range(self.sequence_length)]
            target_indices = list(range(start_idx, start_idx + required_frames))
            if self.views == 3:
                source_paths = [image_paths[v][i] for i in source_indices for v in range(3)]
                target_paths = [image_paths[v][i] for i in target_indices for v in range(3)]
                mask_paths = [path for paths in sky_mask_paths for path in paths]
            else:
                source_paths = [image_paths[i] for i in source_indices]
                target_paths = [image_paths[i] for i in target_indices]
                mask_paths = sky_mask_paths
            masks_by_stem = {os.path.splitext(os.path.basename(path))[0]: path for path in mask_paths}

            def matching_mask_paths(paths):
                matched = []
                for image_path in paths:
                    frame_stem = os.path.splitext(os.path.basename(image_path))[0]
                    mask_path = masks_by_stem.get(frame_stem)
                    if mask_path is None:
                        raise FileNotFoundError(
                            f"mode=6 requires sky mask {frame_stem}.png for scene {self.scenes[idx]}"
                        )
                    matched.append(mask_path)
                return matched

            source_mask_paths = matching_mask_paths(source_paths)
            target_mask_paths = matching_mask_paths(target_paths)
            images = load_and_preprocess_images(source_paths)
            targets = load_and_preprocess_images(target_paths)
            # Rendering masks intentionally reuse the Mode1/2 RGB+BICUBIC preprocessing path.
            masks = load_and_preprocess_images(source_mask_paths)
            # Loss/metric masks remain nearest-neighbor binary masks.
            target_masks = load_and_preprocess_masks(target_mask_paths)
            sample = {
                "images": images,
                "targets": targets,
                "masks": masks,
                "target_masks": target_masks,
                "image_paths": source_paths,
                "interval": [5],
            }
            dynamic_mask_paths = self.dynamic_mask_path[idx]
            if self.views == 3:
                available_dynamic_paths = [path for paths in dynamic_mask_paths for path in paths]
            else:
                available_dynamic_paths = dynamic_mask_paths
            dynamic_masks_by_stem = {
                os.path.splitext(os.path.basename(path))[0]: path
                for path in available_dynamic_paths
            }
            if all(os.path.splitext(os.path.basename(path))[0] in dynamic_masks_by_stem for path in source_paths):
                sample["source_dynamic_mask"] = load_and_preprocess_masks([
                    dynamic_masks_by_stem[os.path.splitext(os.path.basename(path))[0]]
                    for path in source_paths
                ])
            if self.metric_data:
                sample["metric_sky_mask"] = (target_masks > 0.5).float()
                dynamic_mask_paths = self.dynamic_mask_path[idx]
                target_dynamic_paths = None
                if self.views == 1 and len(dynamic_mask_paths) > target_indices[-1]:
                    target_dynamic_paths = [dynamic_mask_paths[i] for i in target_indices]
                elif self.views == 3 and all(len(dynamic_mask_paths[v]) > target_indices[-1] for v in range(3)):
                    target_dynamic_paths = [
                        dynamic_mask_paths[v][i] for i in target_indices for v in range(3)
                    ]
                if target_dynamic_paths is not None:
                    sample["metric_dynamic_mask"] = (
                        load_and_preprocess_masks(target_dynamic_paths) > 0.5
                    ).float()

                depth_paths = self.depth_flow_paths[idx]
                target_depth_paths = None
                if self.views == 1 and len(depth_paths) > target_indices[-1]:
                    target_depth_paths = [depth_paths[i] for i in target_indices]
                elif self.views == 3 and all(len(depth_paths[v]) > target_indices[-1] for v in range(3)):
                    target_depth_paths = [
                        depth_paths[v][i] for i in target_indices for v in range(3)
                    ]
                if target_depth_paths is not None:
                    sample["gt_depth"] = load_and_preprocess_flow(
                        target_depth_paths, None, None, images.shape[2], images.shape[3]
                    )
                else:
                    sample["gt_depth"] = torch.zeros(
                        len(target_paths), images.shape[2], images.shape[3]
                    )
            return sample

        dynamic_mask_paths = self.dynamic_mask_path[idx]
        semantic_mask_paths = self.semantic_mask_path[idx]

        if self.mode == 1:
            frame_count = len(image_paths[0] if self.views == 3 else image_paths)
            if self.continuous_frames:
                start_idx = random.randint(0, frame_count - self.sequence_length)
                indices = list(range(start_idx, start_idx + self.sequence_length))
                intervals = [1 for _ in range(self.sequence_length - 1)]
            else:
                start_idx = random.randint(0, max(1, frame_count - 21))
                intervals = sorted(random.sample(range(1, 20), self.sequence_length - 1))
                indices = [start_idx]
                for interval in intervals:
                    indices.append(start_idx + interval)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]

            timestamps = np.array(indices) - start_idx
            if timestamps[-1] > 0:
                timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            else:
                timestamps = np.zeros_like(timestamps, dtype=np.float32)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)

            input_dict = {
                "images": images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                "interval": intervals,
            }

        
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # legacy Mode1 behavior
                input_dict["dynamic_mask"] = dynamic_mask

            
            # if len(semantic_mask_paths) > 0:
            #     if self.views == 1:
            #         sem_mask_seq = [semantic_mask_paths[i] for i in indices]
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S, C, H, W]
            #     elif self.views == 3:
            #         sem_mask_seq = []
            #         for i in indices:
            #             for v in range(3):
            #                 sem_mask_seq.append(semantic_mask_paths[v][i])
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S*3, C, H, W]
            #     semantic_mask = semantic_mask * 255 / 10
            #     semantic_mask = semantic_mask.int()
            #     semantic_mask[semantic_mask > 9] = 255
            #     input_dict["semantic_mask"] = semantic_mask

            return input_dict

        elif self.mode == 2: 
            start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            
            timestamps = np.array(indices) - start_idx
            if timestamps[-1] > 0:
                timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            else:
                timestamps = np.zeros_like(timestamps, dtype=np.float32)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]
                


            metric_sky_mask = (load_and_preprocess_masks(mask_seq) > 0.5).float()
            input_dict = {
                "images": images,
                "image_paths": seq,
                "masks": masks,
                "metric_sky_mask": metric_sky_mask,
                "timestamps": timestamps,
                "interval": intervals,
            }
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

                input_dict["metric_dynamic_mask"] = (load_and_preprocess_masks(dy_mask_seq) > 0.5).float()
            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                else:
                    depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = depth_data

            return input_dict

        else:  # self.mode in (3, 4, 5)
            if self.mode == 5:
                max_start = min(len(image_paths), len(self.depth_flow_paths[idx])) - (self.sequence_length - 1) * self.interval - 1
                start_idx = self.start_idx if self.start_idx >= 0 else random.randint(0, max_start)
            else:
                start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            if self.mode == 4:
                target_indices = list(range(20))
            elif self.mode == 5:
                target_indices = list(range(start_idx, start_idx + (self.sequence_length - 1) * self.interval + 1))
            else:
                target_indices = [start_idx + i for i in range(self.sequence_length * self.interval - (self.interval - 1))]

            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
            
            # images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
                target_seq = [image_paths[i] for i in target_indices]
                target_images = load_and_preprocess_images(target_seq)  # [T, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

                target_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_seq.append(image_paths[v][i])
                target_images = load_and_preprocess_images(target_seq)  # [T*3, C, H, W]

            # sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
                target_mask_seq = [sky_mask_paths[i] for i in target_indices]
                target_masks = load_and_preprocess_images(target_mask_seq)  # [T, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]

                target_mask_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_mask_seq.append(sky_mask_paths[v][i])
                target_masks = load_and_preprocess_images(target_mask_seq)  # [T*3, C, H, W]

            input_dict = {
                "images": images,
                "targets": target_images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                # "target_timestamps": target_timestamps,
                "interval": intervals,
                "target_masks": target_masks,
            }
            if self.metric_data:
                input_dict["metric_sky_mask"] = (
                    load_and_preprocess_masks(target_mask_seq) > 0.5
                ).float()

            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                    target_dy_mask_seq = [dynamic_mask_paths[i] for i in target_indices]
                    target_dynamic_mask = load_and_preprocess_images(target_dy_mask_seq)  # [T, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    target_dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    for i in target_indices:
                        for v in range(3):
                            target_dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)         # [S*3, C, H, W]
                    target_dynamic_mask = load_and_preprocess_images(target_dy_mask_seq)  # [T*3, C, H, W]
                input_dict["dynamic_mask"] = target_dynamic_mask
                if self.metric_data:
                    input_dict["metric_dynamic_mask"] = (
                        load_and_preprocess_masks(target_dy_mask_seq) > 0.5
                    ).float()
                if self.mode == 5:
                    input_dict["input_dynamic_mask"] = dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_seq = [self.depth_flow_paths[idx][i] for i in target_indices]
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        target_depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        for i in target_indices:
                            for v in range(3):
                                target_depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                else:
                    target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = target_depth_data
            return input_dict
