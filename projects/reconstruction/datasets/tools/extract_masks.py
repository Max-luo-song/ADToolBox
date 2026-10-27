"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir
    
    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.
"""

from mmseg.apis import inference_segmentor, init_segmentor
import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

dataset_classes_in_sematic = {
    'Road': [0],
    'Building': [2],
    'Vegetation': [8],
    'Vehicle': [13, 14, 15],
    'Person': [11],
    'Cyclist': [12, 17, 18],
    'Traffic Sign': [9],
    'Sidewalk': [1],
    'Sky': [10],
    'Other': []
}



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument("--scene_ids", default=None, type=int, nargs="+")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_scenes", type=int, default=200)
    parser.add_argument('--process_dynamic_mask', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="fine_dynamic_masks")
    parser.add_argument('--segformer_path', type=str, default='/home/guojianfei/ai_ws/SegFormer')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--palette', default='cityscapes')

    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')

    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        lines = open(args.split_file, "r").readlines()[1:]
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in lines]
        else:
            scene_ids_list = [int(line.strip().split(",")[0]) for line in lines]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    for scene_id in tqdm(scene_ids_list, desc="Processing Scenes"):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)
        flist = sorted(glob(os.path.join(img_dir, '*')))

        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        os.makedirs(sky_mask_dir, exist_ok=True)

        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")
            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            os.makedirs(all_mask_dir, exist_ok=True)
            os.makedirs(human_mask_dir, exist_ok=True)
            os.makedirs(vehicle_mask_dir, exist_ok=True)

        custom_mask_dir = os.path.join(args.data_root, scene_id, "custom_masks")
        os.makedirs(custom_mask_dir, exist_ok=True)

        for fpath in tqdm(flist, desc=f'scene[{scene_id}]', leave=False):
            fbase = os.path.splitext(os.path.basename(fpath))[0]
            mask_fpath = os.path.join(custom_mask_dir, f"{fbase}.png")
            if args.ignore_existing and os.path.exists(mask_fpath):
                continue

            result = inference_segmentor(model, fpath)
            mask = result[0].astype(np.uint8)

            sky_mask = np.isin(mask, [10])
            imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

            class_value_map = {
                'Road': 10,
                'Building': 20,
                'Vegetation': 30,
                'Vehicle': 40,
                'Person': 50,
                'Cyclist': 60,
                'Traffic Sign': 70,
                'Sidewalk': 80,
                'Sky': 90,
                'Other': 255
            }
            custom_mask = np.full_like(mask, class_value_map['Other'], dtype=np.uint8)
            for class_name, ids in dataset_classes_in_sematic.items():
                if class_name == 'Other': continue
                for city_id in ids:
                    custom_mask[mask == city_id] = class_value_map[class_name]
            imageio.imwrite(mask_fpath, custom_mask)

            if args.process_dynamic_mask:
                rough_human_mask_path = os.path.join(rough_human_mask_dir, f"{fbase}.png")
                rough_vehicle_mask_path = os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")

                if os.path.exists(rough_human_mask_path):
                    rough_human_mask = imageio.imread(rough_human_mask_path) > 0
                    human_mask = np.isin(mask, dataset_classes_in_sematic['Person'] + dataset_classes_in_sematic['Cyclist'])
                    valid_human_mask = np.logical_and(human_mask, rough_human_mask)
                    imageio.imwrite(os.path.join(human_mask_dir, f"{fbase}.png"), valid_human_mask.astype(np.uint8) * 255)
                else:
                    valid_human_mask = np.zeros_like(mask, dtype=bool)

                if os.path.exists(rough_vehicle_mask_path):
                    rough_vehicle_mask = imageio.imread(rough_vehicle_mask_path) > 0
                    vehicle_mask = np.isin(mask, dataset_classes_in_sematic['Vehicle'])
                    valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                    imageio.imwrite(os.path.join(vehicle_mask_dir, f"{fbase}.png"), valid_vehicle_mask.astype(np.uint8) * 255)
                else:
                    valid_vehicle_mask = np.zeros_like(mask, dtype=bool)

                valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                imageio.imwrite(os.path.join(all_mask_dir, f"{fbase}.png"), valid_all_mask.astype(np.uint8) * 255)