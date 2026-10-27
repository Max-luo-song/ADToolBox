"""RL fine-tuning on rare collision rollouts."""

# Training uses the supplied HydraMDP rare-case splits via data.train.finetune_yaml.
# Generate new splits when changing the base model/data; see README.md.

import os

SIMENGINE_ROOT = os.getenv("SIMENGINE_ROOT", os.path.abspath("."))

_base_ = ["./e2e_hydramdp_50pct_rlft_common_log.py"]

train_dataset_type = 'NavSimOpenSceneE2EFineTuneSynthetic'

synthetic_folder_names = [
    'e2e_hydramdp_50pct_collision_navtrain_hydramdp_50pct_collision_NR_260723'
]

data = dict(train=dict(
    _delete_=True,
    type='NavSimOpenSceneE2EFineTuneSynthetic',
    file_client_args=dict(backend='disk'),
    data_root=os.path.join(SIMENGINE_ROOT, 'data/raw/openscene-v1.1/'),
    ann_file=os.path.join(
        SIMENGINE_ROOT,
        'data/policy_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl'
    ),
    nav_filter_path='configs/navsim_splits/navtrain_split/navtrain_50pct.yaml',
    customized_filter='v1',
    folder_name=synthetic_folder_names,
    pipeline=[
        dict(type='LoadMultiViewImageFromFilesWithDownsample',
             to_float32=True,
             img_root=os.path.join(
                 SIMENGINE_ROOT,
                 'data/raw/openscene-v1.1/sensor_blobs/trainval'),
             downsample_factor=2),
        dict(type='PhotoMetricDistortionMultiViewImage'),
        dict(type='NormalizeMultiviewImage',
             mean=[103.53, 116.28, 123.675],
             std=[1.0, 1.0, 1.0],
             to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(type='DefaultFormatBundle3D',
             class_names=[
                 'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
                 'czone_sign', 'generic_object'
             ]),
        dict(type='CustomCollect3D',
             keys=[
                 'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'sdc_planning',
                 'sdc_planning_mask', 'command', 'sdc_planning_world',
                 'sdc_planning_past', 'sdc_planning_mask_past',
                 'gt_pre_command_sdc', 'sdc_status', 'no_at_fault_collisions',
                 'drivable_area_compliance', 'ego_progress',
                 'time_to_collision_within_bound', 'comfort', 'score',
                 'fail_mask'
             ])
    ],
    classes=[
        'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
        'czone_sign', 'generic_object'
    ],
    modality=dict(use_lidar=False,
                  use_camera=True,
                  use_radar=False,
                  use_map=False,
                  use_external=True),
    test_mode=False,
    use_valid_flag=True,
    patch_size=[102.4, 102.4],
    canvas_size=(200, 200),
    bev_size=(200, 200),
    queue_length=4,
    past_steps=3,
    fut_steps=4,
    planning_steps=8,
    load_interval=1,
    box_type_3d='LiDAR',
    fix_can_bus_rotation=True,
    finetune_yaml=[
        'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_collision.yaml',
        'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_ep_1pct.yaml',
        'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_off_road.yaml'
    ]))
