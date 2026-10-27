"""Pure imitation fine-tuning on rare logs (reward shaping and RL disabled)."""

# Training uses the supplied HydraMDP rare-case splits via data.train.finetune_yaml.
# Generate new splits when changing the base model/data; see README.md.

import os

SIMENGINE_ROOT = os.getenv("SIMENGINE_ROOT", os.path.abspath("."))

_base_ = ["./e2e_hydramdp_50pct.py"]

load_from = os.path.join(
    SIMENGINE_ROOT,
    'data/policy_engine/ckpts/hydramdp/e2e_hydramdp_50pct_ep8.pth')
model = dict(
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,
    planning_head=dict(
        _delete_=True,
        type='TrajScoringHeadRL',
        reward_shaping=False,
        use_lora=True,
        trans_use_lora=True,
        rl_finetuning=False,
        importance_sampling=True,
        num_poses=40,
        d_ffn=1024,
        d_model=256,
        vocab_path=os.path.join(SIMENGINE_ROOT,
                                'data/policy_engine/test_8192_kmeans.npy'),
        nhead=8,
        nlayers=1,
        num_commands=4,
        transformer_decoder=dict(
            type='BEVOnlyMotionTransformerDecoder',
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            embed_dims=256,
            num_layers=3,
            transformerlayers=dict(type='MotionTransformerAttentionLayer',
                                   batch_first=True,
                                   use_lora=True,
                                   lora_rank=16,
                                   attn_cfgs=[
                                       dict(type='MotionDeformableAttention',
                                            num_steps=8,
                                            embed_dims=256,
                                            num_levels=1,
                                            num_heads=8,
                                            num_points=4,
                                            sample_index=-1,
                                            use_lora=True,
                                            lora_rank=16)
                                   ],
                                   feedforward_channels=512,
                                   ffn_dropout=0.1,
                                   operation_order=('cross_attn', 'norm',
                                                    'ffn', 'norm'))),
        bev_h=200,
        bev_w=200),
    lora_finetuning=True)
train_pipeline = [
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
             'time_to_collision_within_bound', 'comfort', 'score', 'fail_mask'
         ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesWithDownsample',
         to_float32=True,
         img_root=os.path.join(SIMENGINE_ROOT,
                               'data/raw/openscene-v1.1/sensor_blobs/test'),
         downsample_factor=2),
    dict(type='NormalizeMultiviewImage',
         mean=[103.53, 116.28, 123.675],
         std=[1.0, 1.0, 1.0],
         to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1920, 1080),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D',
                  class_names=[
                      'vehicle', 'bicycle', 'pedestrian', 'traffic_cone',
                      'barrier', 'czone_sign', 'generic_object'
                  ],
                  with_label=False),
             dict(type='CustomCollect3D',
                  keys=[
                      'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'sdc_planning',
                      'sdc_planning_mask', 'command', 'sdc_planning_world',
                      'sdc_planning_past', 'sdc_planning_mask_past',
                      'gt_pre_command_sdc', 'sdc_status',
                      'no_at_fault_collisions', 'drivable_area_compliance',
                      'ego_progress', 'time_to_collision_within_bound',
                      'comfort', 'score'
                  ])
         ])
]
data = dict(train=dict(
    type='NavSimOpenSceneE2EFineTune',
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
    finetune_yaml=[
        'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_collision.yaml',
        'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_ep_1pct.yaml',
        'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_off_road.yaml'
    ]),
            val=dict(pipeline=[
                dict(type='LoadMultiViewImageFromFilesWithDownsample',
                     to_float32=True,
                     img_root=os.path.join(
                         SIMENGINE_ROOT,
                         'data/raw/openscene-v1.1/sensor_blobs/test'),
                     downsample_factor=2),
                dict(type='NormalizeMultiviewImage',
                     mean=[103.53, 116.28, 123.675],
                     std=[1.0, 1.0, 1.0],
                     to_rgb=False),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(type='MultiScaleFlipAug3D',
                     img_scale=(1920, 1080),
                     pts_scale_ratio=1,
                     flip=False,
                     transforms=[
                         dict(type='DefaultFormatBundle3D',
                              class_names=[
                                  'vehicle', 'bicycle', 'pedestrian',
                                  'traffic_cone', 'barrier', 'czone_sign',
                                  'generic_object'
                              ],
                              with_label=False),
                         dict(
                             type='CustomCollect3D',
                             keys=[
                                 'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                                 'sdc_planning', 'sdc_planning_mask',
                                 'command', 'sdc_planning_world',
                                 'sdc_planning_past', 'sdc_planning_mask_past',
                                 'gt_pre_command_sdc', 'sdc_status',
                                 'no_at_fault_collisions',
                                 'drivable_area_compliance', 'ego_progress',
                                 'time_to_collision_within_bound', 'comfort',
                                 'score'
                             ])
                     ])
            ]),
            test=dict(pipeline=[
                dict(type='LoadMultiViewImageFromFilesWithDownsample',
                     to_float32=True,
                     img_root=os.path.join(
                         SIMENGINE_ROOT,
                         'data/raw/openscene-v1.1/sensor_blobs/test'),
                     downsample_factor=2),
                dict(type='NormalizeMultiviewImage',
                     mean=[103.53, 116.28, 123.675],
                     std=[1.0, 1.0, 1.0],
                     to_rgb=False),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(type='MultiScaleFlipAug3D',
                     img_scale=(1920, 1080),
                     pts_scale_ratio=1,
                     flip=False,
                     transforms=[
                         dict(type='DefaultFormatBundle3D',
                              class_names=[
                                  'vehicle', 'bicycle', 'pedestrian',
                                  'traffic_cone', 'barrier', 'czone_sign',
                                  'generic_object'
                              ],
                              with_label=False),
                         dict(
                             type='CustomCollect3D',
                             keys=[
                                 'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                                 'sdc_planning', 'sdc_planning_mask',
                                 'command', 'sdc_planning_world',
                                 'sdc_planning_past', 'sdc_planning_mask_past',
                                 'gt_pre_command_sdc', 'sdc_status',
                                 'no_at_fault_collisions',
                                 'drivable_area_compliance', 'ego_progress',
                                 'time_to_collision_within_bound', 'comfort',
                                 'score'
                             ])
                     ])
            ]))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadMultiViewImageFromFilesWithDownsample',
             to_float32=True,
             img_root=os.path.join(
                 SIMENGINE_ROOT,
                 'data/raw/openscene-v1.1/sensor_blobs/test'),
             downsample_factor=2),
        dict(type='NormalizeMultiviewImage',
             mean=[103.53, 116.28, 123.675],
             std=[1.0, 1.0, 1.0],
             to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(type='MultiScaleFlipAug3D',
             img_scale=(1920, 1080),
             pts_scale_ratio=1,
             flip=False,
             transforms=[
                 dict(type='DefaultFormatBundle3D',
                      class_names=[
                          'vehicle', 'bicycle', 'pedestrian', 'traffic_cone',
                          'barrier', 'czone_sign', 'generic_object'
                      ],
                      with_label=False),
                 dict(type='CustomCollect3D',
                      keys=[
                          'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                          'sdc_planning', 'sdc_planning_mask', 'command',
                          'sdc_planning_world', 'sdc_planning_past',
                          'sdc_planning_mask_past', 'gt_pre_command_sdc',
                          'sdc_status', 'no_at_fault_collisions',
                          'drivable_area_compliance', 'ego_progress',
                          'time_to_collision_within_bound', 'comfort', 'score'
                      ])
             ])
    ])
train_dataset_type = 'NavSimOpenSceneE2EFineTune'
finetune_yaml = [
    'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_collision.yaml',
    'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_ep_1pct.yaml',
    'configs/navsim_splits/navtrain_split/e2e_hydramdp_50pct_rare/navtrain_50pct_off_road.yaml'
]
