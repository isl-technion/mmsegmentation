project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'
data_root = '/media/isl12/Alta/MESSI dataset/'

_base_ = [
    project_dir + 'configs/_base_/models/segformer_mit-b0.py',
    project_dir + 'configs/_base_/datasets/cityscapes_1024x1024.py',
    project_dir + 'configs/_base_/default_runtime.py',
    '../../../schedule_160_epochs.py'
]

num_classes=16
class_weight = [0., 4.86139222, 0.12775909, 0.29381101, 0.38981798, 2.55928649,
                0.87541455, 0.16339358, 0.28703442, 0.1318935, 1.01571681, 0.09114451,
                0.11215303, 0.33036596, 0.08321761, 3.67759923]

crop_size = (512, 512)  # (5472, 3648)  # (1440, 1088)
resize_size = (1280, 720)
# resize_size = (1440, 1080)
# stride_size = (768, 768)
ignore_index=2

model = dict(
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/mit_b0.pth')),
    backbone=dict(embed_dims=64, num_layers=[3, 4, 18, 3]),
    decode_head=dict(num_classes=num_classes,
                     # ignore_index=ignore_index,
                     in_channels=[64, 128, 320, 512],
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),  # , avg_non_ignore=True),
                     ),
    test_cfg=dict(mode='whole', crop_size=resize_size))
    # test_cfg=dict(mode='slide', crop_size=(1366, 2048), stride=(1141, 1712)))


# dataset settings
dataset_type = 'MessiDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=resize_size, keep_ratio=False),  # ratio_range=(0.25, 0.3)), img_scale=(2048, 1024)
    # dict(type='Resize', keep_ratio=True, ratio_range=(0.85, 1.15)),  # img_scale=(2048, 1024)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  ###
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),  ###
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=resize_size,  # None
        img_ratios=1.0,  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=resize_size, keep_ratio=False),  ###  keep_ratio=True
            # dict(type='Resize', keep_ratio=True),  # , img_scale=crop_size, keep_ratio=False),  ###  keep_ratio=True
            dict(type='RandomFlip'),  ###
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

pathA_scenarios_img = [
    'Train and Val/images/Agamim/Path/A/30',
    'Train and Val/images/Agamim/Path/A/50',
    # 'Train and Val/images/Agamim/Path/A/70',
    # 'Train and Val/images/Agamim/Path/A/100',
]
pathB_scenarios_img = [
    'Train and Val/images/Agamim/Path/B/30',
    'Train and Val/images/Agamim/Path/B/50',
    # 'Train and Val/images/Agamim/Path/B/70',
    # 'Train and Val/images/Agamim/Path/B/100',
]
pathC_scenarios_img = [
    'Train and Val/images/Agamim/Path/C/30',
    'Train and Val/images/Agamim/Path/C/50',
    # 'Train and Val/images/Agamim/Path/C/70',
    # 'Train and Val/images/Agamim/Path/C/100',
]
Descend_scenarios_img = [
    # 'Train and Val/images/Agamim/Descend/100_0001',
    'Train and Val/images/Agamim/Descend/100_0002',
    'Train and Val/images/Agamim/Descend/100_0003',
    'Train and Val/images/Agamim/Descend/100_0004',
    'Train and Val/images/Agamim/Descend/100_0005',
    'Train and Val/images/Agamim/Descend/100_0006',
    'Train and Val/images/Agamim/Descend/100_0031',
    # 'Train and Val/images/Agamim/Descend/100_0035',
    # 'Train and Val/images/Agamim/Descend/100_0036',
    'Train and Val/images/Agamim/Descend/100_0037',
    'Train and Val/images/Agamim/Descend/100_0038',
    'Train and Val/images/Agamim/Descend/100_0040',
    'Train and Val/images/Agamim/Descend/100_0041',
    'Train and Val/images/Agamim/Descend/100_0042',
    'Train and Val/images/Agamim/Descend/100_0043',
]
IrYamim_scenarios_img = [
    'Test/images/IrYamim/Path/30',
    'Test/images/IrYamim/Path/50',
    # 'Test/images/IrYamim/Path/70',
    # 'Test/images/IrYamim/Path/100',
]
PilotPath_img = [
    'Test/images/Ha-Medinah Square/Path',
]
pathA_scenarios_ann = [scn.replace('images', 'annotations') for scn in pathA_scenarios_img]
pathB_scenarios_ann = [scn.replace('images', 'annotations') for scn in pathB_scenarios_img]
pathC_scenarios_ann = [scn.replace('images', 'annotations') for scn in pathC_scenarios_img]
Descend_scenarios_ann = [scn.replace('images', 'annotations') for scn in Descend_scenarios_img]
IrYamim_scenarios_ann = [scn.replace('images', 'annotations') for scn in IrYamim_scenarios_img]
PilotPath_ann = [scn.replace('images', 'annotations') for scn in PilotPath_img]

data = dict(
    samples_per_gpu=2,  ###
    workers_per_gpu=2,  ###
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=pathA_scenarios_img + pathB_scenarios_img + pathC_scenarios_img + IrYamim_scenarios_img,
        ann_dir=pathA_scenarios_ann + pathB_scenarios_ann + pathC_scenarios_ann + IrYamim_scenarios_ann,
        reduce_zero_label=False,
        # ignore_index=ignore_index,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=Descend_scenarios_img,
        ann_dir=Descend_scenarios_ann,
        reduce_zero_label=False,
        # ignore_index=ignore_index,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=Descend_scenarios_img,
        ann_dir=Descend_scenarios_ann,
        reduce_zero_label=False,
        # ignore_index=ignore_index,
        pipeline=test_pipeline))


# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

load_from = project_dir + 'pretrain/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth'
