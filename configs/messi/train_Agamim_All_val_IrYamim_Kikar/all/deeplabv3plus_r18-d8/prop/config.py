project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'
data_root = '/media/isl12/Alta/MESSI dataset/'

_base_ = [
    project_dir + 'configs/_base_/models/deeplabv3plus_r50-d8.py',
    project_dir + 'configs/_base_/datasets/cityscapes.py',
    project_dir + 'configs/_base_/default_runtime.py',
    '../../../schedule_320_epochs.py'
]

num_classes=16
class_weight = [0.00000000e+00, 7.70245676e+00, 5.31975439e-03, 2.81348163e-02,
                4.95258197e-02, 2.13474376e+00, 2.49767237e-01, 8.70117583e-03,
                2.68519546e-02, 5.66963042e-03, 3.36242978e-01, 2.70750336e-03,
                4.09949139e-03, 3.55712021e-02, 2.25703558e-03, 4.40795088e+00]

crop_size = (1024, 1024)  # (5472, 3648)  # (1440, 1088)
# stride_size = (768, 768)
ignore_index=2

model = dict(
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/mit_b0.pth')),
    backbone=dict(depth=18),
    decode_head=dict(num_classes=num_classes,
                     # ignore_index=ignore_index,
                     c1_in_channels=64,
                     c1_channels=12,
                     in_channels=512,
                     channels=128,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),  # , avg_non_ignore=True),
                     ),
    auxiliary_head=dict(num_classes=num_classes,
                     # ignore_index=ignore_index,
                     in_channels=256, channels=64,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=class_weight),  # , avg_non_ignore=True),
                     ),
    # test_cfg=dict(mode='whole', crop_size=crop_size))
    test_cfg=dict(mode='slide', crop_size=(1366, 2048), stride=(1141, 1712)))


# dataset settings
dataset_type = 'MessiDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', keep_ratio=True, ratio_range=(0.85, 1.15)),  # img_scale=(2048, 1024)
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
        img_scale=None,
        img_ratios=1.0,  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),  # , img_scale=crop_size, keep_ratio=False),  ###  keep_ratio=True
            dict(type='RandomFlip'),  ###
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

pathA_scenarios_img = [
    'Train and Val/images/Agamim/Path/A/30',
    'Train and Val/images/Agamim/Path/A/50',
    'Train and Val/images/Agamim/Path/A/70',
    'Train and Val/images/Agamim/Path/A/100',
]
pathB_scenarios_img = [
    'Train and Val/images/Agamim/Path/B/30',
    'Train and Val/images/Agamim/Path/B/50',
    'Train and Val/images/Agamim/Path/B/70',
    'Train and Val/images/Agamim/Path/B/100',
]
pathC_scenarios_img = [
    'Train and Val/images/Agamim/Path/C/30',
    'Train and Val/images/Agamim/Path/C/50',
    'Train and Val/images/Agamim/Path/C/70',
    'Train and Val/images/Agamim/Path/C/100',
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
    'Test/images/IrYamim/30',
    'Test/images/IrYamim/50',
    'Test/images/IrYamim/70',
    'Test/images/IrYamim/100',
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
        img_dir=pathA_scenarios_img + pathB_scenarios_img + pathC_scenarios_img + Descend_scenarios_img,
        ann_dir=pathA_scenarios_ann + pathB_scenarios_ann + pathC_scenarios_ann + Descend_scenarios_ann,
        reduce_zero_label=False,
        # ignore_index=ignore_index,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=IrYamim_scenarios_img + PilotPath_img,
        ann_dir=IrYamim_scenarios_ann + PilotPath_ann,
        reduce_zero_label=False,
        # ignore_index=ignore_index,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=IrYamim_scenarios_img + PilotPath_img,
        ann_dir=IrYamim_scenarios_ann + PilotPath_ann,
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

load_from = project_dir + 'pretrain/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth'
