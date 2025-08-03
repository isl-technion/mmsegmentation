project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'
data_root = '/media/isl12/Alta/MESSI dataset/'

_base_ = [
    project_dir + 'configs/_base_/models/bisenetv1_r18-d32.py',
    project_dir + 'configs/_base_/datasets/cityscapes_1024x1024.py',
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

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(type='ResNet', depth=50)),
    decode_head=dict(
        type='FCNHead', in_channels=1024, in_index=0, channels=1024,
        num_classes=num_classes,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight, avg_non_ignore=True),
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=num_classes,
            ignore_index=ignore_index,
            in_index=1,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight, avg_non_ignore=True),
            concat_input=False),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=num_classes,
            ignore_index=ignore_index,
            in_index=2,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight, avg_non_ignore=True),
            concat_input=False),
    ],
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
        ignore_index=ignore_index,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=IrYamim_scenarios_img + PilotPath_img,
        ann_dir=IrYamim_scenarios_ann + PilotPath_ann,
        reduce_zero_label=False,
        ignore_index=ignore_index,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=IrYamim_scenarios_img + PilotPath_img,
        ann_dir=IrYamim_scenarios_ann + PilotPath_ann,
        reduce_zero_label=False,
        ignore_index=ignore_index,
        pipeline=test_pipeline))


# optimizer
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

load_from = project_dir + 'pretrain/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628-8b304447.pth'
