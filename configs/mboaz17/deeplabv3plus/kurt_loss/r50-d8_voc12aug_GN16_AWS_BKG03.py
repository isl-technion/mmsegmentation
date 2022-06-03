_base_ = [
    # '../../../_base_/models/deeplabv3plus_r50-d8.py',
    # '../../../_base_/datasets/pascal_voc12.py',
    '../../../_base_/default_runtime.py',
    # '../../../_base_/schedules/schedule_80k.py'
]

num_classes = 21
class_weight = [1]*num_classes
class_weight[0] = 0.3  # Background
features_num = 512
conv_cfg = dict(type='ConvAWS')
norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
model = dict(
    type='EncoderDecoderEnhanced',  # 'EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        conv_cfg=conv_cfg,  # None
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHeadHistLoss',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=num_classes,  #19,
        conv_cfg=conv_cfg,  # None
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),
        loss_hist=dict(
                type='HistogramLoss', loss_weight=0.0, features_num=features_num, directions_num=8000)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,  # 19,
        conv_cfg=conv_cfg,  # None
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=class_weight)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/home/airsim/repos/open-mmlab/mmsegmentation/data/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        # ann_dir='SegmentationClass',
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ],
        # split='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

# runtime settings
log_config = dict(
    interval=100, #366,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])

hist_save_interval = 10
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=hist_save_interval)
evaluation = dict(interval=hist_save_interval, metric='mIoU', pre_eval=True)
workflow = [('train', int(hist_save_interval)), ('val', int(1))]

custom_hooks = [
    dict(type='HistLossHook', num_classes=num_classes, first_epoch=1e6, layers_num_encoder=0, layers_num_decoder=5,
         features_num=features_num, layer_validity=[0, 0, 0, 0, 1], save_interval=1e6)
]
custom_imports = dict(imports=['tools.alta.histloss_hook'], allow_failed_imports=False)

load_from = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_voc12aug/iter_80000.pth'