# _base_ = './bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes.py'
###########################################################################3
_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/cityscapes.py', #    # '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py' # #, '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(type='ResNet', depth=50, init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))),
    decode_head=dict(
        type='FCNHead', in_channels=1024, in_index=0, channels=1024),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=34,  ###
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False),
    ])
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)

#########################'../_base_/datasets/cityscapes_1024x1024.py'###################################

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
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
        img_scale=(2048, 1024),
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
# dataset settings
dataset_type = 'RELLIS3DDataset'
data_root = '/home/airsim/repos/open-mmlab/mmsegmentation/data/Rellis-3d/'  ###
data = dict(
    samples_per_gpu=2,  ###
    workers_per_gpu=2,  ###
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',  ###
        ann_dir='ann_dir/train',  ###
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',  ###
        ann_dir='ann_dir/train',  ###
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',  ###
        ann_dir='ann_dir/train',  ###
        pipeline=test_pipeline))


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=800, metric='mIoU', pre_eval=True)