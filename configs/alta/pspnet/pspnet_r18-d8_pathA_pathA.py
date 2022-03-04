_base_ = [
    '../../_base_/models/pspnet_r50-d8.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]

num_classes=16
class_weight = [0, 0, 0.1, 0, 5, 0, 0, 1, 0, 10, 0, 0.1, 1, 1, 1, 0]
crop_size = (1440, 1088)

model = dict(
    # pretrained='open-mmlab://resnet18_v1c',  ###
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=num_classes,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0) #, class_weight=class_weight),
    ),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=num_classes))


# dataset settings
dataset_type = 'AltaDataset'
data_root = '/media/isl12/Alta/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=crop_size, keep_ratio=False),  # ratio_range=(0.25, 0.3)), img_scale=(2048, 1024)
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
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=crop_size, keep_ratio=False),  ###  keep_ratio=True
            # dict(type='RandomFlip'),  ###
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

pathA_scenarios_img = [
    'V7_Exp_25_1_21/Agamim/Path/A/30',
    'V7_Exp_25_1_21/Agamim/Path/A/50',
    'V7_Exp_25_1_21/Agamim/Path/A/70',
    'V7_Exp_25_1_21/Agamim/Path/A/100',
]
pathA_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in pathA_scenarios_img]

data = dict(
    samples_per_gpu=2,  ###
    workers_per_gpu=1,  ###
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=pathA_scenarios_img,
        ann_dir=pathA_scenarios_ann,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=pathA_scenarios_img,
        ann_dir=pathA_scenarios_ann,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=pathA_scenarios_img,
        ann_dir=pathA_scenarios_ann,
        pipeline=test_pipeline))


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)  ###
checkpoint_config = dict(by_epoch=False, interval=1000)  ###
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)  ###
