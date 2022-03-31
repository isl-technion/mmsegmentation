_base_ = [
    '../../_base_/models/segformer_mit-b0.py',
    '../../_base_/datasets/cityscapes_1024x1024.py',
    'runtime_schedule_segformer.py'
]

## model settings
num_classes=16
class_weight = [0, 0, 0.1, 0, 5, 0, 0, 1, 0, 10, 0, 0.1, 1, 1, 1, 0]
resize_size = (672, 448)  # (1440, 1088)
crop_size = resize_size[::-1]

model = dict(
    type='EncoderDecoderEnhanced',
    decode_head=dict(type='SegformerHeadHistLoss',
                     num_classes=num_classes,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),
                     loss_hist=dict(
                         type='HistogramLoss', loss_weight=0.0),
                     ),
    test_cfg=dict(mode='whole', crop_size=crop_size))


## dataset settings
dataset_type = 'AltaDataset'
data_root = '/media/isl12/Alta/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=resize_size, keep_ratio=False),  # ratio_range=(0.25, 0.3)), img_scale=(2048, 1024)
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
        img_scale=resize_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=resize_size, keep_ratio=False),  ###  keep_ratio=True
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
    samples_per_gpu=1,
    workers_per_gpu=1,
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

# learning policy
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=10000)
# checkpoint_config = dict(by_epoch=False, interval=500)
# evaluation = dict(interval=500, metric='mIoU', pre_eval=True)
# workflow = [('train', int(480)), ('val', int(96))]

runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=5, metric='mIoU', pre_eval=True)
workflow = [('train', int(5)), ('val', int(1))]

load_from = '/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'

custom_hooks = [
    dict(type='HistLossHook', num_classes=num_classes, features_num=256)
]
custom_imports = dict(imports=['tools.alta.histloss_hook'], allow_failed_imports=False)
