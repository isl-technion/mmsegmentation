_base_ = [
    '../../_base_/models/deeplabv3_r50-d8.py',
    # '../../_base_/datasets/pascal_voc12_aug.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]

num_classes=16
class_weight = [0, 0, 0.1, 0, 5, 0, 0, 1, 0, 10, 0, 0.1, 1, 1, 1, 0]
crop_size = (1440, 1088)  # (5472, 3648)  # (1440, 1088)

model = dict(
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/mit_b0.pth')),
    decode_head=dict(num_classes=num_classes,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),
                     ),
    auxiliary_head=dict(num_classes=num_classes,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=class_weight)),
)


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
    workers_per_gpu=2,  ###
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


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)

load_from = '/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth'