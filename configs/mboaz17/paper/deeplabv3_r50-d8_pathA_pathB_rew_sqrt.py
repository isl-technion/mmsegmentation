_base_ = [
    '../../_base_/models/deeplabv3_r50-d8.py',
    # '../../_base_/datasets/pascal_voc12_aug.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]

num_classes=15
AgamimPathA_hist = [   268814, 229262856,  35228303,  20217782,   1150188,   3838590, 142051812,   15539169,  13295549,   3026753, 711029194, 215685325,  57806647, 467720714,      86985]
# class_weight =  np.array([1/np.sqrt(AgamimPathA_hist[i]) for i in range(num_classes)])
# class_weight /= class_weight.mean()
class_weight = [3.32959074, 0.11401184, 0.29085125, 0.38392823, 1.60965343,
       0.88111163, 0.14484163, 0.43792819, 0.47343859, 0.99226645,
       0.06474006, 0.11754563, 0.22705335, 0.07982216, 5.85321683]
crop_size = (1024, 1024)  # (5472, 3648)  # (1440, 1088)
stride_size = (768, 768)

model = dict(
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/mit_b0.pth')),
    decode_head=dict(num_classes=num_classes,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),
                     ),
    auxiliary_head=dict(num_classes=num_classes,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=class_weight)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride_size)
)


# dataset settings
dataset_type = 'AltaDataset'
data_root = '/media/isl12/Alta/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='Resize', img_scale=crop_size, keep_ratio=False),  # ratio_range=(0.25, 0.3)), img_scale=(2048, 1024)
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
    'V7_Exp_25_1_21/Agamim/Path/A/30',
    'V7_Exp_25_1_21/Agamim/Path/A/50',
    'V7_Exp_25_1_21/Agamim/Path/A/70',
    'V7_Exp_25_1_21/Agamim/Path/A/100',
]
pathB_scenarios_img = [
    'V7_Exp_25_1_21/Agamim/Path/B/30',
    'V7_Exp_25_1_21/Agamim/Path/B/50',
    'V7_Exp_25_1_21/Agamim/Path/B/70',
    'V7_Exp_25_1_21/Agamim/Path/B/100',
]
pathA_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in pathA_scenarios_img]
pathB_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in pathB_scenarios_img]

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
        img_dir=pathB_scenarios_img,
        ann_dir=pathB_scenarios_ann,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=pathB_scenarios_img,
        ann_dir=pathB_scenarios_ann,
        pipeline=test_pipeline))


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)

load_from = '/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth'