_base_ = [
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/datasets/cityscapes_1024x1024.py',
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/schedules/schedule_160k.py'
]

num_classes=15
AgamimPathA_hist = [383468, 227730863, 35295341, 18005882, 1173707, 4244303, 125600525, 13837139, 11737163, 5778866, 714936537, 234801492, 58013721, 464674200, 93447, 31522,]
AgamimPathB_hist = [117317, 164123392, 57421339, 20429650, 105651, 5048636, 328455851, 21792463, 96998734, 1634637, 781109580, 473067269, 55777244, 409130156, 60616, 112041,]
AgamimPathC_hist = [97745, 199139015, 24103206, 21193658, 94631, 4143944, 109727685, 18196612, 308767480, 916901, 663199290, 565340325, 36926677, 203529072, 404646, 99561,]
# class_weight =  np.array([1/np.sqrt(AgamimPathA_hist[i]+AgamimPathB_hist[i]+AgamimPathC_hist[i]) for i in range(num_classes)])
# class_weight /= class_weight.mean()
class_weight = [4.07770567, 0.12976799, 0.29187739, 0.40853511, 2.69133159,
       0.8606163 , 0.13286251, 0.42999353, 0.15439344, 1.0930142 ,
       0.06789032, 0.08841154, 0.25696667, 0.09611336, 4.22052039]
crop_size = (1024, 1024)  # (5472, 3648)  # (1440, 1088)
# stride_size = (768, 768)

model = dict(
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/mit_b0.pth')),
    decode_head=dict(num_classes=num_classes,
                     # ignore_index=1,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),  # , avg_non_ignore=True),
                     ),
    # test_cfg=dict(mode='whole', crop_size=crop_size))
    test_cfg=dict(mode='slide', crop_size=(1366, 2048), stride=(1141, 1712)))


# dataset settings
dataset_type = 'MessiDataset'
data_root = '/media/isl12/Alta/MESSI dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
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
IrYamim_scenarios_img = [
    'Test/images/IrYamim/Path/30',
    'Test/images/IrYamim/Path/50',
    'Test/images/IrYamim/Path/70',
    'Test/images/IrYamim/Path/100',
]
PilotPath_img = [
    'Test/images/Ha-Medinah Square/Path',
]
pathA_scenarios_ann = [scn.replace('images', 'annotations') for scn in pathA_scenarios_img]
pathB_scenarios_ann = [scn.replace('images', 'annotations') for scn in pathB_scenarios_img]
pathC_scenarios_ann = [scn.replace('images', 'annotations') for scn in pathC_scenarios_img]
IrYamim_scenarios_ann = [scn.replace('images', 'annotations') for scn in IrYamim_scenarios_img]
PilotPath_ann = [scn.replace('images', 'annotations') for scn in PilotPath_img]

data = dict(
    samples_per_gpu=2,  ###
    workers_per_gpu=2,  ###
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=pathA_scenarios_img + pathB_scenarios_img + pathC_scenarios_img,
        ann_dir=pathA_scenarios_ann + pathB_scenarios_ann + pathC_scenarios_ann,
        reduce_zero_label=True,
        # ignore_index=1,  # Ignoring buildings (after reducing the labels by 1)
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=IrYamim_scenarios_img + PilotPath_img,
        ann_dir=IrYamim_scenarios_ann + PilotPath_ann,
        reduce_zero_label=True,
        # ignore_index=1,  # Ignoring buildings (after reducing the labels by 1)
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=IrYamim_scenarios_img + PilotPath_img,
        ann_dir=IrYamim_scenarios_ann + PilotPath_ann,
        reduce_zero_label=True,
        # ignore_index=1,  # Ignoring buildings (after reducing the labels by 1)
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


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=400)
checkpoint_config = dict(by_epoch=False, interval=400)
evaluation = dict(interval=400, metric='mIoU', pre_eval=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

load_from = '/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'