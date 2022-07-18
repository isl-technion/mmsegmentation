_base_ = [
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/datasets/cityscapes_1024x1024.py',
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/airsim/repos/open-mmlab/mmsegmentation/configs/_base_/schedules/schedule_160k.py'
]

num_classes=15
AgamimPathA_hist = [383468, 227731094, 35444928, 18011157, 1178969, 4245155, 125601125, 13837139, 11735673, 5778866, 709932588, 234808574, 58020097, 469504390, 93447, 31506,]
AgamimPathB_hist = [117317, 164123392, 57421339, 20413641, 105651, 5048636, 328455851, 21808472, 96998734, 1634637, 781109580, 473067269, 55777244, 409130156, 60616, 112041,]
AgamimPathC_hist = [97267, 199139211, 23985210, 21064893, 91455, 4123993, 103423366, 18196612, 309953597, 916901, 656928313, 559045295, 36786860, 201651098, 404646, 110554,]
# class_weight =  np.array([1/np.sqrt(AgamimPathA_hist[i]+AgamimPathB_hist[i]+AgamimPathC_hist[i]) for i in range(num_classes)])
# class_weight /= class_weight.mean()
class_weight = [4.0789308 , 0.12975509, 0.29180902, 0.4089733 , 2.6890245 ,
       0.86114327, 0.13359833, 0.42988701, 0.15415959, 1.09290592,
       0.06806152, 0.08862189, 0.25705503, 0.09597243, 4.22010229]
class_weight = [1.0 for i in class_weight]
crop_size = (1024, 1024)  # (5472, 3648)  # (1440, 1088)
# stride_size = (768, 768)

model = dict(
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/mit_b0.pth')),
    backbone=dict(embed_dims=64, num_layers=[3, 4, 6, 3]),
    decode_head=dict(num_classes=num_classes,
                     # ignore_index=1,
                     in_channels=[64, 128, 320, 512],
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),  # , avg_non_ignore=True),
                     ),
    # test_cfg=dict(mode='whole', crop_size=crop_size))
    test_cfg=dict(mode='slide', crop_size=(1024, 2048), stride=(768, 1536)))


# dataset settings
dataset_type = 'AltaDataset'
data_root = '/media/isl12/Alta/'
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
pathC_scenarios_img = [
    'V7_Exp_25_1_21/Agamim/Path/C/30',
    'V7_Exp_25_1_21/Agamim/Path/C/50',
    'V7_Exp_25_1_21/Agamim/Path/C/70',
    'V7_Exp_25_1_21/Agamim/Path/C/100',
]
IrYamim_scenarios_img = [
    'V7_Exp_25_1_21/Ir yamim/30',
    'V7_Exp_25_1_21/Ir yamim/50',
    'V7_Exp_25_1_21/Ir yamim/70',
    'V7_Exp_25_1_21/Ir yamim/100',
]
PilotPath_img = [
    'V7_Exp_25_1_21/Pilot/Path',
]
pathA_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in pathA_scenarios_img]
pathB_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in pathB_scenarios_img]
pathC_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in pathC_scenarios_img]
IrYamim_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in IrYamim_scenarios_img]
PilotPath_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in PilotPath_img]

data = dict(
    samples_per_gpu=3,  ###
    workers_per_gpu=3,  ###
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
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)

load_from = '/home/airsim/repos/open-mmlab/mmsegmentation/pretrain/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'