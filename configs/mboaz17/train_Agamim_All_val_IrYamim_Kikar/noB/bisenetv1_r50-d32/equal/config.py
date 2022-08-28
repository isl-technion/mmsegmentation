running_location = 'local'
# running_location = 'remote'
if running_location == 'local':
    project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'
    data_root = '/media/isl12/Alta/'  # local
elif running_location == 'remote':
    project_dir = '/home/boaz/Projects/open-mmlab/mmsegmentation/'
    data_root = '/home/boaz/Projects/open-mmlab/mmsegmentation/data/'  # remote

_base_ = [
    project_dir + 'configs/_base_/models/bisenetv1_r18-d32.py',
    project_dir + 'configs/_base_/datasets/cityscapes_1024x1024.py',
    project_dir + 'configs/_base_/default_runtime.py',
    '../../../schedule_320_epochs.py'
]

num_classes=16
AgamimPathA_hist = [383468, 227730863, 35295341, 18005882, 1173707, 4244303, 125600525, 13837139, 11737163, 5778866, 714936537, 234801492, 58013721, 464674200, 93447, 31522,]
AgamimPathB_hist = [117317, 164123392, 57421339, 20429650, 105651, 5048636, 328455851, 21792463, 96998734, 1634637, 781109580, 473067269, 55777244, 409130156, 60616, 112041,]
AgamimPathC_hist = [97745, 199139015, 24103206, 21193658, 94631, 4143944, 109727685, 18196612, 308767480, 916901, 663199290, 565340325, 36926677, 203529072, 404646, 99561,]
Descend_hist = [4422502, 4882135267, 491870426, 338537919, 7635196, 66208360, 2813913853, 401135195, 1855076148, 17919689, 9781365668, 4204606894, 658432951, 6402717589, 6376412, 6615531,]  # without 0001, 0035, pre-sampled 1:5
# class_weight = np.array([1/np.sqrt(AgamimPathA_hist[i]+AgamimPathB_hist[i]+AgamimPathC_hist[i]+Descend_hist[i]) for i in range(num_classes)])
# class_weight /= class_weight.mean()
class_weight = [0, 3.88063889, 0.11753898, 0.35245326, 0.43577974, 2.89705656,
       0.97436081, 0.14961989, 0.40767347, 0.18240645, 1.697205  ,
       0.0795768 , 0.11748868, 0.30569269, 0.10054198, 3.30196682]
class_weight = [1.0 for i in class_weight]
class_weight[0] = 0
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
dataset_type = 'AltaDataset'
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
Descend_scenarios_img = [
    # 'V7_Exp_25_1_21/Agamim/Descend/100_0001',
    'V7_Exp_25_1_21/Agamim/Descend/100_0002',
    'V7_Exp_25_1_21/Agamim/Descend/100_0003',
    'V7_Exp_25_1_21/Agamim/Descend/100_0004',
    'V7_Exp_25_1_21/Agamim/Descend/100_0005',
    'V7_Exp_25_1_21/Agamim/Descend/100_0006',
    'V7_Exp_25_1_21/Agamim/Descend/100_0031',
    # 'V7_Exp_25_1_21/Agamim/Descend/100_0035',
    'V7_Exp_25_1_21/Agamim/Descend/100_0036',
    'V7_Exp_25_1_21/Agamim/Descend/100_0037',
    'V7_Exp_25_1_21/Agamim/Descend/100_0038',
    'V7_Exp_25_1_21/Agamim/Descend/100_0040',
    'V7_Exp_25_1_21/Agamim/Descend/100_0041',
    'V7_Exp_25_1_21/Agamim/Descend/100_0042',
    'V7_Exp_25_1_21/Agamim/Descend/100_0043',
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
Descend_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in Descend_scenarios_img]
IrYamim_scenarios_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in IrYamim_scenarios_img]
PilotPath_ann = [scn.replace('V7_Exp_25_1_21', 'V7_Exp_25_1_21_annot') for scn in PilotPath_img]

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
        ann_dir=IrYamim_scenarios_img + PilotPath_ann,
        reduce_zero_label=False,
        ignore_index=ignore_index,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=IrYamim_scenarios_img + PilotPath_img,
        ann_dir=IrYamim_scenarios_img + PilotPath_ann,
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
