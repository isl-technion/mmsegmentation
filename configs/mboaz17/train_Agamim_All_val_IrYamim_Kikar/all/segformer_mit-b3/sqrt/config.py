# running_location = 'local'
# running_location = 'remote'
# if running_location == 'local':
project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'
data_root = '/media/isl12/Alta/'  # local
# elif running_location == 'remote':
#     project_dir = '/home/boaz/Projects/open-mmlab/mmsegmentation/'
#     data_root = '/home/boaz/Projects/open-mmlab/mmsegmentation/data/'  # remote

_base_ = [
    '/home/airsim/repos/open-mmlab/mmsegmentation/' + 'configs/_base_/models/segformer_mit-b0.py',
    # '/home/airsim/repos/open-mmlab/mmsegmentation/' + 'configs/_base_/datasets/cityscapes_1024x1024.py',
    '/home/airsim/repos/open-mmlab/mmsegmentation/' + 'configs/_base_/default_runtime.py',
    '../../../schedule_320_epochs.py'
]
num_classes=16
class_weight = [0., 4.86139222, 0.12775909, 0.29381101, 0.38981798, 2.55928649,
                0.87541455, 0.16339358, 0.28703442, 0.1318935, 1.01571681, 0.09114451,
                0.11215303, 0.33036596, 0.08321761, 3.67759923]

crop_size = (1024, 1024)  # (5472, 3648)  # (1440, 1088)
# stride_size = (768, 768)
ignore_index=2

data_preprocessor = dict(size=crop_size)
# load_from = project_dir + 'pretrain/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint), embed_dims=64, num_layers=[3, 4, 18, 3]),
    decode_head=dict(num_classes=num_classes,
                     # ignore_index=ignore_index,
                     in_channels=[64, 128, 320, 512],
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),  # , avg_non_ignore=True),
                     ),
    # test_cfg=dict(mode='whole', crop_size=crop_size))
    test_cfg=dict(mode='slide', crop_size=(1366, 2048), stride=(1141, 1712)))


# dataset settings
dataset_type = 'AltaDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='RandomResize', keep_ratio=True, ratio_range=(0.85, 1.15), scale=None),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  ###
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0),  # , seg_pad_val=255),  ###
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]

pathA_30 = 'V7_Exp_25_1_21/Agamim/Path/A/30'
pathA_50 = 'V7_Exp_25_1_21/Agamim/Path/A/50'
pathA_70 = 'V7_Exp_25_1_21/Agamim/Path/A/70'
pathA_100 = 'V7_Exp_25_1_21/Agamim/Path/A/100'
pathA_30_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/A/30'
pathA_50_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/A/50'
pathA_70_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/A/70'
pathA_100_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/A/100'

pathB_30 = 'V7_Exp_25_1_21/Agamim/Path/B/30'
pathB_50 = 'V7_Exp_25_1_21/Agamim/Path/B/50'
pathB_70 = 'V7_Exp_25_1_21/Agamim/Path/B/70'
pathB_100 = 'V7_Exp_25_1_21/Agamim/Path/B/100'
pathB_30_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/B/30'
pathB_50_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/B/50'
pathB_70_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/B/70'
pathB_100_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/B/100'

pathC_30 = 'V7_Exp_25_1_21/Agamim/Path/C/30'
pathC_50 = 'V7_Exp_25_1_21/Agamim/Path/C/50'
pathC_70 = 'V7_Exp_25_1_21/Agamim/Path/C/70'
pathC_100 = 'V7_Exp_25_1_21/Agamim/Path/C/100'
pathC_30_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/C/30'
pathC_50_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/C/50'
pathC_70_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/C/70'
pathC_100_ann = 'V7_Exp_25_1_21_annot/Agamim/Path/C/100'

Descend_0001 = 'V7_Exp_25_1_21/Agamim/Descend/100_0001'
Descend_0002 = 'V7_Exp_25_1_21/Agamim/Descend/100_0002'
Descend_0003 = 'V7_Exp_25_1_21/Agamim/Descend/100_0003'
Descend_0004 = 'V7_Exp_25_1_21/Agamim/Descend/100_0004'
Descend_0005 = 'V7_Exp_25_1_21/Agamim/Descend/100_0005'
Descend_0006 = 'V7_Exp_25_1_21/Agamim/Descend/100_0006'
Descend_0031 = 'V7_Exp_25_1_21/Agamim/Descend/100_0031'
Descend_0035 = 'V7_Exp_25_1_21/Agamim/Descend/100_0035'
Descend_0036 = 'V7_Exp_25_1_21/Agamim/Descend/100_0036'
Descend_0037 = 'V7_Exp_25_1_21/Agamim/Descend/100_0037'
Descend_0038 = 'V7_Exp_25_1_21/Agamim/Descend/100_0038'
Descend_0040 = 'V7_Exp_25_1_21/Agamim/Descend/100_0040'
Descend_0041 = 'V7_Exp_25_1_21/Agamim/Descend/100_0041'
Descend_0042 = 'V7_Exp_25_1_21/Agamim/Descend/100_0042'
Descend_0043 = 'V7_Exp_25_1_21/Agamim/Descend/100_0043'

Descend_0001_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0001'
Descend_0002_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0002'
Descend_0003_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0003'
Descend_0004_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0004'
Descend_0005_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0005'
Descend_0006_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0006'
Descend_0031_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0031'
Descend_0035_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0035'
Descend_0036_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0036'
Descend_0037_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0037'
Descend_0038_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0038'
Descend_0040_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0040'
Descend_0041_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0041'
Descend_0042_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0042'
Descend_0043_ann = 'V7_Exp_25_1_21_annot/Agamim/Descend/100_0043'

IrYamim_30 = 'V7_Exp_25_1_21/Ir yamim/30'
IrYamim_50 = 'V7_Exp_25_1_21/Ir yamim/50'
IrYamim_70 = 'V7_Exp_25_1_21/Ir yamim/70'
IrYamim_100 = 'V7_Exp_25_1_21/Ir yamim/100'
IrYamim_30_ann = 'V7_Exp_25_1_21_annot/Ir yamim/30'
IrYamim_50_ann = 'V7_Exp_25_1_21_annot/Ir yamim/50'
IrYamim_70_ann = 'V7_Exp_25_1_21_annot/Ir yamim/70'
IrYamim_100_ann = 'V7_Exp_25_1_21_annot/Ir yamim/100'

PilotPath = 'V7_Exp_25_1_21/Pilot/Path'
PilotPath_ann = 'V7_Exp_25_1_21_annot/Pilot/Path'

dataset_train = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path=pathA_30, seg_map_path=pathA_30_ann),
    reduce_zero_label=False,
    # ignore_index=ignore_index,
    pipeline=train_pipeline)
dataset_A_30_train = dataset_train.copy()
dataset_A_30_train['data_prefix'] = dict(img_path=pathA_30, seg_map_path=pathA_30_ann)
dataset_A_50_train = dataset_train.copy()
dataset_A_50_train['data_prefix'] = dict(img_path=pathA_50, seg_map_path=pathA_50_ann)
dataset_A_70_train = dataset_train.copy()
dataset_A_70_train['data_prefix'] = dict(img_path=pathA_70, seg_map_path=pathA_70_ann)
dataset_A_100_train = dataset_train.copy()
dataset_A_100_train['data_prefix'] = dict(img_path=pathA_100, seg_map_path=pathA_100_ann)
dataset_B_30_train = dataset_train.copy()
dataset_B_30_train['data_prefix'] = dict(img_path=pathB_30, seg_map_path=pathB_30_ann)
dataset_B_50_train = dataset_train.copy()
dataset_B_50_train['data_prefix'] = dict(img_path=pathB_50, seg_map_path=pathB_50_ann)
dataset_B_70_train = dataset_train.copy()
dataset_B_70_train['data_prefix'] = dict(img_path=pathB_70, seg_map_path=pathB_70_ann)
dataset_B_100_train = dataset_train.copy()
dataset_B_100_train['data_prefix'] = dict(img_path=pathB_100, seg_map_path=pathB_100_ann)
dataset_C_30_train = dataset_train.copy()
dataset_C_30_train['data_prefix'] = dict(img_path=pathC_30, seg_map_path=pathC_30_ann)
dataset_C_50_train = dataset_train.copy()
dataset_C_50_train['data_prefix'] = dict(img_path=pathC_50, seg_map_path=pathC_50_ann)
dataset_C_70_train = dataset_train.copy()
dataset_C_70_train['data_prefix'] = dict(img_path=pathC_70, seg_map_path=pathC_70_ann)
dataset_C_100_train = dataset_train.copy()
dataset_C_100_train['data_prefix'] = dict(img_path=pathC_100, seg_map_path=pathC_100_ann)
Descend_0001_train = dataset_train.copy()
Descend_0001_train['data_prefix'] = dict(img_path=Descend_0001, seg_map_path=Descend_0001_ann)
Descend_0002_train = dataset_train.copy()
Descend_0002_train['data_prefix'] = dict(img_path=Descend_0002, seg_map_path=Descend_0002_ann)
Descend_0003_train = dataset_train.copy()
Descend_0003_train['data_prefix'] = dict(img_path=Descend_0003, seg_map_path=Descend_0003_ann)
Descend_0004_train = dataset_train.copy()
Descend_0004_train['data_prefix'] = dict(img_path=Descend_0004, seg_map_path=Descend_0004_ann)
Descend_0005_train = dataset_train.copy()
Descend_0005_train['data_prefix'] = dict(img_path=Descend_0005, seg_map_path=Descend_0005_ann)
Descend_0006_train = dataset_train.copy()
Descend_0006_train['data_prefix'] = dict(img_path=Descend_0006, seg_map_path=Descend_0006_ann)
Descend_0031_train = dataset_train.copy()
Descend_0031_train['data_prefix'] = dict(img_path=Descend_0031, seg_map_path=Descend_0031_ann)
Descend_0035_train = dataset_train.copy()
Descend_0035_train['data_prefix'] = dict(img_path=Descend_0035, seg_map_path=Descend_0035_ann)
Descend_0036_train = dataset_train.copy()
Descend_0036_train['data_prefix'] = dict(img_path=Descend_0036, seg_map_path=Descend_0036_ann)
Descend_0037_train = dataset_train.copy()
Descend_0037_train['data_prefix'] = dict(img_path=Descend_0037, seg_map_path=Descend_0037_ann)
Descend_0038_train = dataset_train.copy()
Descend_0038_train['data_prefix'] = dict(img_path=Descend_0038, seg_map_path=Descend_0038_ann)
Descend_0040_train = dataset_train.copy()
Descend_0040_train['data_prefix'] = dict(img_path=Descend_0040, seg_map_path=Descend_0040_ann)
Descend_0041_train = dataset_train.copy()
Descend_0041_train['data_prefix'] = dict(img_path=Descend_0041, seg_map_path=Descend_0041_ann)
Descend_0042_train = dataset_train.copy()
Descend_0042_train['data_prefix'] = dict(img_path=Descend_0042, seg_map_path=Descend_0042_ann)
Descend_0043_train = dataset_train.copy()
Descend_0043_train['data_prefix'] = dict(img_path=Descend_0043, seg_map_path=Descend_0043_ann)

dataset_test = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path=pathA_30, seg_map_path=pathA_30_ann),
    reduce_zero_label=False,
    # ignore_index=ignore_index,
    pipeline=test_pipeline)
IrYamim_30_test = dataset_test.copy()
IrYamim_30_test['data_prefix'] = dict(img_path=IrYamim_30, seg_map_path=IrYamim_30_ann)
IrYamim_50_test = dataset_test.copy()
IrYamim_50_test['data_prefix'] = dict(img_path=IrYamim_50, seg_map_path=IrYamim_50_ann)
IrYamim_70_test = dataset_test.copy()
IrYamim_70_test['data_prefix'] = dict(img_path=IrYamim_70, seg_map_path=IrYamim_70_ann)
IrYamim_100_test = dataset_test.copy()
IrYamim_100_test['data_prefix'] = dict(img_path=IrYamim_100, seg_map_path=IrYamim_100_ann)
PilotPath_test = dataset_test.copy()
PilotPath_test['data_prefix'] = dict(img_path=PilotPath, seg_map_path=PilotPath_ann)

# dataset_A_30_test = dataset_test.copy()
# dataset_A_30_test['data_prefix'] = dict(img_path=pathA_30, seg_map_path=pathA_30_ann)
# dataset_A_50_test = dataset_test.copy()
# dataset_A_50_test['data_prefix'] = dict(img_path=pathA_50, seg_map_path=pathA_50_ann)
# dataset_A_70_test = dataset_test.copy()
# dataset_A_70_test['data_prefix'] = dict(img_path=pathA_70, seg_map_path=pathA_70_ann)
# dataset_A_100_test = dataset_test.copy()
# dataset_A_100_test['data_prefix'] = dict(img_path=pathA_100, seg_map_path=pathA_100_ann)
# dataset_B_30_test = dataset_test.copy()
# dataset_B_30_test['data_prefix'] = dict(img_path=pathB_30, seg_map_path=pathB_30_ann)
# dataset_B_50_test = dataset_test.copy()
# dataset_B_50_test['data_prefix'] = dict(img_path=pathB_50, seg_map_path=pathB_50_ann)
# dataset_B_70_test = dataset_test.copy()
# dataset_B_70_test['data_prefix'] = dict(img_path=pathB_70, seg_map_path=pathB_70_ann)
# dataset_B_100_test = dataset_test.copy()
# dataset_B_100_test['data_prefix'] = dict(img_path=pathB_100, seg_map_path=pathB_100_ann)

train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type='ConcatDataset',
        datasets=[dataset_A_30_train, dataset_A_50_train, dataset_A_70_train, dataset_A_100_train,
                  dataset_B_30_train, dataset_B_50_train, dataset_B_70_train, dataset_B_100_train,
                  dataset_C_30_train, dataset_C_50_train, dataset_C_70_train, dataset_C_100_train,
                  Descend_0002_train, Descend_0003_train, Descend_0004_train, Descend_0005_train,
                  Descend_0006_train, Descend_0031_train, Descend_0037_train, Descend_0040_train,
                  Descend_0040_train, Descend_0041_train, Descend_0042_train, Descend_0043_train])
)

val_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type='ConcatDataset',
        datasets=[IrYamim_30_test, IrYamim_50_test, IrYamim_70_test, IrYamim_100_test, PilotPath_test])
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
