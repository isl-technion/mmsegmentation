project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'
data_root = '/media/isl12/Alta/MESSI dataset/'

_base_ = [
    '../mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
]
num_classes=15  # 16
class_weight = [0., 4.86139222, 0.12775909, 0.29381101, 0.38981798, 2.55928649,
                0.87541455, 0.16339358, 0.28703442, 0.1318935, 1.01571681, 0.09114451,
                0.11215303, 0.33036596, 0.08321761, 3.67759923]
class_weight = [1.0 for i in class_weight]
class_weight[0] = 0

crop_size = (1024, 1024)  # (5472, 3648)  # (1440, 1088)
# stride_size = (768, 768)
ignore_index=2

################### mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py ###########################
# _base_ = ['./mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py']
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),

    decode_head=dict(num_classes=num_classes,
                     ignore_index=ignore_index,
                     in_channels=[128, 256, 512, 1024],
                     loss_cls=dict(
                         type='mmdet.CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=2.0,
                         reduction='mean',
                         class_weight=class_weight,
                         avg_non_ignore=True,
                     ),
                     # loss_decode=dict(
                     #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),
                     # , avg_non_ignore=True),
                     ),
    test_cfg=dict(mode='whole'))
    # test_cfg=dict(mode='slide', crop_size=(1366, 2048), stride=(1141, 1712)))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))


#################### other MESSI-specific params ############################
# dataset settings
dataset_type = 'MessiDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='RandomResize', keep_ratio=True, ratio_range=(0.85, 1.15), scale=None),
    dict(
        type='RandomChoiceResize',
        scales=[int(3648 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=12000),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(4800, 4800), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]

pathA_30 = 'Train and Val/images/Agamim/Path/A/30'
pathA_50 = 'Train and Val/images/Agamim/Path/A/50'
pathA_70 = 'Train and Val/images/Agamim/Path/A/70'
pathA_100 = 'Train and Val/images/Agamim/Path/A/100'
pathA_30_ann = 'Train and Val/annotations/Agamim/Path/A/30'
pathA_50_ann = 'Train and Val/annotations/Agamim/Path/A/50'
pathA_70_ann = 'Train and Val/annotations/Agamim/Path/A/70'
pathA_100_ann = 'Train and Val/annotations/Agamim/Path/A/100'

pathB_30 = 'Train and Val/images/Agamim/Path/B/30'
pathB_50 = 'Train and Val/images/Agamim/Path/B/50'
pathB_70 = 'Train and Val/images/Agamim/Path/B/70'
pathB_100 = 'Train and Val/images/Agamim/Path/B/100'
pathB_30_ann = 'Train and Val/annotations/Agamim/Path/B/30'
pathB_50_ann = 'Train and Val/annotations/Agamim/Path/B/50'
pathB_70_ann = 'Train and Val/annotations/Agamim/Path/B/70'
pathB_100_ann = 'Train and Val/annotations/Agamim/Path/B/100'

pathC_30 = 'Train and Val/images/Agamim/Path/C/30'
pathC_50 = 'Train and Val/images/Agamim/Path/C/50'
pathC_70 = 'Train and Val/images/Agamim/Path/C/70'
pathC_100 = 'Train and Val/images/Agamim/Path/C/100'
pathC_30_ann = 'Train and Val/annotations/Agamim/Path/C/30'
pathC_50_ann = 'Train and Val/annotations/Agamim/Path/C/50'
pathC_70_ann = 'Train and Val/annotations/Agamim/Path/C/70'
pathC_100_ann = 'Train and Val/annotations/Agamim/Path/C/100'

Descend_0001 = 'Train and Val/images/Agamim/Descend/100_0001'
Descend_0002 = 'Train and Val/images/Agamim/Descend/100_0002'
Descend_0003 = 'Train and Val/images/Agamim/Descend/100_0003'
Descend_0004 = 'Train and Val/images/Agamim/Descend/100_0004'
Descend_0005 = 'Train and Val/images/Agamim/Descend/100_0005'
Descend_0006 = 'Train and Val/images/Agamim/Descend/100_0006'
Descend_0031 = 'Train and Val/images/Agamim/Descend/100_0031'
Descend_0035 = 'Train and Val/images/Agamim/Descend/100_0035'
Descend_0036 = 'Train and Val/images/Agamim/Descend/100_0036'
Descend_0037 = 'Train and Val/images/Agamim/Descend/100_0037'
Descend_0038 = 'Train and Val/images/Agamim/Descend/100_0038'
Descend_0040 = 'Train and Val/images/Agamim/Descend/100_0040'
Descend_0041 = 'Train and Val/images/Agamim/Descend/100_0041'
Descend_0042 = 'Train and Val/images/Agamim/Descend/100_0042'
Descend_0043 = 'Train and Val/images/Agamim/Descend/100_0043'

Descend_0001_ann = 'Train and Val/annotations/Agamim/Descend/100_0001'
Descend_0002_ann = 'Train and Val/annotations/Agamim/Descend/100_0002'
Descend_0003_ann = 'Train and Val/annotations/Agamim/Descend/100_0003'
Descend_0004_ann = 'Train and Val/annotations/Agamim/Descend/100_0004'
Descend_0005_ann = 'Train and Val/annotations/Agamim/Descend/100_0005'
Descend_0006_ann = 'Train and Val/annotations/Agamim/Descend/100_0006'
Descend_0031_ann = 'Train and Val/annotations/Agamim/Descend/100_0031'
Descend_0035_ann = 'Train and Val/annotations/Agamim/Descend/100_0035'
Descend_0036_ann = 'Train and Val/annotations/Agamim/Descend/100_0036'
Descend_0037_ann = 'Train and Val/annotations/Agamim/Descend/100_0037'
Descend_0038_ann = 'Train and Val/annotations/Agamim/Descend/100_0038'
Descend_0040_ann = 'Train and Val/annotations/Agamim/Descend/100_0040'
Descend_0041_ann = 'Train and Val/annotations/Agamim/Descend/100_0041'
Descend_0042_ann = 'Train and Val/annotations/Agamim/Descend/100_0042'
Descend_0043_ann = 'Train and Val/annotations/Agamim/Descend/100_0043'

IrYamim_30 = 'Test/images/IrYamim/30'
IrYamim_50 = 'Test/images/IrYamim/50'
IrYamim_70 = 'Test/images/IrYamim/70'
IrYamim_100 = 'Test/images/IrYamim/100'
IrYamim_30_ann = 'Train and Val/annotations/IrYamim/30'
IrYamim_50_ann = 'Train and Val/annotations/IrYamim/50'
IrYamim_70_ann = 'Train and Val/annotations/IrYamim/70'
IrYamim_100_ann = 'Train and Val/annotations/IrYamim/100'

PilotPath = 'Test/images/Ha-Medinah Square/Path'
PilotPath_ann = 'Train and Val/annotations/Ha-Medinah Square/Path'

dataset_train = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path=pathA_30, seg_map_path=pathA_30_ann),
    reduce_zero_label=False,
    ignore_index=ignore_index,
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
    ignore_index=ignore_index,
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
    batch_size=1, num_workers=4,
    dataset=dict(
        type='ConcatDataset',
        datasets=[IrYamim_30_test, IrYamim_50_test, IrYamim_70_test, IrYamim_100_test, PilotPath_test])
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# training schedule for 320k
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=320, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
