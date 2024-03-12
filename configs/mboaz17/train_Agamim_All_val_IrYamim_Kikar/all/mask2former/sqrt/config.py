# running_location = 'local'
# running_location = 'remote'
# if running_location == 'local':
project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'
data_root = '/media/isl12/Alta/'  # local
# elif running_location == 'remote':
# project_dir = '/home/barakp/Projects/open-mmlab/mmsegmentation/'
# data_root = '/home/barakp/Projects/open-mmlab/mmsegmentation/data/'  # remote

_base_ = [
    '../mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
    # '/home/airsim/repos/open-mmlab/mmsegmentation/' + 'configs/_base_/default_runtime.py',
]
# _base_ = [
#     '../mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
#     '/home/barakp/Projects/open-mmlab/mmsegmentation/' + 'configs/_base_/default_runtime.py',
# ]
num_classes=16
class_weight = [0., 4.86139222, 0.12775909, 0.29381101, 0.38981798, 2.55928649,
                0.87541455, 0.16339358, 0.28703442, 0.1318935, 1.01571681, 0.09114451,
                0.11215303, 0.33036596, 0.08321761, 3.67759923]

crop_size = (256, 256)  # (1024, 1024)  # (5472, 3648)  # (1440, 1088)
# stride_size = (768, 768)
ignore_index=2

############### mask2former_r50_8xb2-90k_cityscapes-512x1024.py #################
# # _base_ = ['../_base_/default_runtime.py', '../_base_/datasets/cityscapes.py']
#
# # crop_size = (512, 1024)
# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255,
#     size=crop_size,
#     test_cfg=dict(size_divisor=32))
# # num_classes = 19
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         deep_stem=False,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=-1,
#         norm_cfg=dict(type='SyncBN', requires_grad=False),
#         style='pytorch',
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
#     decode_head=dict(
#         type='Mask2FormerHead',
#         in_channels=[256, 512, 1024, 2048],
#         strides=[4, 8, 16, 32],
#         feat_channels=256,
#         out_channels=256,
#         num_classes=num_classes,
#         num_queries=100,
#         num_transformer_feat_level=3,
#         align_corners=False,
#         pixel_decoder=dict(
#             type='mmdet.MSDeformAttnPixelDecoder',
#             num_outs=3,
#             norm_cfg=dict(type='GN', num_groups=32),
#             act_cfg=dict(type='ReLU'),
#             encoder=dict(  # DeformableDetrTransformerEncoder
#                 num_layers=6,
#                 layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
#                     self_attn_cfg=dict(  # MultiScaleDeformableAttention
#                         embed_dims=256,
#                         num_heads=8,
#                         num_levels=3,
#                         num_points=4,
#                         im2col_step=64,
#                         dropout=0.0,
#                         batch_first=True,
#                         norm_cfg=None,
#                         init_cfg=None),
#                     ffn_cfg=dict(
#                         embed_dims=256,
#                         feedforward_channels=1024,
#                         num_fcs=2,
#                         ffn_drop=0.0,
#                         act_cfg=dict(type='ReLU', inplace=True))),
#                 init_cfg=None),
#             positional_encoding=dict(  # SinePositionalEncoding
#                 num_feats=128, normalize=True),
#             init_cfg=None),
#         enforce_decoder_input_project=False,
#         positional_encoding=dict(  # SinePositionalEncoding
#             num_feats=128, normalize=True),
#         transformer_decoder=dict(  # Mask2FormerTransformerDecoder
#             return_intermediate=True,
#             num_layers=9,
#             layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
#                 self_attn_cfg=dict(  # MultiheadAttention
#                     embed_dims=256,
#                     num_heads=8,
#                     attn_drop=0.0,
#                     proj_drop=0.0,
#                     dropout_layer=None,
#                     batch_first=True),
#                 cross_attn_cfg=dict(  # MultiheadAttention
#                     embed_dims=256,
#                     num_heads=8,
#                     attn_drop=0.0,
#                     proj_drop=0.0,
#                     dropout_layer=None,
#                     batch_first=True),
#                 ffn_cfg=dict(
#                     embed_dims=256,
#                     feedforward_channels=2048,
#                     num_fcs=2,
#                     act_cfg=dict(type='ReLU', inplace=True),
#                     ffn_drop=0.0,
#                     dropout_layer=None,
#                     add_identity=True)),
#             init_cfg=None),
#         loss_cls=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=False,
#             loss_weight=2.0,
#             reduction='mean',
#             class_weight=[1.0] * num_classes + [0.1]),
#         loss_mask=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='mean',
#             loss_weight=5.0),
#         loss_dice=dict(
#             type='mmdet.DiceLoss',
#             use_sigmoid=True,
#             activate=True,
#             reduction='mean',
#             naive_dice=True,
#             eps=1.0,
#             loss_weight=5.0),
#         train_cfg=dict(
#             num_points=12544,
#             oversample_ratio=3.0,
#             importance_sample_ratio=0.75,
#             assigner=dict(
#                 type='mmdet.HungarianAssigner',
#                 match_costs=[
#                     dict(type='mmdet.ClassificationCost', weight=2.0),
#                     dict(
#                         type='mmdet.CrossEntropyLossCost',
#                         weight=5.0,
#                         use_sigmoid=True),
#                     dict(
#                         type='mmdet.DiceCost',
#                         weight=5.0,
#                         pred_act=True,
#                         eps=1.0)
#                 ]),
#             sampler=dict(type='mmdet.MaskPseudoSampler'))),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))
#
# # dataset config
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(
#         type='RandomChoiceResize',
#         scales=[int(1024 * x * 0.1) for x in range(5, 21)],
#         resize_type='ResizeShortestEdge',
#         max_size=4096),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
#
# # optimizer
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer = dict(
#     type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=optimizer,
#     clip_grad=dict(max_norm=0.01, norm_type=2),
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#             'query_embed': embed_multi,
#             'query_feat': embed_multi,
#             'level_embed': embed_multi,
#         },
#         norm_decay_mult=0.0))
# # learning policy
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=0,
#         power=0.9,
#         begin=0,
#         end=90000,
#         by_epoch=False)
# ]
#
# # training schedule for 90k
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=90000, val_interval=5000)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(
#         type='CheckpointHook', by_epoch=False, interval=5000,
#         save_best='mIoU'),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='SegVisualizationHook'))
#
# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)


############### mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py #################
# # _base_ = ['./mask2former_r50_8xb2-90k_cityscapes-512x1024.py']
# pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
# depths = [2, 2, 6, 2]
# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=depths,
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         with_cp=False,
#         frozen_stages=-1,
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     decode_head=dict(in_channels=[96, 192, 384, 768]))
#
# # set all layers in backbone to lr_mult=0.1
# # set all norm layers, position_embeding,
# # query_embeding, level_embeding to decay_multi=0.0
# backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
# backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# custom_keys = {
#     'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#     'backbone.patch_embed.norm': backbone_norm_multi,
#     'backbone.norm': backbone_norm_multi,
#     'absolute_pos_embed': backbone_embed_multi,
#     'relative_position_bias_table': backbone_embed_multi,
#     'query_embed': embed_multi,
#     'query_feat': embed_multi,
#     'level_embed': embed_multi
# }
# custom_keys.update({
#     f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
#     for stage_id, num_blocks in enumerate(depths)
#     for block_id in range(num_blocks)
# })
# custom_keys.update({
#     f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
#     for stage_id in range(len(depths) - 1)
# })
# # optimizer
# optim_wrapper = dict(
#     paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))


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
                     # ignore_index=ignore_index,
                     in_channels=[128, 256, 512, 1024],
                     loss_cls=dict(
                         type='mmdet.CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=2.0,
                         reduction='mean',
                         class_weight=class_weight),
                     # loss_decode=dict(
                     #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight),
                     # , avg_non_ignore=True),
                     ),
    # test_cfg=dict(mode='whole', crop_size=crop_size))
    test_cfg=dict(mode='slide', crop_size=(1366, 2048), stride=(1141, 1712)))

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


#################### other ALTA-specific params ############################
# dataset settings
dataset_type = 'AltaDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    # dict(type='Normalize', **img_norm_cfg),
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

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# training schedule for 320k
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
