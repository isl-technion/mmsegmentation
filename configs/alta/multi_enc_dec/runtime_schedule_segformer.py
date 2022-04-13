# yapf:disable
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=160000)
# checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
