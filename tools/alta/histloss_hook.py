# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.runner.hooks import HOOKS, Hook
from torch import nn
import torch

from mmseg.models.losses.mboaz17.histogram_loss import HistogramLoss
import datetime;
import os


@HOOKS.register_module()
class HistLossHook(Hook):

    def __init__(self, num_classes=2, features_num=1):

        self.num_classes = num_classes
        self.features_num = features_num
        self.save_folder = ''  # will be determined later

        self.miu_all = torch.zeros((self.features_num, self.num_classes), device='cuda')
        self.moment2_all = torch.zeros((self.features_num, self.num_classes), device='cuda')
        self.samples_num_all = torch.zeros(self.num_classes, device='cuda')
        # self.active_classes_num = torch.tensor(0, device='cuda')
        self.var_all = torch.zeros((self.features_num, self.num_classes), device='cuda')
        self.iter = 0

    def before_val_epoch(self, runner):
        self.save_folder = os.path.join(runner.work_dir, 'hooks')
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        self.miu_all[:] = 0
        self.moment2_all[:] = 0
        self.samples_num_all[:] = 0
        self.iter = 0

    def after_val_iter(self, runner):
        self.miu_all += runner.model.module.decode_head.loss_hist.miu_all
        self.moment2_all += runner.model.module.decode_head.loss_hist.moment2_all
        self.samples_num_all += runner.model.module.decode_head.loss_hist.samples_num_all
        self.iter += 1

    def after_val_epoch(self, runner):
        """Synchronizing norm."""

        self.miu_all /= (self.samples_num_all.unsqueeze(dim=0)+1e-12)
        self.moment2_all /= (self.samples_num_all.unsqueeze(dim=0)+1e-12)
        self.var_all = self.moment2_all - self.miu_all**2 + 1e-12

        torch.save(self, os.path.join(self.save_folder, 'epoch_{}'.format(runner.epoch)))
