# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.runner.hooks import HOOKS, Hook
import numpy as np
import torch
import pickle

from mmseg.models.losses.mboaz17.histogram_loss import HistogramLoss
import datetime;
import os


@HOOKS.register_module()
class HistLossHook(Hook):

    def __init__(self, num_classes=2, features_num=1):

        self.num_classes = num_classes
        self.features_num = features_num
        self.save_folder = ''  # will be determined later

        self.miu_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))  # For in-epoch calculations
        self.cov_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))  # For in-epoch calculations
        self.covinv_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))  # For in-epoch calculations
        self.samples_num_all = np.zeros(self.num_classes)
        self.loss_per_dim_all = np.zeros((self.features_num, self.num_classes))
        self.var_all = np.zeros((self.features_num, self.num_classes))
        self.iter = 0
        self.epsilon = 1e-12

    def before_val_epoch(self, runner):
        self.save_folder = os.path.join(runner.work_dir, 'hooks')
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        self.miu_all[:] = 0
        self.moment2_all[:] = 0
        self.moment2_mat_all[:] = 0
        self.samples_num_all[:] = 0
        self.iter = 0

    def after_val_iter(self, runner):
        self.miu_all += runner.model.module.decode_head.loss_hist.miu_all_curr_batch
        self.moment2_all += runner.model.module.decode_head.loss_hist.moment2_all_curr_batch
        self.moment2_mat_all += runner.model.module.decode_head.loss_hist.moment2_mat_all_curr_batch
        self.samples_num_all += runner.model.module.decode_head.loss_hist.samples_num_all_curr_batch
        self.iter += 1

    def after_val_epoch(self, runner):
        """Synchronizing norm."""

        self.miu_all /= (np.expand_dims(self.samples_num_all, axis=0)+self.epsilon)
        self.moment2_all /= (np.expand_dims(self.samples_num_all, axis=0)+self.epsilon)
        self.moment2_mat_all /= (np.expand_dims(self.samples_num_all, axis=(0,1))+self.epsilon)
        self.var_all = np.maximum(self.moment2_all - self.miu_all**2, self.epsilon)
        for c in range(0, self.num_classes):
            self.cov_mat_all[:, :, c] = self.moment2_mat_all[:, :, c] - \
                                        np.matmul(self.miu_all[:, c:c+1], self.miu_all[:, c:c+1].T) + \
                                        self.epsilon*np.eye(self.features_num)
            self.covinv_mat_all[:, :, c] = np.linalg.inv(self.cov_mat_all[:, :, c])

        # filename = os.path.join(self.save_folder, 'epoch_{}'.format(runner.epoch)+'.pickle')
        # with open(filename, 'wb') as handle:
        #     pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def before_train_epoch(self, runner):
        self.save_folder = os.path.join(runner.work_dir, 'hooks')
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # randomize projection matrix
        runner.model.module.decode_head.loss_hist.proj_mat = torch.randn_like(runner.model.module.decode_head.loss_hist.proj_mat)
        runner.model.module.decode_head.loss_hist.proj_mat /= torch.sum(runner.model.module.decode_head.loss_hist.proj_mat**2, dim=1).sqrt().unsqueeze(dim=1)

        runner.model.module.decode_head.loss_hist.hist_values = np.ones_like(runner.model.module.decode_head.loss_hist.hist_values) \
                                                                / runner.model.module.decode_head.loss_hist.bins_num
        runner.model.module.decode_head.loss_hist.samples_num_all_curr_epoch[:] = 0

    def after_train_epoch(self, runner):
        """Synchronizing norm."""

        self.miu_all = runner.model.module.decode_head.loss_hist.miu_all
        self.moment2_all = runner.model.module.decode_head.loss_hist.moment2_all
        self.moment2_mat_all = runner.model.module.decode_head.loss_hist.moment2_mat_all
        self.cov_mat_all = runner.model.module.decode_head.loss_hist.cov_mat_all
        self.loss_per_dim_all = runner.model.module.decode_head.loss_hist.loss_per_dim_all
        for c in range(0, self.num_classes):
            self.covinv_mat_all[:, :, c] = np.linalg.inv(self.cov_mat_all[:, :, c] + self.epsilon*np.eye(self.features_num))

        filename = os.path.join(self.save_folder, 'epoch_{}'.format(runner.epoch+1)+'.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
