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
        self.eigen_vals = np.zeros((self.features_num, self.num_classes))
        self.eigen_vecs = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.covinv_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))  # For in-epoch calculations
        self.samples_num_all = np.zeros(self.num_classes)
        self.samples_num_all_curr_epoch = np.zeros(self.num_classes)
        self.loss_per_dim_all = np.zeros((self.features_num, self.num_classes))
        self.var_all = np.zeros((self.features_num, self.num_classes))
        self.epsilon = 1e-12


    def before_train_iter(self, runner):
        # Increase the loss weight as more batches are involved in the histogram estimation
        runner.model.module.decode_head.loss_hist.relative_weight = (runner.inner_iter+1) / runner.data_loader.sampler.num_samples

    def before_train_epoch(self, runner):

        # Start training only after 15 epochs
        runner.model.module.decode_head.loss_hist.loss_weight = runner.model.module.decode_head.loss_hist.loss_weight_orig * \
                                                                (runner.epoch>=15)

        self.save_folder = os.path.join(runner.work_dir, 'hooks')
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # randomize projection matrix
        runner.model.module.decode_head.loss_hist.proj_mat = torch.randn_like(runner.model.module.decode_head.loss_hist.proj_mat)
        runner.model.module.decode_head.loss_hist.proj_mat /= torch.sum(runner.model.module.decode_head.loss_hist.proj_mat**2, dim=1).sqrt().unsqueeze(dim=1)

        ## Split space to groups
        # feature_num_pre = runner.model.module.decode_head.loss_hist.proj_mat.shape[1]
        # feature_num_post = runner.model.module.decode_head.loss_hist.proj_mat.shape[0]
        # group_size = 8
        # groups_num = int(feature_num_pre / group_size)
        # feature_num_per_group = int(feature_num_post / groups_num)
        # for g in range(groups_num):
        #     runner.model.module.decode_head.loss_hist.proj_mat[g*feature_num_per_group:(g+1)*feature_num_per_group, :g*group_size] = 0
        #     runner.model.module.decode_head.loss_hist.proj_mat[g*feature_num_per_group:(g+1)*feature_num_per_group, (g+1)*group_size:] = 0
        # runner.model.module.decode_head.loss_hist.proj_mat /= torch.sum(runner.model.module.decode_head.loss_hist.proj_mat**2, dim=1).sqrt().unsqueeze(dim=1)

        ## Debug - take only N first dimensions, the other are zeroed out
        # runner.model.module.decode_head.loss_hist.proj_mat[:, 10:] = 0
        # runner.model.module.decode_head.loss_hist.proj_mat /= torch.sum(runner.model.module.decode_head.loss_hist.proj_mat**2, dim=1).sqrt().unsqueeze(dim=1)

        ## Debug - take only exact principal components
        # runner.model.module.decode_head.loss_hist.proj_mat[:] = 0
        # indices = torch.randint(0, 256, (runner.model.module.decode_head.loss_hist.proj_mat.shape[0], 1))
        # for f in range(0, runner.model.module.decode_head.loss_hist.proj_mat.shape[0]):
        #     runner.model.module.decode_head.loss_hist.proj_mat[f, indices[f]] = 1

        self.samples_num_all_curr_epoch[:] = 0
        runner.model.module.decode_head.loss_hist.samples_num_all_curr_epoch[:] = 0

    def after_train_epoch(self, runner):
        """Synchronizing norm."""

        self.samples_num_all_curr_epoch = runner.model.module.decode_head.loss_hist.samples_num_all_curr_epoch
        self.miu_all = runner.model.module.decode_head.loss_hist.miu_all / (np.expand_dims(self.samples_num_all_curr_epoch,0) + 1e-12)
        self.moment2_all = runner.model.module.decode_head.loss_hist.moment2_all / (np.expand_dims(self.samples_num_all_curr_epoch,0) + 1e-12)
        self.moment2_mat_all = runner.model.module.decode_head.loss_hist.moment2_mat_all / (np.expand_dims(self.samples_num_all_curr_epoch,0) + 1e-12)
        self.cov_mat_all = runner.model.module.decode_head.loss_hist.cov_mat_all
        self.loss_per_dim_all = runner.model.module.decode_head.loss_hist.loss_per_dim_all
        for c in range(0, self.num_classes):
            if self.samples_num_all_curr_epoch[c]:
                self.covinv_mat_all[:, :, c] = np.linalg.inv(self.cov_mat_all[:, :, c])

                eigen_vals, eigen_vecs = np.linalg.eig(self.cov_mat_all[:, :, c])
                indices = np.argsort(eigen_vals)[::-1]  # From high to low
                self.eigen_vals[:, c] = eigen_vals[indices]
                self.eigen_vecs[:, :, c] = eigen_vecs[:, indices]

        filename = os.path.join(self.save_folder, 'epoch_{}'.format(runner.epoch+1)+'.pickle')
        if not np.mod(runner.epoch+1, 5):
            with open(filename, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
