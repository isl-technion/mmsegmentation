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

    def __init__(self, num_classes=2, features_num=1, layers_num_decoder=5, layers_num_encoder=4):

        self.save_folder = ''  # will be determined later
        self.layers_num_decoder = layers_num_decoder
        self.layers_num_encoder = layers_num_encoder
        self.models_list = []

        for l in range(self.layers_num_decoder):
            model = ModelParams(num_classes=num_classes, features_num=features_num)
            self.models_list.append(model)

        for l in range(self.layers_num_encoder):
            model = ModelParams(num_classes=num_classes, features_num=[32, 64, 160, 256][l])
            self.models_list.append(model)

    def before_train_iter(self, runner):
        # Increase the loss weight as more batches are involved in the histogram estimation
        for l in range(self.layers_num_decoder):
            runner.model.module.decode_head.loss_hist_list[l].relative_weight = \
                (runner.inner_iter+1) / runner.data_loader.sampler.num_samples
        for l in range(self.layers_num_encoder):
            runner.model.module.backbone.loss_hist_list[l].relative_weight = \
                (runner.inner_iter+1) / runner.data_loader.sampler.num_samples

    def before_train_epoch(self, runner):

        self.save_folder = os.path.join(runner.work_dir, 'hooks')
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        for l in range(self.layers_num_decoder):
            # Start training only after 10 epochs
            runner.model.module.decode_head.loss_hist_list[l].loss_weight = runner.model.module.decode_head.loss_hist_list[l].loss_weight_orig * \
                                                                    (runner.epoch>=10)

            # randomize projection matrix
            runner.model.module.decode_head.loss_hist_list[l].proj_mat = torch.randn_like(runner.model.module.decode_head.loss_hist_list[l].proj_mat)
            runner.model.module.decode_head.loss_hist_list[l].proj_mat /= torch.sum(runner.model.module.decode_head.loss_hist_list[l].proj_mat**2, dim=1).sqrt().unsqueeze(dim=1)

            self.models_list[l].samples_num_all_curr_epoch[:] = 0
            runner.model.module.decode_head.loss_hist_list[l].samples_num_all_curr_epoch[:] = 0

        for l in range(self.layers_num_encoder):
            # Start training only after 10 epochs
            runner.model.module.backbone.loss_hist_list[l].loss_weight = runner.model.module.backbone.loss_hist_list[l].loss_weight_orig * \
                                                                    (runner.epoch>=10)

            # randomize projection matrix
            runner.model.module.backbone.loss_hist_list[l].proj_mat = torch.randn_like(runner.model.module.backbone.loss_hist_list[l].proj_mat)
            runner.model.module.backbone.loss_hist_list[l].proj_mat /= torch.sum(runner.model.module.backbone.loss_hist_list[l].proj_mat**2, dim=1).sqrt().unsqueeze(dim=1)

            self.models_list[l+self.layers_num_decoder].samples_num_all_curr_epoch[:] = 0
            runner.model.module.backbone.loss_hist_list[l].samples_num_all_curr_epoch[:] = 0

    def after_train_epoch(self, runner):
        """Synchronizing norm."""

        for l in range(self.layers_num_decoder):
            self.models_list[l].samples_num_all_curr_epoch = runner.model.module.decode_head.loss_hist_list[l].samples_num_all_curr_epoch
            self.models_list[l].miu_all = runner.model.module.decode_head.loss_hist_list[l].miu_all / (np.expand_dims(self.models_list[l].samples_num_all_curr_epoch,0) + 1e-12)
            self.models_list[l].moment2_all = runner.model.module.decode_head.loss_hist_list[l].moment2_all / (np.expand_dims(self.models_list[l].samples_num_all_curr_epoch,0) + 1e-12)
            self.models_list[l].moment2_mat_all = runner.model.module.decode_head.loss_hist_list[l].moment2_mat_all / (np.expand_dims(self.models_list[l].samples_num_all_curr_epoch,0) + 1e-12)
            self.models_list[l].cov_mat_all = runner.model.module.decode_head.loss_hist_list[l].cov_mat_all
            for c in range(0, self.models_list[l].num_classes):
                if self.models_list[l].samples_num_all_curr_epoch[c]:
                    self.models_list[l].covinv_mat_all[:, :, c] = np.linalg.inv(self.models_list[l].cov_mat_all[:, :, c])

                    eigen_vals, eigen_vecs = np.linalg.eig(self.models_list[l].cov_mat_all[:, :, c])
                    indices = np.argsort(eigen_vals)[::-1]  # From high to low
                    self.models_list[l].eigen_vals_all[:, c] = eigen_vals[indices]
                    self.models_list[l].eigen_vecs_all[:, :, c] = eigen_vecs[:, indices]

        for l in range(self.layers_num_encoder):
            self.models_list[l+self.layers_num_decoder].samples_num_all_curr_epoch = runner.model.module.backbone.loss_hist_list[l].samples_num_all_curr_epoch
            self.models_list[l+self.layers_num_decoder].miu_all = runner.model.module.backbone.loss_hist_list[l].miu_all / (np.expand_dims(self.models_list[l+self.layers_num_decoder].samples_num_all_curr_epoch,0) + 1e-12)
            self.models_list[l+self.layers_num_decoder].moment2_all = runner.model.module.backbone.loss_hist_list[l].moment2_all / (np.expand_dims(self.models_list[l+self.layers_num_decoder].samples_num_all_curr_epoch,0) + 1e-12)
            self.models_list[l+self.layers_num_decoder].moment2_mat_all = runner.model.module.backbone.loss_hist_list[l].moment2_mat_all / (np.expand_dims(self.models_list[l+self.layers_num_decoder].samples_num_all_curr_epoch,0) + 1e-12)
            self.models_list[l+self.layers_num_decoder].cov_mat_all = runner.model.module.backbone.loss_hist_list[l].cov_mat_all
            for c in range(0, self.models_list[l+self.layers_num_decoder].num_classes):
                if self.models_list[l+self.layers_num_decoder].samples_num_all_curr_epoch[c]:
                    self.models_list[l+self.layers_num_decoder].covinv_mat_all[:, :, c] = np.linalg.inv(self.models_list[l+self.layers_num_decoder].cov_mat_all[:, :, c])

                    eigen_vals, eigen_vecs = np.linalg.eig(self.models_list[l+self.layers_num_decoder].cov_mat_all[:, :, c])
                    indices = np.argsort(eigen_vals)[::-1]  # From high to low
                    self.models_list[l+self.layers_num_decoder].eigen_vals_all[:, c] = eigen_vals[indices]
                    self.models_list[l+self.layers_num_decoder].eigen_vecs_all[:, :, c] = eigen_vecs[:, indices]

        filename = os.path.join(self.save_folder, 'epoch_{}'.format(runner.epoch+1)+'.pickle')
        if not np.mod(runner.epoch+1, 5):
            with open(filename, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ModelParams(object):
    def __init__(self, num_classes=2, features_num=1):
        self.num_classes = num_classes
        self.features_num = features_num
        self.miu_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.cov_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.covinv_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.eigen_vals_all = np.zeros((self.features_num, self.num_classes))
        self.eigen_vecs_all = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.samples_num_all_curr_epoch = np.zeros(self.num_classes)
