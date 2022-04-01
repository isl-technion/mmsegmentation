# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ortho_group # Requires version 0.18 of scipy

from ...builder import LOSSES
from ..utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class HistogramLoss(nn.Module):
    """HistogramLoss.  <mboaz17>

    Args:
        num_classes (int): Number of GT classes
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_hist'.
    """

    def __init__(self,
                 num_classes,
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_hist'):
        super(HistogramLoss, self).__init__()

        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self._loss_name = loss_name

        self.features_num = 256  # 16
        self.iters = 0
        self.miu_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.cov_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))  # For in-epoch calculations
        self.samples_num_all = np.zeros(self.num_classes)
        self.loss_per_dim_all = np.zeros((self.features_num, self.num_classes))
        self.miu_all_curr_batch = np.zeros((self.features_num, self.num_classes))
        self.moment2_all_curr_batch = np.zeros((self.features_num, self.num_classes))
        self.moment2_mat_all_curr_batch = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.samples_num_all_curr_batch = np.zeros(self.num_classes)

        self.alpha_hist = 0.995
        self.bins_num = 51
        self.bins_vals = np.linspace(-5, 5, self.bins_num)
        self.hist_values = np.ones((self.features_num, self.bins_num, self.num_classes)) / self.bins_num
        self.epsilon = 1e-12

    def forward(self,
                feature,
                label,
                weight=None,
                avg_factor=None,
                **kwargs):
        """Forward function."""
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        self.miu_all_curr_batch[:] = 0
        self.moment2_all_curr_batch[:] = 0
        self.moment2_mat_all_curr_batch[:] = 0
        self.samples_num_all_curr_batch[:] = 0

        # TODO: Handle batch size > 1  !!!
        batch_size = feature.shape[0]
        feature_dim = feature.shape[1]
        height = feature.shape[2]
        width = feature.shape[3]
        label_downscaled = torch.nn.functional.interpolate(label.to(torch.float32), (height, width)).to(torch.long)
        ortho_mat = torch.tensor(ortho_group.rvs(dim=feature_dim), device='cuda').to(torch.float)

        val_low = 1e3
        val_high = -1e3
        class_interval = 1
        active_classes_num = 0
        loss_hist = torch.tensor(0.0, device='cuda')
        for c in range(self.num_classes):
            miu_unnormalized = np.zeros(feature_dim)
            moment2_unnormalized = np.zeros(feature_dim)
            moment2_mat_unnormalized = np.zeros((feature_dim, feature_dim))
            class_indices = (label_downscaled[0, 0, :, :] == torch.tensor(c, device='cuda')).nonzero()
            sampled_indices = torch.linspace(0, len(class_indices) - 1,
                                             np.int32(len(class_indices) / class_interval)).long()
            samples_num = len(sampled_indices)
            if samples_num:    # if class_indices.size(0):
                feat_vecs_curr = feature[0, :, class_indices[sampled_indices, 0], class_indices[sampled_indices, 1]]
                miu_unnormalized = torch.sum(feat_vecs_curr, dim=1).detach().cpu().numpy()
                miu = miu_unnormalized / samples_num
                moment2_unnormalized = torch.sum(feat_vecs_curr**2, dim=1).detach().cpu().numpy()
                moment2 = moment2_unnormalized / samples_num
                moment2_mat_unnormalized = torch.matmul(feat_vecs_curr, feat_vecs_curr.T).detach().cpu().numpy()
                moment2_mat = moment2_mat_unnormalized / samples_num

                if self.samples_num_all[c]:
                    self.miu_all[:, c] = self.alpha_hist * self.miu_all[:, c] + (1-self.alpha_hist) * miu
                    self.moment2_all[:, c] = self.alpha_hist * self.moment2_all[:, c] + (1-self.alpha_hist) * moment2
                    self.moment2_mat_all[:, :, c] = self.alpha_hist * self.moment2_mat_all[:, :, c] + (1-self.alpha_hist) * moment2_mat
                else:
                    self.miu_all[:, c] = miu
                    self.moment2_all[:, c] = moment2
                    self.moment2_mat_all[:, :, c] = moment2_mat

                self.cov_mat_all[:, :, c] = self.moment2_mat_all[:, :, c] - \
                                            np.matmul(self.miu_all[:, c:c + 1], self.miu_all[:, c:c + 1].T) + \
                                            self.epsilon * np.eye(self.features_num)

                # var = np.maximum(self.moment2_all[:, c] - self.miu_all[:, c]**2, self.epsilon)
                var = np.diag(self.cov_mat_all[:, :, c])
                miu_t = torch.tensor(self.miu_all[:,c], device='cuda', dtype=torch.float32).detach().clone()
                var_t = torch.tensor(var, device='cuda', dtype=torch.float32).detach().clone()
                std_t = var_t.sqrt()
                var_sample_t = var_t / 25

                # feature rotation around the class mean
                if 0:  # np.mod(self.iters, 30):  # Don't rotate axes every () iterations
                    eigen_vals, eigen_vecs = np.linalg.eig(self.cov_mat_all[:, :, c])
                    eigen_vecs_t = torch.from_numpy(eigen_vecs).float().to('cuda')
                    eigen_vals_t = torch.from_numpy(eigen_vals).float().to('cuda')

                    feat_vecs_curr_centered = feat_vecs_curr - miu_t.unsqueeze(dim=1)
                    proj = torch.matmul(eigen_vecs_t.T, feat_vecs_curr_centered)
                    proj_normalized = proj / eigen_vals_t.sqrt().unsqueeze(dim=1)
                    proj_rotated = torch.matmul(ortho_mat, proj_normalized)
                    proj = proj_rotated * eigen_vals_t.sqrt().unsqueeze(dim=1)
                    if eigen_vals.min() >= self.epsilon:
                        feat_vecs_curr = torch.matmul(eigen_vecs_t, proj) + miu_t.unsqueeze(dim=1)
                        feature[0, :, class_indices[sampled_indices, 0], class_indices[sampled_indices, 1]] = feat_vecs_curr

                    # feat_vecs_curr = miu_t.unsqueeze(dim=1) + torch.matmul(ortho_mat, feat_vecs_curr - miu_t.unsqueeze(dim=1))
                    # feature[0, :, class_indices[sampled_indices, 0], class_indices[sampled_indices, 1]] = feat_vecs_curr

                target_values = torch.zeros((feature_dim, self.bins_num), device='cuda')
                sample_values = torch.zeros((feature_dim, self.bins_num), device='cuda')
                for ind, bin in enumerate(self.bins_vals):
                    with torch.no_grad():
                        target_values[:, ind] = torch.exp( -0.5 * (torch.tensor(bin - self.miu_all[:, c], device='cuda'))**2 / var_t) * \
                                        (1/torch.sqrt(2*torch.pi*var_t))
                    sample_values[:, ind] = torch.sum(torch.exp( -0.5 * (bin - feat_vecs_curr) ** 2 / var_sample_t.unsqueeze(dim=1)) \
                                  * (1 / torch.sqrt(2 * torch.pi * var_sample_t.unsqueeze(dim=1))), dim=1)

                hist_values = sample_values / sample_values.sum(dim=1).unsqueeze(dim=1)
                target_values = target_values / target_values.sum(dim=1).unsqueeze(dim=1)

                if c > 0:  # TODO: remove this after removing the background from the classes list
                    active_classes_num += 1
                    if self.samples_num_all[c]:
                        hist_values_filtered = self.alpha_hist * torch.tensor(self.hist_values[:, :, c], device='cuda') + (1 - self.alpha_hist) * hist_values
                    else:
                        hist_values_filtered = hist_values

                    loss_vect = torch.zeros(feature_dim, device='cuda')
                    for f in range(feature_dim):
                        # loss_vect[f] = F.smooth_l1_loss(hist_values_filtered[f], target_values[f])
                        loss_vect[f] = 1 - torch.sum(torch.sqrt(hist_values_filtered[f] * target_values[f] + self.epsilon))
                        loss_hist += loss_vect[f]
                        if c==11:
                            aaa=1
                    self.loss_per_dim_all[:, c] = loss_vect.detach().cpu().numpy()
                    if c==11:
                        print(1000*loss_vect.sort()[0][::50].detach().cpu().numpy())
                    self.hist_values[:, :, c] =  hist_values_filtered.detach().cpu().numpy()

                    if 0:  # for c=11 (or 14?), after several epochs
                        f1 = 50
                        f2 = 210
                        feat_vecs_curr_2d = feat_vecs_curr[(f1, f2), :]
                        sample_values_2d = torch.zeros((51, 51), device='cuda')
                        var_sample_t_2d = var_sample_t.unsqueeze(dim=1)[(f1, f2), :]
                        for ind1, bin1 in enumerate(self.bins_vals):
                            for ind2, bin2 in enumerate(self.bins_vals):
                                bin = torch.tensor((bin1, bin2), device='cuda').unsqueeze(dim=1)
                                with torch.no_grad():
                                    sample_values_2d[ind1, ind2] = torch.sum(
                                        torch.exp(-0.5 * torch.sum((bin - feat_vecs_curr_2d) ** 2 / var_sample_t_2d, dim=0))
                                        * (1 / torch.sqrt(2 * torch.pi * var_sample_t_2d)))

                        # Make data.
                        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                        X = self.bins_vals
                        Y = self.bins_vals
                        X, Y = np.meshgrid(X, Y)
                        Z = sample_values_2d.detach().cpu().numpy()
                        ax.plot_surface(X, Y, Z)
                        plt.show()

                        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                        Z = torch.matmul(hist_values[f1:f1+1].T, hist_values[f2:f2+1]).detach().cpu().numpy()
                        ax.plot_surface(X, Y, Z)
                        plt.show()

                    val_low = np.minimum(val_low, (miu_t - 3*std_t).min().detach().cpu().numpy())
                    val_high = np.maximum(val_high, (miu_t + 3*std_t).max().detach().cpu().numpy())

                self.samples_num_all[c] += samples_num

            self.miu_all_curr_batch[:, c] = miu_unnormalized
            self.moment2_all_curr_batch[:, c] = moment2_unnormalized
            self.moment2_mat_all_curr_batch[:, :, c] = moment2_mat_unnormalized
            self.samples_num_all_curr_batch[c] = samples_num

        loss_hist /= (active_classes_num + self.epsilon)
        print('loss_hist = {}, val_low = {}, val_high = {}, active = {}'.format(self.loss_weight * loss_hist, val_low, val_high, active_classes_num))

        self.iters += 1
        return self.loss_weight * loss_hist, feature

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
