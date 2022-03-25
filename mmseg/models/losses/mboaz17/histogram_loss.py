# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.miu_all = np.zeros((self.features_num, self.num_classes))  # For in-epoch calculations
        self.moment2_all = np.zeros((self.features_num, self.num_classes))  # For in-epoch calculations
        self.moment2_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))  # For in-epoch calculations
        self.samples_num_all = np.zeros(self.num_classes)  # cumulated samples number of each class over the current epoch

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

        if isinstance(label, list):  # TODO: remove?
            label = label[0]
        if len(label.shape) > 3:
            label = label[:,0,:,:]

        self.miu_all[:] = 0
        self.moment2_all[:] = 0
        self.moment2_mat_all[:] = 0
        self.samples_num_all[:] = 0

        # TODO: Handle batch size > 1  !!!
        batch_size = feature.shape[0]
        feature_dim = feature.shape[1]
        height = feature.shape[2]
        width = feature.shape[3]
        if feature.requires_grad:  # TODO: Improve this indication of training vs. validation.
            ortho_mat = torch.tensor(ortho_group.rvs(dim=feature_dim), device='cuda').to(torch.float)
            feature = torch.matmul(ortho_mat, feature.view((feature_dim, -1))).view((batch_size, feature_dim, height, width))

        feature_upscaled = torch.nn.functional.interpolate(feature, (label.shape[1], label.shape[2]))

        class_interval = 1
        active_classes_num = 0
        loss_hist = torch.tensor(0.0, device='cuda')
        for c in range(self.num_classes):
            miu_unnormalized = np.zeros(feature_dim)
            moment2_unnormalized = np.zeros(feature_dim)
            moment2_mat_unnormalized = np.zeros((feature_dim, feature_dim))
            class_indices = (label[0, :, :] == torch.tensor(c, device='cuda')).nonzero()
            sampled_indices = torch.linspace(0, len(class_indices) - 1,
                                             np.int32(len(class_indices) / class_interval)).long()
            samples_num = len(sampled_indices)
            if samples_num:    # if class_indices.size(0):
                active_classes_num += 1

                feat_vecs_curr = feature_upscaled[0, :, class_indices[sampled_indices, 0], class_indices[sampled_indices, 1]]
                miu_unnormalized = torch.sum(feat_vecs_curr, dim=1).detach().cpu()
                miu = miu_unnormalized / samples_num
                moment2_unnormalized = torch.sum(feat_vecs_curr**2, dim=1).detach().cpu()
                moment2_mat_unnormalized = torch.matmul(feat_vecs_curr, feat_vecs_curr.T).detach().cpu()
                moment2 = moment2_unnormalized / samples_num
                var = moment2 - miu**2 + 1e-12  # torch.mean((feat_vecs_curr-miu.unsqueeze(dim=1))**2, dim=1) + 1e-12
                if c == 0:  # TODO: remove this after removing the background from the classes list
                    var[:] = 1e-12
                    continue

                if samples_num >= 1000:  # compare histograms only when enough samples exist
                    miu_t = torch.tensor(miu, device='cuda')
                    var_t = torch.tensor(var, device='cuda')
                    std_t = var_t.sqrt()
                    var_sample_t = var_t / 25

                    bins = [miu_t + k*std_t for k in [-3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3.]]
                    target_values = torch.zeros((feature_dim, len(bins)), device='cuda')
                    sample_values = torch.zeros((feature_dim, len(bins)), device='cuda')
                    for ind, bin in enumerate(bins):
                        with torch.no_grad():
                            target_values[:, ind] = torch.exp( -0.5 * (bin - miu_t)**2 / var_t) * \
                                            (1/torch.sqrt(2*torch.pi*var_t))

                        sample_values[:, ind] = torch.sum(torch.exp( -0.5 * (bin.unsqueeze(dim=1) - feat_vecs_curr) ** 2 / var_sample_t.unsqueeze(dim=1)) \
                                      * (1 / torch.sqrt(2 * torch.pi * var_sample_t.unsqueeze(dim=1))), dim=1)

                    hist_values = sample_values / sample_values.sum(dim=1).unsqueeze(dim=1)
                    target_values = target_values / target_values.sum(dim=1).unsqueeze(dim=1)

                    loss_hist += self.loss_weight * F.smooth_l1_loss(hist_values, target_values)

            self.miu_all[:, c] = miu_unnormalized
            self.moment2_all[:, c] = moment2_unnormalized
            self.moment2_mat_all[:, :, c] = moment2_mat_unnormalized
            self.samples_num_all[c] = samples_num

        assert (active_classes_num > 0)
        loss_hist /= active_classes_num
        print('loss_hist = {}'.format(loss_hist))

        return loss_hist

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
