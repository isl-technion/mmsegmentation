# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        # TODO: Handle batch size > 1  !!!
        feature_dim = feature.shape[1]
        feature_upscaled = torch.nn.functional.interpolate(feature, (label.shape[2], label.shape[3]))

        miu_all = torch.zeros((feature_dim, self.num_classes), device='cuda')
        var_all = torch.zeros((feature_dim, self.num_classes), device='cuda')
        samples_num_all = torch.zeros(self.num_classes, device='cuda')

        class_interval = 1
        active_classes_num = 0
        loss_hist = torch.tensor(0.0, device='cuda')
        for c in range(self.num_classes):
            miu = torch.zeros(feature_dim, device='cuda')
            var = torch.zeros(feature_dim, device='cuda')
            class_indices = (label[0, 0, :, :] == torch.tensor(c, device='cuda')).nonzero()
            sampled_indices = torch.linspace(0, len(class_indices) - 1,
                                             np.int32(len(class_indices) / class_interval)).long()
            samples_num = len(sampled_indices)
            if samples_num>=1000:    # if class_indices.size(0):
                active_classes_num += 1

                feat_vecs_curr = feature_upscaled[0, :, class_indices[sampled_indices, 0], class_indices[sampled_indices, 1]]
                miu = torch.mean(feat_vecs_curr, dim=1)
                var = torch.mean((feat_vecs_curr-miu.unsqueeze(dim=1))**2, dim=1) + 1e-10
                std = var.sqrt()
                var_sample = var / 25

                bins = [miu + k*std for k in range (-3, 4, 1)]
                target_values = torch.zeros((feature_dim, len(bins)), device='cuda')
                sample_values = torch.zeros((feature_dim, len(bins)), device='cuda')
                for ind, bin in enumerate(bins):
                    with torch.no_grad():
                        target_values[:, ind] = torch.exp( -0.5 * (bin - miu)**2 / var) * \
                                        (1/torch.sqrt(2*torch.pi*var))


                    sample_values[:, ind] = torch.sum(torch.exp( -0.5 * (bin.unsqueeze(dim=1) - feat_vecs_curr) ** 2 / var_sample.unsqueeze(dim=1)) \
                                  * (1 / torch.sqrt(2 * torch.pi * var_sample.unsqueeze(dim=1))), dim=1)

                hist_values = sample_values / sample_values.sum(dim=1).unsqueeze(dim=1)
                target_values = target_values / target_values.sum(dim=1).unsqueeze(dim=1)

                loss_hist += self.loss_weight * F.smooth_l1_loss(hist_values, target_values)

            miu_all[:, c] = miu
            var_all[:, c] = var
            samples_num_all[c] = samples_num

        loss_hist /= active_classes_num
        print(loss_hist)

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
