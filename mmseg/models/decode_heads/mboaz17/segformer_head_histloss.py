# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.mboaz17.decode_head_histloss import BaseDecodeHead
from mmseg.ops import resize

from mmseg.models.losses.mboaz17.histogram_loss import HistogramLoss
import numpy as np

@HEADS.register_module()
class SegformerHeadHistLoss(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None))  # <mboaz17>

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg, act_cfg=None)  # <mboaz17>

        self.relu_operation = ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)  # <mboaz17>

    def forward(self, inputs, label=None, hist_model=None):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        loss_hist_vals = []
        prob_scores_list = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            conv_x = conv(x)
            if label is not None:
                loss =  self.loss_hist_list[idx](conv_x, label)
                loss_hist_vals.append(loss)
            if hist_model is not None:
                prob_scores = calc_log_prob(conv_x, hist_model.models_list[idx+hist_model.layers_num_encoder])
                prob_scores_list.append(resize(
                    input=prob_scores,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
            res = resize(
                    input=conv_x,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
            outs.append(self.relu_operation.activate(res))  # mboaz17

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if label is not None:
            loss = self.loss_hist_list[-1](out, label)
            loss_hist_vals.append(loss)

        if hist_model is not None:
            prob_scores = calc_log_prob(out, hist_model.models_list[len(inputs) + hist_model.layers_num_encoder])
            prob_scores_list.append(prob_scores)
            return prob_scores_list

        out = self.relu_operation.activate(out)  # doing the removed activation function of the fusion_conv

        out = self.cls_seg(out)

        if label is not None:
            return out, loss_hist_vals

        return out


def calc_log_prob(feature, hist_model):
    batch_size = feature.shape[0]
    feature_dim = feature.shape[1]
    height = feature.shape[2]
    width = feature.shape[3]
    prob_scores = -1e6 * torch.ones((batch_size, hist_model.num_classes, height, width), device='cuda')
    for c in range(0, hist_model.num_classes):
        if not hist_model.samples_num_all_curr_epoch[c]:
            continue
        miu_curr = torch.tensor(hist_model.miu_all[:, c], device='cuda').clone().float().detach(). \
            unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
        if 0:  # use variance
            var_curr = torch.tensor(np.diag(hist_model.cov_mat_all[:, :, c]), device='cuda').clone().detach(). \
                unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
            weight_factors = torch.tensor(1 / (hist_model.loss_per_dim_all[:, c] + 1e-20),
                                          device='cuda').clone().detach()
            # weight_factors *= weight_factors
            weight_factors /= weight_factors.mean()
            weight_factors = weight_factors.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
            maha_dist = (weight_factors * (feature - miu_curr) ** 2 / var_curr).mean(dim=1)
            # log_prob = -0.5 * (maha_dist + torch.log(var_curr.prod()) + feature.shape[1]*torch.log(2*torch.tensor(torch.pi)))
            log_prob = -0.5 * maha_dist
        elif 1:  # use covariance
            covinv_curr = torch.from_numpy(hist_model.covinv_mat_all[:, :, c]).float().to('cuda')
            # epsilon = np.maximum( - hist_model.eigen_vals_all[:, c].min(), 0) + 1e-12
            # covinv_curr = torch.from_numpy(hist_model.cov_mat_all[:, :, c] + epsilon * np.eye(hist_model.features_num)).float().to('cuda')
            diff = (feature - miu_curr).view((feature_dim, -1))
            maha_dist = (diff * torch.matmul(covinv_curr, diff)).mean(dim=0).view((1, height, width))
        else:  # use PCA-trained covariance
            eigen_vecs_t = torch.from_numpy(hist_model.eigen_vecs_all[:, :, c]).float().to('cuda')
            eigen_vals_t = torch.from_numpy(hist_model.eigen_vals_all[:, c]).float().to('cuda')
            diff = (feature - miu_curr).view((feature_dim, -1))
            proj = torch.matmul(eigen_vecs_t.T, diff)
            proj = proj / torch.maximum(eigen_vals_t.sqrt(), torch.tensor(1e-15)).unsqueeze(dim=1)
            if 1:  # equal weight per dimensions
                maha_dist = (proj ** 2).mean(dim=0).view((1, height, width))
            else:  # larger eigen_vals get more emphasis
                weight_factors = torch.maximum(eigen_vals_t.sqrt(), torch.tensor(1e-15))
                weight_factors /= weight_factors.mean()
                maha_dist = (weight_factors.unsqueeze(dim=1) * (proj ** 2)).mean(dim=0).view((1, height, width))
        log_prob = -0.5 * maha_dist
        prob_scores[0, c] = log_prob[0]

    prob_scores[prob_scores.isinf()] = -1e6
    prob_scores[prob_scores.isnan()] = -1e6
    # res = prob_scores.argmax(dim=1)

    return prob_scores
