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
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg, act_cfg=None) # <mboaz17>

    def forward(self, inputs, label=None, hist_model=None):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if label is not None:
            loss, out = self.loss_hist(out, label)

        if hist_model is not None:
            batch_size = out.shape[0]
            feature_dim = out.shape[1]
            height = out.shape[2]
            width = out.shape[3]
            prob_scores = torch.zeros((batch_size, hist_model.num_classes, height, width), device='cuda')
            for c in range(0, hist_model.num_classes):
                miu_curr = torch.tensor(hist_model.miu_all[:,c], device='cuda').clone().detach().\
                    unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
                if 0:  # use variance
                    var_curr = torch.tensor(np.diag(hist_model.cov_mat_all[:, :, c]), device='cuda').clone().detach().\
                        unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
                    weight_factors = torch.tensor(1 / (hist_model.loss_per_dim_all[:, c] + 1e-20), device='cuda').clone().detach()
                    # weight_factors *= weight_factors
                    weight_factors /= weight_factors.mean()
                    weight_factors = weight_factors.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
                    maha_dist = (weight_factors * (out - miu_curr)**2 / var_curr).mean(dim=1)
                    # log_prob = -0.5 * (maha_dist + torch.log(var_curr.prod()) + out.shape[1]*torch.log(2*torch.tensor(torch.pi)))
                    log_prob = -0.5 * maha_dist
                    if c == 0:  # TODO: remove...
                        log_prob[:] = -1e6
                else:  # use covariance
                    covinv_curr = torch.tensor(hist_model.covinv_mat_all[:, :, c], device='cuda').clone().detach()
                    diff = (out - miu_curr).view((feature_dim, -1))
                    maha_dist = (diff * torch.matmul(covinv_curr, diff)).mean(dim=0).view((1, height, width))
                    log_prob = -0.5 * maha_dist
                prob_scores[0, c] = log_prob[0]

            prob_scores[prob_scores.isinf()] = -1e6
            # res = prob_scores.argmax(dim=1)
            out = prob_scores
            return out

        out = self.cls_seg(out)

        if label is not None:
            return out, loss

        return out
