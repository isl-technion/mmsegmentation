# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from ...builder import LOSSES
from ..utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class HistogramLoss(nn.Module):
    """HistogramLoss.  <messi>

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
                 features_num=256,
                 directions_num=10000,
                 loss_name='loss_hist'):
        super(HistogramLoss, self).__init__()

        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.loss_weight_orig = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self._loss_name = loss_name

        self.features_num = features_num
        self.directions_num = directions_num
        self.iters_since_init = 0
        self.iters_since_epoch_init = 0
        self.miu_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_all = np.zeros((self.features_num, self.num_classes))
        self.moment2_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.cov_mat_all = np.zeros((self.features_num, self.features_num, self.num_classes))  # For in-epoch calculations
        self.samples_num_all = np.zeros(self.num_classes)
        self.samples_num_all_curr_epoch = np.zeros(self.num_classes)
        self.samples_num_all_in_loss = np.zeros(self.num_classes)

        self.estimate_hist_flag = False
        self.alpha_hist = 0.95  # was 0.995 when samples_num was not considered
        self.bins_num = 81
        self.bins_vals = np.linspace(-4, 4, self.bins_num)
        self.hist_values = np.ones((self.directions_num, self.bins_num, self.num_classes)) / self.bins_num
        self.moment1_proj = np.zeros((self.directions_num, self.num_classes))
        self.moment2_proj = np.zeros((self.directions_num, self.num_classes))
        self.moment3_proj = np.zeros((self.directions_num, self.num_classes))
        self.moment4_proj = np.zeros((self.directions_num, self.num_classes))
        self.epsilon = 1e-12
        self.relative_weight = 1.0  # how much to multiply the loss. It's changed every iteration in the histloss_hook

        self.proj_mat = torch.randn((self.directions_num, self.features_num), device='cuda')  # it should be constant within an epoch!
        self.proj_mat /= torch.sum(self.proj_mat**2, dim=1).sqrt().unsqueeze(dim=1)
        self.loss_per_dim_all = np.zeros((self.directions_num, self.num_classes))

        self.miu_all_prev = np.zeros((self.features_num, self.num_classes))
        self.eigen_vecs_prev = np.zeros((self.features_num, self.features_num, self.num_classes))
        self.eigen_vals_prev = np.zeros((self.features_num, self.num_classes))
        self.model_prev_exists =  np.zeros(self.num_classes)

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

        samples_num_all_in_loss_pre = np.copy(self.samples_num_all_in_loss)

        # TODO: Handle batch size > 1  !!!
        batch_size = feature.shape[0]
        feature_dim = feature.shape[1]
        height = feature.shape[2]
        width = feature.shape[3]
        label_downscaled = torch.nn.functional.interpolate(label.to(torch.float32), (height, width)).to(torch.long)

        class_interval = 1
        active_classes_num = 0
        loss_hist = torch.tensor(0.0, device='cuda')
        if self.loss_weight == 0:  # save time
            return loss_hist

        loss_kurtosis = torch.tensor(0.0, device='cuda')
        loss_moment2 = torch.tensor(0.0, device='cuda')
        loss_hist_vect = torch.zeros(self.num_classes, device='cuda')
        loss_kurtosis_vect = torch.zeros(self.num_classes, device='cuda')
        loss_moment2_vect = torch.zeros(self.num_classes, device='cuda')
        for c in range(0, self.num_classes):   # TODO: start from 0 after removing the background from the classes list
            miu_unnormalized = np.zeros(feature_dim)
            moment2_unnormalized = np.zeros(feature_dim)
            moment2_mat_unnormalized = np.zeros((feature_dim, feature_dim))
            class_indices = (label_downscaled[:, 0, :, :] == torch.tensor(c, device='cuda')).nonzero()
            samples_num = len(class_indices)
            if samples_num:
                feat_vecs_curr = feature[class_indices[:, 0], :, class_indices[:, 1], class_indices[:, 2]].T
                miu_unnormalized = torch.sum(feat_vecs_curr, dim=1).detach().cpu().numpy()
                miu = miu_unnormalized / samples_num
                moment2_unnormalized = torch.sum(feat_vecs_curr**2, dim=1).detach().cpu().numpy()
                moment2 = moment2_unnormalized / samples_num
                moment2_mat_unnormalized = torch.matmul(feat_vecs_curr, feat_vecs_curr.T).detach().cpu().numpy()
                moment2_mat = moment2_mat_unnormalized / samples_num

                if self.samples_num_all_curr_epoch[c]:
                    self.miu_all[:, c] = self.miu_all[:, c] + miu_unnormalized
                    self.moment2_all[:, c] = self.moment2_all[:, c] + moment2_unnormalized
                    self.moment2_mat_all[:, :, c] = self.moment2_mat_all[:, :, c] + moment2_mat_unnormalized
                else:
                    self.miu_all[:, c] = miu_unnormalized
                    self.moment2_all[:, c] = moment2_unnormalized
                    self.moment2_mat_all[:, :, c] = moment2_mat_unnormalized
                self.samples_num_all[c] += samples_num
                self.samples_num_all_curr_epoch[c] += samples_num

                miu_curr = self.miu_all[:, c] / self.samples_num_all_curr_epoch[c]
                moment2_curr = self.moment2_all[:, c] / self.samples_num_all_curr_epoch[c]
                moment2_mat_curr = self.moment2_mat_all[:, :, c] / self.samples_num_all_curr_epoch[c]
                cov_eps = 0  # np.maximum( np.minimum( 1e-8 * 10000 / self.samples_num_all_curr_epoch[c], 1e-4), 1e-8)  # TODO: should depend on the dimension!
                cov_mat_all_curr = (moment2_mat_curr - np.matmul(np.expand_dims(miu_curr, 1), np.expand_dims(miu_curr, 1).T)) +\
                                            cov_eps * np.eye(self.features_num)
                self.cov_mat_all[:, :, c] = cov_mat_all_curr

                if self.model_prev_exists[c]:  # Estimated in the previous epoch, loaded in the hook
                    miu_curr = self.miu_all_prev[:, c]
                    eigen_vals = self.eigen_vals_prev[:, c]  # not actually used
                    eigen_vecs = self.eigen_vecs_prev[:, :, c]
                else:
                    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat_all_curr)
                    indices = np.argsort(eigen_vals)[::-1]  # From high to low
                    eigen_vals = eigen_vals[indices]
                    eigen_vecs = eigen_vecs[:, indices]

                    self.miu_all_prev[:, c] = miu_curr
                    self.eigen_vecs_prev[:, :, c] = eigen_vecs
                if np.any(np.iscomplex(eigen_vals)):  # or eigen_vals[-1] < self.epsilon:
                    print('Invalid: c = {}, eig_min = {}'.format(c, eigen_vals[-1]))
                    continue

                active_classes_num += 1
                miu_curr_t = torch.tensor(miu_curr, device='cuda').to(torch.float32)
                # eigen_vals = np.maximum(eigen_vals, self.epsilon)
                eigen_vecs_t = torch.from_numpy(eigen_vecs).float().to('cuda')
                eigen_vals_t = torch.from_numpy(eigen_vals).float().to('cuda')

                proj = torch.matmul(eigen_vecs_t.T, feat_vecs_curr - miu_curr_t.unsqueeze(dim=1))
                # proj_mat_curr = self.proj_mat * eigen_vals_t.sqrt().unsqueeze(dim=0)  # prioritizing axes according to their std
                proj_mat_curr = self.proj_mat * torch.ones_like(eigen_vals_t).unsqueeze(dim=0)  # prioritizing axes according to their std
                proj_mat_curr /= (proj_mat_curr ** 2).sum(dim=1).sqrt().unsqueeze(dim=1)  # normalizing to norm 1
                feat_vecs_curr = torch.matmul(proj_mat_curr, proj)

                del eigen_vecs_t, eigen_vals_t, proj, proj_mat_curr
                torch.cuda.empty_cache()

                moment1_proj = (feat_vecs_curr).sum(dim=1)
                moment2_proj = (feat_vecs_curr**2).sum(dim=1)  # non-centered
                moment3_proj = (feat_vecs_curr**3).sum(dim=1)  # non-centered
                moment4_proj = (feat_vecs_curr**4).sum(dim=1)  # non-centered
                if self.samples_num_all_in_loss[c]:
                    moment1_proj_filtered = torch.tensor(self.moment1_proj[:, c], device='cuda') + moment1_proj
                    moment2_proj_filtered = torch.tensor(self.moment2_proj[:, c], device='cuda') + moment2_proj
                    moment3_proj_filtered = torch.tensor(self.moment3_proj[:, c], device='cuda') + moment3_proj
                    moment4_proj_filtered = torch.tensor(self.moment4_proj[:, c], device='cuda') + moment4_proj
                else:
                    moment1_proj_filtered = moment1_proj
                    moment2_proj_filtered = moment2_proj
                    moment3_proj_filtered = moment3_proj
                    moment4_proj_filtered = moment4_proj

                self.samples_num_all_in_loss[c] += samples_num
                moment1_proj_for_loss = moment1_proj_filtered / self.samples_num_all_in_loss[c]
                moment2_proj_for_loss = moment2_proj_filtered / self.samples_num_all_in_loss[c]
                moment3_proj_for_loss = moment3_proj_filtered / self.samples_num_all_in_loss[c]
                moment4_proj_for_loss = moment4_proj_filtered / self.samples_num_all_in_loss[c]

                var_curr_t = moment2_proj_for_loss - moment1_proj_for_loss**2
                var_curr_t = torch.maximum(var_curr_t, torch.tensor(1e-16, device='cuda'))
                std_curr_t = var_curr_t.sqrt()
                self.eigen_vals_prev[:, c] = var_curr_t[:self.features_num].detach().cpu().numpy()  # Assuming the first features_num directions are principal components

                moment4_proj_for_loss_central = 1 * moment1_proj_for_loss**4 + 4 * (-1) * moment1_proj_for_loss ** 4 + \
                                                6 * 1 * moment2_proj_for_loss * moment1_proj_for_loss**2 + \
                                                4 * (-1) * moment3_proj_for_loss * moment1_proj_for_loss + 1 * moment4_proj_for_loss
                kurtosis = moment4_proj_for_loss_central / var_curr_t**2 - 3
                kurtosis = torch.maximum(torch.minimum(kurtosis, torch.tensor(3, device='cuda')), torch.tensor(-3, device='cuda'))
                loss_kurtosis_curr = kurtosis.abs().mean()  # kurtosis
                loss_moment2_curr = 0  # (moment2_proj_for_loss - moment1_proj_for_loss**2 - 1).abs().mean()  # moment2
                loss_kurtosis_vect[c] = loss_kurtosis_curr
                loss_moment2_vect[c] = loss_moment2_curr
                loss_hist_vect[c] = 1.0 * loss_kurtosis_vect[c] + 0.0 * loss_moment2_vect[c]
                if c==9:
                    aaa=1
                if c==14:
                    aaa=1
                self.moment1_proj[:, c] =  moment1_proj_filtered.detach().cpu().numpy()
                self.moment2_proj[:, c] =  moment2_proj_filtered.detach().cpu().numpy()
                self.moment3_proj[:, c] =  moment3_proj_filtered.detach().cpu().numpy()
                self.moment4_proj[:, c] =  moment4_proj_filtered.detach().cpu().numpy()

                if self.estimate_hist_flag:  # Default=False, but can be changed in the hook
                    feat_vecs_curr_norm = feat_vecs_curr / std_curr_t.unsqueeze(dim=1)
                    var_sample_t = torch.tensor(1 / 100, device='cuda')  # 25  # after whitening
                    with torch.no_grad():
                        target_values = torch.zeros((1, self.bins_num), device='cuda')
                        hist_values = torch.zeros((self.directions_num, self.bins_num), device='cuda')
                        for ind, bin in enumerate(self.bins_vals):
                            target_values[0, ind] = torch.exp(-0.5 * (torch.tensor(bin, device='cuda')) ** 2) * (
                                        1 / np.sqrt(2 * np.pi))
                            hist_values[:, ind] = torch.sum(torch.exp(-0.5 * (bin - feat_vecs_curr_norm) ** 2 / var_sample_t) *
                                                   (1 / torch.sqrt(2 * torch.pi * var_sample_t)), dim=1)

                        if self.samples_num_all_in_loss[c]:
                            hist_values_filtered = self.hist_values[:, :, c] + hist_values.detach().cpu().numpy()
                        else:
                            hist_values_filtered = hist_values.detach().cpu().numpy()
                        self.hist_values[:, :, c] = hist_values_filtered
                        target_values /= target_values.sum()
                        del feat_vecs_curr_norm, hist_values  # trying to save some memory
                        torch.cuda.empty_cache()
                        hist_values_for_loss = hist_values_filtered / (np.expand_dims(hist_values_filtered.sum(1), 1) + self.epsilon)
                        aaa=1
                        if self.relative_weight>1e6:
                            import os

                            save_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/results/' + self._loss_name + '/class_{}/'.format(c)
                            if not os.path.isdir(save_dir):
                                os.makedirs(save_dir)

                            for f in range(0, 256, 1):
                                save_path = os.path.join(save_dir, 'dir_{}.png'.format(f))
                                plt.plot(hist_values_for_loss[f]);
                                plt.plot(target_values[0].cpu());
                                plt.title(
                                    'c={}, dir={}, kurt = {:.3f}, mom2 = {}'.format(c, f, kurtosis[f], var_curr_t[f]));
                                plt.savefig(save_path);
                                plt.close();

                            for f in range(0, 10000, 100):
                                save_path = os.path.join(save_dir, 'dir_{}.png'.format(f))
                                plt.plot(hist_values_for_loss[f]);
                                plt.plot(target_values[0].cpu());
                                plt.title(
                                    'c={}, dir={}, kurt = {:.3f}, mom2 = {}'.format(c, f, kurtosis[f], var_curr_t[f]));
                                plt.savefig(save_path);
                                plt.close();
                            aaa=1

        # weighing each class according to sqrt(sample_num)
        samples_num_all_in_loss_curr = self.samples_num_all_in_loss - samples_num_all_in_loss_pre
        weight_per_class = torch.from_numpy(samples_num_all_in_loss_curr).float().to('cuda').sqrt()
        weight_per_class /= (weight_per_class.sum() + 1e-15)
        loss_kurtosis = torch.sum(weight_per_class * loss_kurtosis_vect)
        loss_moment2 = torch.sum(weight_per_class * loss_moment2_vect)
        loss_hist = torch.sum(weight_per_class * loss_hist_vect)

        print(self._loss_name + ' = {:.3f}, loss_kurtosis = {:.3f}, loss_moment2 = {:.3f}, active = {}, weight = {:.3f}, '.
              format(loss_hist, loss_kurtosis, loss_moment2, active_classes_num, self.relative_weight))

        loss_kurtosis *= self.relative_weight
        loss_moment2 *= self.relative_weight
        loss_hist *= self.relative_weight

        self.iters_since_init += 1
        self.iters_since_epoch_init += 1
        return self.loss_weight*loss_hist

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
