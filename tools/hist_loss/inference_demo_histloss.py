# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis.alta.inference_alta import inference_segmentor, init_segmentor
import mmcv
import os
import torch
import numpy as np
import pickle

return_scores = True
use_hist_model = True
if 0:
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/multi_enc_dec/segformer_mit-b0_pathA_rew_672_448_HL5_dec/segformer_mit-b0_pathA_rew_672_448_HL5_dec.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/multi_enc_dec/segformer_mit-b0_pathA_rew_672_448_HL5_dec/epoch_495.pth'
elif 0:
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/no_batchnorm/segformer_mit-b0_pathA_rew_672_448_HL5_small_eigs/segformer_mit-b0_pathA_rew_672_448_HL5_small_eigs.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/no_batchnorm/segformer_mit-b0_pathA_rew_672_448_HL5_small_eigs/epoch_30.pth'
elif 1:
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/no_batchnorm/segformer_mit-b0_pathA_rew_672_448_HL5_dec458/segformer_mit-b0_pathA_rew_672_448_HL5_dec458.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/no_batchnorm/segformer_mit-b0_pathA_rew_672_448_HL5_dec458/epoch_130.pth'

hist_model = None
hist_model_path = os.path.join(os.path.split(checkpoint_file)[0], 'hooks', os.path.split(checkpoint_file)[1].split('.')[0]+'.pickle')
if use_hist_model and os.path.isfile(hist_model_path):
    with open(hist_model_path, 'rb') as handle:
        hist_model = pickle.load(handle)

        # for k in [2, 4, 7, 9, 11, 12, 13, 14]:
        #     print(hist_model.var_all[:, k].std())# / hist_model.var_all[:, k].mean())

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0001'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0005'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0038'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Path/A/100'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Path/B/100'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Ir yamim/50'
images_path = '/home/airsim/repos/segmentation_models.pytorch/examples/data/CamVid/train'
# images_path = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Airsim/train'
# images_path = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Kitti/2011_09_26_drive_0001_extract/image_00/data'

images_list = os.listdir(images_path)
images_list.sort()

results_path = os.path.join(checkpoint_file.split('.')[0], os.path.split(images_path)[-1])
interval = 4
score_th1 = 0.75
score_th2 = 0.9
if 'Descend' not in images_path:
    interval = 1
    results_path = os.path.join(os.path.split(results_path)[0], images_path.split('Agamim/')[-1].replace('/', '_'))

if not os.path.isdir(results_path):
    os.makedirs(results_path)

for imgname in images_list[::interval]:
    imgname_full = os.path.join(images_path, imgname)
    # test a single image and show the results
    # img = '/home/airsim/repos/open-mmlab/mmsegmentation/data/Alta/img_dir/train/DJI_0060.JPG'  #or img = mmcv.imread(img), which will only load it once
    # img = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Path/A/70/DJI_0118.JPG'  #or img = mmcv.imread(img), which will only load it once

    out_file = os.path.join(results_path, os.path.split(imgname_full)[-1])
    result = inference_segmentor(model, imgname_full, return_scores=return_scores, hist_model=hist_model)
    # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    if return_scores:
        if hist_model:
            score_th2 = -0.5 * (1.25**2)
        out_file_score = os.path.join(results_path, 'scores_map', os.path.split(imgname_full)[-1])
        model.show_result(imgname_full, result[0], out_file=out_file, opacity=1)
        # conf_map = (result[1][0] - score_th1) / (1-score_th1)
        # mmcv.imwrite(conf_map * 255, out_file_score)
        img = mmcv.imread(out_file)
        conf_mask = torch.max(result[1][0], dim=0)[0].detach().cpu().numpy() < score_th2
        indices = np.nonzero(conf_mask)
        img[indices[0], indices[1], :] = 0
        out_file_combined = os.path.join(results_path, 'combined_{}'.format(score_th2), os.path.split(imgname_full)[-1])
        mmcv.imwrite(img, out_file_combined)
    else:
        model.show_result(imgname_full, result, out_file=out_file, opacity=1)
    aaa=1

# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html