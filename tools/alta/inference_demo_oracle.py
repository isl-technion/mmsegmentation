# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis.alta.inference_alta import inference_segmentor, init_segmentor
from mmseg.datasets.alta import AltaDataset
import mmcv
import os
import numpy as np

if 0:  # PSPnet - PathA
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/pspnet_r18-d8_pathA_pathA/pspnet_r18-d8_pathA_pathA.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/pspnet_r18-d8_pathA_pathA/iter_20000.pth'
elif 0:  # PSPnet - PathA - mislabeled DJI_0149
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/pspnet_r18-d8_pathA_pathA_DJI_0149_mislabeled/pspnet_r18-d8_pathA_pathA_DJI_0149_mislabeled.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/pspnet_r18-d8_pathA_pathA_DJI_0149_mislabeled/iter_20000.pth'
elif 0:  # Segformer - PathA - mislabeled DJI_0149
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes/iter_1000.pth'
elif 0:  # Segformer - PathA, reweighted (also fixed DJI_149)
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1/iter_10000.pth'
elif 0:  # Segformer - PathA->PathB, reweighted (also fixed DJI_149)
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathB_loadfrom_cityscapes_reweighted1/segformer_mit-b0_pathA_pathB_loadfrom_cityscapes_reweighted1.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathB_loadfrom_cityscapes_reweighted1/iter_20000.pth'
elif 0:  # Segformer - PathA, reweighted (also fixed DJI_149), resized to 672*448
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_reweighted1_672_448/segformer_mit-b0_pathA_pathA_reweighted1_672_448.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_reweighted1_672_448/iter_20000.pth'
elif 0:  # Segformer - PathA, resized to 672*448, with histogramm loss (16 dims)
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_672_448_histloss/segformer_mit-b0_pathA_pathA_672_448_histloss.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_672_448_histloss/iter_4000.pth'
elif 0:  # Segformer - PathA, resized to 672*448, with histogramm loss (256 dims)
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_672_448_histloss_1000_feat256/segformer_mit-b0_pathA_pathA_672_448_histloss_1000_feat256.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_672_448_histloss_1000_feat256/epoch_100.pth'
##############################################3
elif 0:  # for_paper - DeepLabV3+
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/mboaz17/paper/deeplabv3_r50-d8_pathA_pathB.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17/paper/deeplabv3_r50-d8_pathA_pathB/iter_40000.pth'
elif 0:  # for_paper - DeepLabV3+ reweighted
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/mboaz17/paper/deeplabv3_r50-d8_pathA_pathB_rew_sqrt.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17/paper/deeplabv3_r50-d8_pathA_pathB_rew_sqrt/iter_40000.pth'
elif 0:  # for_paper - DeepLabV3+ reweighted, 2048x1024, trial experiment
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/mboaz17/paper/deeplabv3_r50-d8_pathA_pathB_rew_sqrt_temp.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17/paper/deeplabv3_r50-d8_pathA_pathB_rew_sqrt_temp/iter_20000.pth'
elif 0:  # for_paper - DeepLabV3+ reweighted, pathABC -> IrYamim+Pilot
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/mboaz17/paper/deeplabv3_r50-d8_train_pathABC_val_IrYamim_Pilot_rew_sqrt.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17/paper/deeplabv3_r50-d8_train_pathABC_val_IrYamim_Pilot_rew_sqrt/iter_40000.pth'
elif 0:  # for_paper - DeepLabV3+ reweighted, pathABC -> IrYamim+Pilot
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17/paper/segformer_mit-b0_train_pathABC_val_IrYamim_Pilot_rew_sqrt_noB/segformer_mit-b0_train_pathABC_val_IrYamim_Pilot_rew_sqrt_noB.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17/paper/segformer_mit-b0_train_pathABC_val_IrYamim_Pilot_rew_sqrt_noB/iter_40000.pth'
elif 1:  # for_paper -
    if 0:
        model_name = 'segformer_mit-b0'
        config_file = '/media/omek/Alta/experiments/remote_test/20220812_133530/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/equal/trial_1/config.py'
        checkpoint_file = '/media/omek/Alta/experiments/remote_test/20220812_133530/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/equal/trial_1/epoch_100.pth'
    elif 0:
        model_name = 'segformer_mit-b3'
        config_file = '/media/omek/Alta/experiments/arabella_test_annot_17082022/20220924_093847_all_sqrt/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/sqrt/trial_1/config.py'
        checkpoint_file = '/media/omek/Alta/experiments/arabella_test_annot_17082022/20220924_093847_all_sqrt/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/sqrt/trial_1/epoch_320.pth'
    elif 1:  # 1440x1080
        config_file = '/media/omek/Alta/experiments/20241211_124626/train_all_heights_val_descends/all/segformer_mit-b3/sqrt/trial_1/config_1440_1080.py'
        checkpoint_file = '/media/omek/Alta/experiments/20241211_124626/train_all_heights_val_descends/all/segformer_mit-b3/sqrt/trial_1/epoch_2.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0001'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0005'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0038'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Path/A/100'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Path/B/100'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Ir yamim/100'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Pilot/Path'
images_list = os.listdir(images_path)
images_list.sort()

results_path = os.path.join(checkpoint_file.split('.')[0], os.path.split(images_path)[-1])
interval = 3
return_scores = True
rescale = False
if 'Descend' not in images_path:
    interval = 1
    results_path = os.path.join(os.path.split(results_path)[0], images_path.split('Agamim/')[-1].replace('/', '_'))

if not os.path.isdir(results_path):
    os.makedirs(results_path)

# Run inference on every image
for imgname in images_list[::interval]:
    imgname_full = os.path.join(images_path, imgname)

    result = inference_segmentor(model, imgname_full, rescale=rescale, return_scores=True)
    out_file = os.path.join(results_path, 'Segmentation', os.path.split(imgname_full)[-1])
    model.show_result(imgname_full, result[0], out_file=out_file, opacity=1)

    scores_map = result[1][0].cpu().numpy()
    oracle_map = np.sum(np.expand_dims(AltaDataset.class_scores, axis=(1, 2)) * scores_map, axis=0)
    out_file_score = os.path.join(results_path, 'Oracle', os.path.split(imgname_full)[-1])
    mmcv.imwrite(oracle_map * 255, out_file_score)

    aaa=1
