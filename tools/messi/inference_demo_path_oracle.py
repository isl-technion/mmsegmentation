# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis.alta.inference_alta import inference_segmentor, init_segmentor
from mmseg.datasets.alta import MessiDataset
import mmcv
import os
import numpy as np

# exp_names = [
#     'train_A_B_C_HS_test_IrY_100',
#     'train_A_B_IrY_HS_test_C_100',
#     'train_A_C_IrY_HS_test_B_100',
#     'train_B_C_IrY_HS_test_A_100',
#     'train_A_B_C_HS_test_IrY',
#     'train_A_B_IrY_HS_test_C',
#     'train_A_C_IrY_HS_test_B',
#     'train_B_C_IrY_HS_test_A',
# ]

# # exp_names = [
# scenarios_names = [
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Agamim/Path/C/30', 'Agamim/Path/C/50', 'Agamim/Path/C/70', 'Agamim/Path/C/100'],
#                     ['Agamim/Path/B/30', 'Agamim/Path/B/50', 'Agamim/Path/B/70', 'Agamim/Path/B/100'],
#                     ['Agamim/Path/A/30', 'Agamim/Path/A/50', 'Agamim/Path/A/70', 'Agamim/Path/A/100'],
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Agamim/Path/C/30', 'Agamim/Path/C/50', 'Agamim/Path/C/70', 'Agamim/Path/C/100'],
#                     ['Agamim/Path/B/30', 'Agamim/Path/B/50', 'Agamim/Path/B/70', 'Agamim/Path/B/100'],
#                     ['Agamim/Path/A/30', 'Agamim/Path/A/50', 'Agamim/Path/A/70', 'Agamim/Path/A/100']
#                 ]

# exp_names = [
#     'train_A_B_C_HS_test_IrY_30',
#     'train_A_B_C_HS_test_IrY_50',
#     'train_A_B_C_HS_test_IrY_50_30',
#     'train_A_B_C_HS_test_IrY_70',
#     'train_A_B_C_HS_test_IrY_70_50_30',
#     'train_A_B_C_HS_test_IrY_100_70',
#     'train_A_B_C_HS_test_IrY_100_70_50',
# ]

# exp_names = ['train_A_B_C_HS_test_IrY_100_50',
#              'train_A_B_C_HS_test_IrY_100_30',
#              'train_A_B_C_HS_test_IrY_70_50',
#              'train_A_B_C_HS_test_IrY_70_30']

# scenarios_names = [
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100']
#                 ]

# exp_names = ['train_A_B_C_HS_test_IrY_30']
#
# scenarios_names = [
#                     ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100']
#                 ]
exp_names = ['train_A_B_C_HS_test_IrY_100_50_30',
             'train_A_B_C_HS_test_IrY_100_70_30']

scenarios_names = [
                    ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100'],
                    ['Ir yamim/30', 'Ir yamim/50', 'Ir yamim/70', 'Ir yamim/100']
                ]

#     'train_30_50_70_val_descends',
#     'train_30_50_val_descends',
#     'train_30_val_descends',
#     'train_50_70_100_val_descends',
#     'train_50_70_val_descends',
#     'train_70_100_val_descends',
#     'train_100_val_descends',
#     'train_all_heights_val_descends',
# ]

for exp_name in exp_names:
    config_file = '/home/airsim/projects/datasets/Alta/experiments/path_resized_1080_1440_4/' + exp_name + '/all/segformer_mit-b3/sqrt/trial_1/config_1440_1080.py'
    checkpoint_file = '/home/airsim/projects/datasets/Alta/experiments/path_resized_1080_1440_4/' + exp_name + '/all/segformer_mit-b3/sqrt/trial_1/epoch_160.pth'
    # config_file = '/media/omek/Alta/experiments/resized_720_1280/' + exp_name + '/all/segformer_mit-b3/sqrt/trial_1/config_1280_720.py'
    # checkpoint_file = '/media/omek/Alta/experiments/resized_720_1280/' + exp_name + '/all/segformer_mit-b3/sqrt/trial_1/epoch_160.pth'

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # scenario_names = [
    #     '100_0001',
    #     '100_0002',
    #     '100_0003',
    #     '100_0004',
    #     '100_0005',
    #     '100_0006',
    #     '100_0031',
    #     '100_0035',
    #     '100_0036',
    #     '100_0037',
    #     '100_0038',
    #     '100_0040',
    #     '100_0041',
    #     '100_0042',
    #     '100_0043',
    # ]
    exp_index = exp_names.index(exp_name)
    scenario_names = scenarios_names[exp_index]
    for scenario_name in scenario_names:

        # images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/' + scenario_name
        images_path = '/media/isl12/Alta/V7_Exp_25_1_21/' + scenario_name
        gt_path  = '/media/isl12/Alta/V7_Exp_25_1_21_annot/'+ scenario_name
        images_list = os.listdir(images_path)
        images_list.sort()

        results_path = os.path.join(checkpoint_file.split('.')[0], os.path.split(images_path)[-1])
        interval = 1
        return_scores = True
        rescale = False

        if not os.path.isdir(results_path):
            os.makedirs(results_path)

        # Run inference on every image
        for imgname in images_list[::interval]:
            if not imgname.endswith('.JPG'):
                continue
            imgname_full = os.path.join(images_path, imgname)
            gt_name = imgname[:-3]
            gtname_full = os.path.join(gt_path, gt_name+'png')

            img = mmcv.imread(imgname_full)
            resized_img = mmcv.imresize(img, (1440, 1080), interpolation='bilinear')
            # resized_img = mmcv.imresize(img, (1280, 720), interpolation='bilinear')
            out_file = os.path.join(results_path, 'Orig_resized', os.path.split(imgname_full)[-1])
            mmcv.imwrite(resized_img, out_file)

            gt_img = mmcv.imread(gtname_full)
            resized_gt_img = mmcv.imresize(gt_img, (1440, 1080), interpolation='nearest')
            # resized_img = mmcv.imresize(img, (1280, 720), interpolation='bilinear')
            out_file = os.path.join(results_path, 'GT_resized', os.path.split(gtname_full)[-1])
            mmcv.imwrite(resized_gt_img, out_file)

            result = inference_segmentor(model, imgname_full, rescale=rescale, return_scores=True)
            out_file = os.path.join(results_path, 'Segmentation', os.path.split(imgname_full)[-1])
            model.show_result(imgname_full, result[0], out_file=out_file, opacity=1)

            scores_map = result[1][0].cpu().numpy()
            oracle_map = np.sum(np.expand_dims(MessiDataset.class_scores, axis=(1, 2)) * scores_map, axis=0)
            out_file_score = os.path.join(results_path, 'Oracle', os.path.split(imgname_full)[-1])
            mmcv.imwrite(oracle_map * 255, out_file_score)
