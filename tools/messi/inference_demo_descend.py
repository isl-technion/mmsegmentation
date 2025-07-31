# from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis.alta.inference_alta import inference_segmentor, init_segmentor
import mmcv
import os
import numpy as np

config_file = '/configs/messi/train_100_val_descends/all/segformer_mit-b3/sqrt/config_for_test.py'
checkpoint_file = '/media/omek/Alta/experiments/arabella_const_height/train_100_val_descends/all/segformer_mit-b3/sqrt/trial_1/epoch_160.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

scenarios_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/'
scenarios_list = os.listdir(scenarios_path)

for scn_name in scenarios_list:
    images_path = os.path.join(scenarios_path, scn_name)

    images_list = os.listdir(images_path)
    images_list.sort()

    results_path = os.path.join(checkpoint_file.split('.')[0], os.path.split(images_path)[-1])
    interval = 1
    return_scores = False
    score_th1 = 0.8
    score_th2 = 0.9
    if 'Descend' not in images_path:
        interval = 1
        results_path = os.path.join(os.path.split(results_path)[0], images_path.split('Agamim/')[-1].replace('/', '_'))

    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    for imgname in images_list[::interval]:
        if not imgname.endswith('.JPG'):
            continue
        imgname_full = os.path.join(images_path, imgname)
        out_file = os.path.join(results_path, os.path.split(imgname_full)[-1])
        result = inference_segmentor(model, imgname_full, return_scores=return_scores)
        # visualize the results in a new window
        # model.show_result(img, result, show=True)
        # or save the visualization results to image files
        # you can change the opacity of the painted segmentation map in (0, 1].
        if return_scores:
            out_file_score = os.path.join(results_path, 'scores_map', os.path.split(imgname_full)[-1])
            model.show_result(imgname_full, result[0], out_file=out_file, opacity=1)
            conf_map = (result[1][0] - score_th1) / (1-score_th1)
            mmcv.imwrite(conf_map * 255, out_file_score)
            img = mmcv.imread(out_file)
            conf_mask = result[1][0] < score_th2
            indices = np.nonzero(conf_mask)
            img[indices[0], indices[1], :] = 0
            out_file_combined = os.path.join(results_path, 'combined_{}'.format(score_th2), os.path.split(imgname_full)[-1])
            mmcv.imwrite(img, out_file_combined)
        else:
            model.show_result(imgname_full, result, out_file=out_file, opacity=1)
        aaa=1
