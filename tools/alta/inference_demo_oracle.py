from mmseg.apis.alta.inference_alta import inference_segmentor, init_segmentor
from mmseg.datasets.alta import AltaDataset
import mmcv
import os
import numpy as np

# Define the model
model_name = 'segformer_mit-b3'
config_file = '/media/omek/Alta/experiments/arabella_test_annot_17082022/20220924_093847_all_sqrt/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/sqrt/trial_1/config.py'
checkpoint_file = '/media/omek/Alta/experiments/arabella_test_annot_17082022/20220924_093847_all_sqrt/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/sqrt/trial_1/epoch_320.pth'

# Build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Choose the Descend scenario
images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0001'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0005'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0038'

# Read images list
images_list = os.listdir(images_path)
images_list.sort()
interval = 1

# Define output folder
results_path = os.path.join('/home/airsim/repos/open-mmlab/mmsegmentation/results/barakp/', os.path.split(images_path)[-1])
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# Run inference on every image
for imgname in images_list[::interval]:
    imgname_full = os.path.join(images_path, imgname)

    result = inference_segmentor(model, imgname_full, return_scores=True)
    out_file = os.path.join(results_path, 'Segmentation', os.path.split(imgname_full)[-1])
    model.show_result(imgname_full, result[0], out_file=out_file, opacity=1)

    scores_map = result[1][0].cpu().numpy()
    oracle_map = np.sum(np.expand_dims(AltaDataset.class_scores, axis=(1, 2)) * scores_map, axis=0)
    out_file_score = os.path.join(results_path, 'Oracle', os.path.split(imgname_full)[-1])
    mmcv.imwrite(oracle_map * 255, out_file_score)

    aaa=1
