import torch
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot

model_name = 'segformer_mit-b3'
config_file = '/media/omek/Alta/experiments/arabella_test_annot_17082022/20220924_093847_all_sqrt/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/sqrt/trial_1/config.py'
checkpoint_file = '/media/omek/Alta/experiments/arabella_test_annot_17082022/20220924_093847_all_sqrt/train_Agamim_All_val_IrYamim_Kikar/all/' + model_name + '/sqrt/trial_1/epoch_320.pth'
# config_file = '../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
# checkpoint_file = '../checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cpu')

# test a single image
img = 'demo.png'
if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)
result = inference_model(model, img)

# show the results
# vis_result = show_result_pyplot(model, img, result, show=False)
# plt.imshow(vis_result)