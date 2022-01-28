from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os

if 0:
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/alta/pspnet_r18-d8_512x1024_80k_exp3.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/pspnet_r18-d8_512x1024_80k_exp3/iter_1500.pth'
else:
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/alta/segformer_mit-b0_height100_exp2.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_height100_exp2/iter_2000.pth'

results_path = checkpoint_file.split('.')[0]
if not os.path.isdir(results_path):
    os.mkdir(results_path)

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = '/home/airsim/repos/open-mmlab/mmsegmentation/data/Alta/img_dir/train/DJI_0060.JPG'  #or img = mmcv.imread(img), which will only load it once
img = '/home/airsim/repos/open-mmlab/mmsegmentation/data/Alta/img_dir/val/DJI_0272.JPG'  #or img = mmcv.imread(img), which will only load it once

out_file = os.path.join(results_path, os.path.split(img)[-1])
result = inference_segmentor(model, img)
# visualize the results in a new window
# model.show_result(img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
model.show_result(img, result, out_file=out_file, opacity=1)
aaa=1

# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html