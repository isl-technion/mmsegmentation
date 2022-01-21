from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/alta/pspnet_r18-d8_512x1024_80k_exp3.py'
checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/pspnet_r18-d8_512x1024_80k_exp3/iter_1500.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = '/home/airsim/repos/open-mmlab/mmsegmentation/data/Alta/img_dir/train/DJI_0060.JPG'  #or img = mmcv.imread(img), which will only load it once
img = '/home/airsim/repos/open-mmlab/mmsegmentation/data/Alta/img_dir/val/DJI_0159.JPG'  #or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# visualize the results in a new window
# model.show_result(img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
model.show_result(img, result, out_file='result3.jpg', opacity=1)
aaa=1

# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html