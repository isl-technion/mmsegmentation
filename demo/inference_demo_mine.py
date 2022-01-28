from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import time

if 0:
    # PSPnet
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
elif 0:
    # BiSeNetV2
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/checkpoints/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth'
else:
    # Segformer
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/checkpoints/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/airsim/repos/open-mmlab/mmsegmentation/tests/data/color.jpg'  # 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
for i in range(2):
    start = time.time()
    result = inference_segmentor(model, img)
    end = time.time()
    print(end-start)

# visualize the results in a new window
model.show_result(img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
model.show_result(img, result, out_file='result1.jpg', opacity=1)
aaa=1

# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html