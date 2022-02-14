from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os

if 0:  # Segformer - PathA_70
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_exp1/segformer_mit-b0_pathA_pathA_exp1.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_exp1/iter_10000.pth'
elif 0:  # Segformer - PathA
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes/iter_2000.pth'
elif 0:  # Segformer - PathA, reweighted (also fixed DJI_149)
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1/iter_10000.pth'
else:  # Segformer - PathA->PathB, reweighted (also fixed DJI_149)
    config_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1/segformer_mit-b0_pathA_pathA_loadfrom_cityscapes_reweighted1.py'
    checkpoint_file = '/home/airsim/repos/open-mmlab/mmsegmentation/results/alta/segformer_mit-b0_pathA_pathB_loadfrom_cityscapes_reweighted1/iter_20000.pth'


# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0001'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0005'
images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Descend/100_0038'
# images_path = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Path/B/30'
images_list = os.listdir(images_path)
images_list.sort()

results_path = os.path.join(checkpoint_file.split('.')[0], os.path.split(images_path)[-1])
interval = 3
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
    result = inference_segmentor(model, imgname_full)
    # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    model.show_result(imgname_full, result, out_file=out_file, opacity=1)
    aaa=1

# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html