import sys
from tools.train import main as mmseg_train

config_file_path = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/alta/segformer/segformer_mit-b0_pathA_pathA_672_448_histloss.py'

sys.argv.append(config_file_path)
mmseg_train()