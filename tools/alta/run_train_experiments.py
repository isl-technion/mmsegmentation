import os
import sys
import shutil
import time
from tools.train import main as mmseg_train

dest_dir0 = '/media/omek/Alta/experiments/'
project_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/'

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
dest_dir = os.path.join(dest_dir0, timestamp)
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

shutil.copyfile(__file__, os.path.join(dest_dir, os.path.split(__file__)[1]))

trials_per_config = 1
configs_dir = os.path.join(project_dir, 'configs/messi')
results_dir = os.path.join(project_dir, 'results/messi')

train_val_spec_list = ['train_Agamim_All_val_IrYamim_Kikar']
classes_type_list = ['all', 'noB']
model_type_list = ['segformer_mit-b0', 'segformer_mit-b3', 'bisenetv1_r50-d32', 'bisenetv1_r18-d32',
                   'deeplabv3plus_r50-d8', 'deeplabv3plus_r18-d8']
weighting_method_list = ['sqrt', 'equal', 'prop']

for train_val_spec in train_val_spec_list:
    for classes_type in classes_type_list:
        for model_type in model_type_list:
            for weighting_method in weighting_method_list:
                config_rel_path = os.path.join(train_val_spec, classes_type, model_type, weighting_method)
                config_file_path = os.path.join(configs_dir, config_rel_path, 'config.py')
                if not os.path.isfile(config_file_path):
                    print('Missing config file: ' + config_file_path)
                    continue

                for trial_ind in range(trials_per_config):
                    work_dir = os.path.join(results_dir, __file__.split('.')[0].split('/')[-1])
                    if os.path.isdir(work_dir):
                        shutil.rmtree(work_dir)
                    os.makedirs(work_dir)

                    sys.argv = [sys.argv[0]]
                    sys.argv.append(config_file_path)
                    sys.argv.append('--work-dir')
                    sys.argv.append(work_dir)
                    sys.argv.append('--cfg-options')
                    sys.argv.append('data.val.separate_eval=0')

                    with open(os.path.join(work_dir, 'experiment_log.txt'), 'w') as f:
                        try:
                            mmseg_train()
                            f.write('Successfully trained ' + config_file_path + '\n')
                        except Exception as inst:
                            f.write(str(inst) + '\n')
                            f.write('Error while training ' + config_file_path + '\n')
                        f.close()

                    # move experiment content to a server and delete original directory
                    if os.path.isfile(os.path.join(work_dir, 'latest.pth')):
                        os.remove(os.path.join(work_dir, 'latest.pth'))
                    dest_dir_curr = os.path.join(dest_dir, config_rel_path, 'trial_{}'.format(trial_ind+1))
                    shutil.move(work_dir, dest_dir_curr)
                    time.sleep(10)
