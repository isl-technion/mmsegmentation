import os
import sys
import shutil
import time
from datetime import datetime
from tools.alta.run_train_alta import main as mmseg_train

time_now = str(datetime.now())
dest_dir = os.path.join('/media/omek/Alta/experiments', time_now)
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

trials_per_config = 3
configs_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/configs/mboaz17'
results_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/results/mboaz17'

train_val_spec_list = ['train_Agamim_ABC_val_IrYamim_Kikar']
classes_type_list = ['all']  # 'all' \ 'noB' \ ?
model_type_list = ['segformer_mit-b0']  # 'segformer_mit-b0' \ ...
weighting_method_list = ['equal', 'sqrt']  # 'equal' \ 'sqrt' \ ?

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

                    for trial_ind in range(trials_per_config):
                        work_dir = os.path.join(results_dir, 'curr_run')
                        if os.path.isdir(work_dir):
                            shutil.rmtree(work_dir)
                        os.makedirs(work_dir)

                        sys.argv = [sys.argv[0]]
                        sys.argv.append(config_file_path)
                        sys.argv.append('--work-dir')
                        sys.argv.append(work_dir)

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
