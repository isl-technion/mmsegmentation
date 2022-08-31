import os
import sys
import shutil
import time
import mmcv
import glob
# import xlsxwriter
import csv

# dest_dir = '/media/omek/Alta/experiments/arabella_test_post_sampling5/20220819_092331_equal'
dest_dir = '/media/omek/Alta/experiments/arabella_test_post_sampling5/20220823_094529_sqrt'

trials_per_config = 1
epoch_num = 320

test_ind = 1

train_val_spec_list = ['train_Agamim_All_val_IrYamim_Kikar']
classes_type_list = ['all']  # 'all' \ 'noB' \ ?
model_type_list = ['segformer_mit-b0', 'deeplabv3plus_r50-d8', 'deeplabv3plus_r18-d8', 'segformer_mit-b3', 'bisenetv1_r50-d32', 'bisenetv1_r18-d32']  # Second GPU
weighting_method_list = ['sqrt']  # 'equal' \ 'sqrt' \ ?

for train_val_spec in train_val_spec_list:
    for classes_type in classes_type_list:
        for model_type in model_type_list:
            for weighting_method in weighting_method_list:
                for trial_ind in range(trials_per_config):
                    trial_folder_name = 'trial_{}'.format(trial_ind+1)
                    config_rel_path = os.path.join(train_val_spec, classes_type, model_type, weighting_method, trial_folder_name)
                    work_dir = os.path.join(dest_dir, config_rel_path)
                    config_file_path = os.path.join(work_dir, 'config.py')
                    checkpoint_file_path = os.path.join(work_dir, 'epoch_{}.pth'.format(epoch_num))
                    if not os.path.isfile(config_file_path):
                        print('Missing config file: ' + config_file_path)
                        break
                    if not os.path.isfile(config_file_path):
                        print('Missing checkpoint file: ' + checkpoint_file_path)
                        break

                    test_folder_path = os.path.join(work_dir, 'test_cfg_{}'.format(test_ind+1))
                    if not os.path.isdir(test_folder_path):
                        print('Missing test folder: ' + test_folder_path)
                        break

                    eval_pkl_path = glob.glob(test_folder_path + '/*.pkl')[0]
                    if not os.path.isfile(eval_pkl_path):
                        print('Missing eval file: ' + eval_pkl_path)
                        break
                    results = mmcv.load(eval_pkl_path)
                    if trial_ind == 0:
                        results_avg = results
                    else:
                        keys = results_avg.keys()
                        for k,v in results['metric'].items():
                            results_avg['metric'][k] += v

                for k, v in results['metric'].items():
                    results_avg['metric'][k] /= trials_per_config
                results_avg_path = os.path.join(os.path.split(work_dir)[0], 'eval_avg_test_cfg_{}.json'.format(test_ind+1))
                mmcv.dump(results_avg, results_avg_path, indent=4)
                mmcv.dump(results_avg, results_avg_path.replace('.json', '.pkl'))

                with open(results_avg_path.replace('.json', '.csv'), 'w') as output:
                    writer = csv.writer(output)
                    for key, value in results_avg['metric'].items():
                        writer.writerow([key, value])
