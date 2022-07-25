import os
import sys
import shutil
import time
from tools.alta.run_test_alta import main as mmseg_test

dest_dir = '/media/omek/Alta/experiments/20220717_144803'
# dest_dir = '/media/omek/Alta/experiments/20220718_184306'

trials_per_config = 2
iter_num = 200

cfg_options_list = ['data.test.separate_eval=1', 'data.test.separate_eval=0']

train_val_spec_list = ['train_Agamim_ABC_val_IrYamim_Kikar']
classes_type_list = ['all']  # 'all' \ 'noB' \ ?
# model_type_list = ['segformer_mit-b2', 'segformer_mit-b1', 'segformer_mit-b0', 'deeplabv3plus_r50-d8', 'bisenetv2']  #  'bisenetv2' \ 'segformer_mit-b0\1\2' \ ...
model_type_list = ['segformer_mit-b2', 'segformer_mit-b1', 'segformer_mit-b0', 'bisenetv2']  #  'bisenetv2' \ 'segformer_mit-b0\1\2' \ ...
weighting_method_list = ['equal']  # 'equal' \ 'sqrt' \ ?

for train_val_spec in train_val_spec_list:
    for classes_type in classes_type_list:
        for model_type in model_type_list:
            for weighting_method in weighting_method_list:
                for trial_ind in range(trials_per_config):
                    trial_folder_name = 'trial_{}'.format(trial_ind+1)
                    config_rel_path = os.path.join(train_val_spec, classes_type, model_type, weighting_method, trial_folder_name)
                    work_dir = os.path.join(dest_dir, config_rel_path)
                    config_file_path = os.path.join(work_dir, 'config.py')
                    checkpoint_file_path = os.path.join(work_dir, 'iter_{}.pth'.format(iter_num))
                    if not os.path.isfile(config_file_path):
                        print('Missing config file: ' + config_file_path)
                        continue
                    if not os.path.isfile(config_file_path):
                        print('Missing checkpoint file: ' + checkpoint_file_path)
                        continue

                    for test_ind in range(len(cfg_options_list)):
                        test_folder_path = os.path.join(work_dir, 'test_cfg_{}'.format(test_ind+1))
                        if not os.path.isdir(test_folder_path):
                            os.mkdir(test_folder_path)
                        results_pkl_path = os.path.join(work_dir, 'results.pkl')
                        sys.argv = [sys.argv[0]]
                        sys.argv.append(config_file_path)
                        sys.argv.append(checkpoint_file_path)
                        sys.argv.append('--work-dir')
                        sys.argv.append(test_folder_path)
                        sys.argv.append('--out')
                        sys.argv.append(results_pkl_path)
                        sys.argv.append('--cfg-options')
                        sys.argv.append(cfg_options_list[test_ind])
                        if test_ind > 0:
                            sys.argv.append('--load_pkl')
                            sys.argv.append('1')

                        with open(os.path.join(test_folder_path, 'test_log.txt'), 'w') as f:
                            try:
                                mmseg_test()
                                f.write('Successfully tested ' + checkpoint_file_path + '\n')
                            except Exception as inst:
                                f.write(str(inst) + '\n')
                                f.write('Error while testing ' + checkpoint_file_path + '\n')
                            f.close()

                        time.sleep(3)


# /media/omek/Alta/experiments/20220718_184306/train_Agamim_ABC_val_IrYamim_Kikar/all/deeplabv3plus_r50-d8/sqrt/trial_1/config2.py /media/omek/Alta/experiments/20220718_184306/train_Agamim_ABC_val_IrYamim_Kikar/all/deeplabv3plus_r50-d8/sqrt/trial_1/iter_2000.pth --out /media/omek/Alta/experiments/20220718_184306/train_Agamim_ABC_val_IrYamim_Kikar/all/deeplabv3plus_r50-d8/sqrt/trial_1/res.pkl --cfg-options data.test.separate_eval=0