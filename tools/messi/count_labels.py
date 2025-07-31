import os
import glob
import shutil
import numpy as np
import cv2
from mmseg.datasets.alta import MessiDataset

datasets_list_file = open("./datasets_list.txt", "w+")

dir_agamim_descend = '/media/isl12/Alta/V7_Exp_25_1_21_annot/Agamim/Descend/'
dir_agamim_path_A = '/media/isl12/Alta/V7_Exp_25_1_21_annot/Agamim/Path/A/'
dir_agamim_path_B = '/media/isl12/Alta/V7_Exp_25_1_21_annot/Agamim/Path/B/'
dir_agamim_path_C = '/media/isl12/Alta/V7_Exp_25_1_21_annot/Agamim/Path/C/'
dir_ir_yamim = '/media/isl12/Alta/V7_Exp_25_1_21_annot/Ir yamim/'
dir_pilot = '/media/isl12/Alta/V7_Exp_25_1_21_annot/Pilot/'
dir_list = [dir_agamim_path_A, dir_agamim_path_B, dir_agamim_path_C, dir_ir_yamim, dir_agamim_descend, dir_pilot]

for dir_name in dir_list:
    scenario_list = [scn for scn in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, scn))]
    population_vect_per_dir = np.zeros(len(MessiDataset.CLASSES)+1, dtype=np.uint64)
    for scenario_name in scenario_list:
        print(scenario_name)
        population_vect = np.zeros(len(MessiDataset.CLASSES)+1, dtype=np.uint64)

        if dir_name == dir_agamim_path_A:
            dataset_name = 'AgamimPathA_' + scenario_name
        elif dir_name == dir_agamim_path_B:
            dataset_name = 'AgamimPathB_' + scenario_name
        elif dir_name == dir_agamim_path_C:
            dataset_name = 'AgamimPathC_' + scenario_name
        elif dir_name == dir_ir_yamim:
            dataset_name = 'IrYamim_' + scenario_name
        elif dir_name == dir_agamim_descend:
            dataset_name = 'AgamimDescend_' + scenario_name
        elif dir_name == dir_pilot:
            dataset_name = 'Pilot' + scenario_name

        # Create dataset
        scenario_full_path = os.path.join(dir_name, scenario_name)
        img_names = glob.glob(scenario_full_path + '/*.png')

        if scenario_name in ['100_0001', '100_0035', '100_0036']:  # skipped scenarios
            continue
        if 'AgamimDescend_' in dataset_name:  # skipped images of Descent scenarios
            interval = 5
        else:
            interval = 1
        for img_name in img_names[::interval]:
            print(img_name)
            img = cv2.imread(img_name)
            for ind, color in enumerate(MessiDataset.PALETTE + [[0, 0, 0]]):
                population_vect[ind] += np.sum(np.all(img == color[::-1], axis=2))

        datasets_list_file.write(dataset_name + '\n')
        for val in population_vect:
            datasets_list_file.write("{}, ".format(val))
        datasets_list_file.write('\n\n')

        population_vect_per_dir += population_vect

    datasets_list_file.write(dir_name + '\n')
    for val in population_vect_per_dir:
        datasets_list_file.write("{}, ".format(val))
    datasets_list_file.write('\n\n')


datasets_list_file.close()
