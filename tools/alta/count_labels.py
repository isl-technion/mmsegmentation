import os
import glob
import shutil
import numpy as np
import cv2
from mmseg.datasets.alta import AltaDataset

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
    for scenario_name in scenario_list:
        population_vect = np.zeros(len(AltaDataset.CLASSES), dtype=np.uint64)

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

        for img_name in img_names:
            img = cv2.imread(img_name)
            for ind, color in enumerate(AltaDataset.PALETTE):
                print(color)
                population_vect[ind] += np.sum(np.all(img == color[::-1], axis=2))

        datasets_list_file.write(dataset_name)
        datasets_list_file.write("{}".format(population_vect))
        datasets_list_file.write("\n")

        aaa=1

datasets_list_file.close()
