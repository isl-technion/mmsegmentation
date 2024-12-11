import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

base_path = '/media/omek/Alta/experiments/arabella_const_height'
desecend_f = True
# setup_list = os.listdir(base_path)
setup_list = [
    'train_all_heights_val_descends',
    'train_50_70_100_val_descends',
    'train_70_100_val_descends',
    'train_100_val_descends',
    'train_30_50_70_val_descends',
    'train_50_70_val_descends',
    'train_30_50_val_descends',
    'train_30_val_descends',
]
if desecend_f:
    setup_ind = [3, 2, 1, 0]
    plot_name = 'Accuracy_ascend.png'
else:
    setup_ind = [7, 6, 4, 0]
    plot_name = 'Accuracy_descend.png'

plot_list = []
for setup in setup_list:
    test_path = os.path.join(base_path, setup, 'all/segformer_mit-b3/sqrt/trial_1/test_results')
    if not os.path.isdir(test_path):
        print(test_path + ' deos not exist')
        continue
    scenarios_list = os.listdir(test_path)

    altitudes_grid = np.linspace(0, 150, 151)
    accuracy_cum = np.zeros_like(altitudes_grid)
    altitudes_occupancy = np.zeros_like(altitudes_grid)
    for scn_name in scenarios_list:
        # accuracy_vect = np.zeros_like(altitudes_grid)

        scn_path = os.path.join(test_path, scn_name)
        if not os.path.isdir(scn_path):
            continue

        results_file = os.path.join(test_path, scn_name, 'results_scn.pkl')
        img_indices, altitudes, acc_vect, mIoU_vect, mAcc_vect = mmcv.load(results_file)

        # for alt, acc in zip(altitudes, acc_vect):
        #     index = int(np.round(alt))
        #     altitudes_occupancy[index] += 1
        #     accuracy_vect[index] += acc
        #
        # accuracy_quantized = accuracy_vect / (altitudes_occupancy + 1e-12)

        f = interpolate.interp1d(altitudes, acc_vect, bounds_error=False, fill_value=0)
        accuracy_interpolated = f(altitudes_grid)
        alt_min = altitudes.min()
        alt_max = altitudes.max()
        altitudes_occupancy[np.logical_and(altitudes_grid >= alt_min, altitudes_grid <= alt_max)] += 1
        accuracy_cum += accuracy_interpolated

    accuracy_cum /= (altitudes_occupancy + 1e-12)
    plt.plot(altitudes_grid[altitudes_occupancy > 0], accuracy_cum[altitudes_occupancy > 0], '.')
    # accuracy_quantized = accuracy_vect / (altitudes_occupancy + 1e-12)
    # plt.plot(altitudes_grid[altitudes_occupancy > 0], accuracy_quantized[altitudes_occupancy > 0], '.')
    plt.xlabel('Altitude')
    plt.ylabel('Accuracy')
    plt.title(setup)
    plt.ylim([0, 1])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.yticks(np.arange(0, 1.05, 0.05))

    plt.savefig(os.path.join(base_path, setup, plot_name))
    plt.close()

    plot_list.append([setup, altitudes_grid[altitudes_occupancy > 0], accuracy_cum[altitudes_occupancy > 0]])

# Combined plot
labels = []
setup_relevant = setup_list
if desecend_f:
    legend_list = ['Horizontal trajectory training altitudes = {100}',
                   'Horizontal trajectory training altitudes = {70, 100}',
                   'Horizontal trajectory training altitudes = {50, 70, 100}',
                   'Horizontal trajectory training altitudes = {30, 50, 70, 100}']
else:
    legend_list = ['Horizontal trajectory training altitudes = {30}',
                   'Horizontal trajectory training altitudes = {30, 50}',
                   'Horizontal trajectory training altitudes = {30, 50, 70}',
                   'Horizontal trajectory training altitudes = {30, 50, 70, 100}']

legend_indices = []
for ind, plot_curr in enumerate([plot_list[i] for i in setup_ind]):
    plt.plot(plot_curr[1], plot_curr[2], '.')
    plt.xlabel('Vertical trajectory test altitude[m]')
    plt.ylabel('Accuracy')
    # plt.title('Influence of training altitude on accuracy based altitude test')
    plt.ylim([0, 1])
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.gca().invert_xaxis()
plt.yticks(np.arange(0, 1.05, 0.05))
plt.legend(legend_list)

plt.savefig(os.path.join(base_path, plot_name))
plt.close()
aaa=1
