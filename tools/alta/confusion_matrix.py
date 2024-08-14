# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv import Config, DictAction

from mmseg.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='rainbow',  # 'winter',
        help='theme of the matrix color map')
    parser.add_argument(
        '--title',
        default='Normalized Confusion Matrix (%)',
        help='title of the matrix color map')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset, results):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of segmentation results in each image.
    """
    n = len(dataset.CLASSES)
    confusion_matrix = np.zeros(shape=[n, n])
    assert len(dataset) == len(results)
    prog_bar = mmcv.ProgressBar(len(results))

    dataset_ind = 0
    idx_inside_dataset = -1
    for idx, per_img_res in enumerate(results):
        idx_inside_dataset += 1

        res_segm = per_img_res#.astype(np.int64)
        try:  # original code
            gt_segm = dataset.get_gt_seg_map_by_idx(idx)
        except:  # in case of concat dataset
            gt_segm = dataset.datasets[dataset_ind].get_gt_seg_map_by_idx(idx_inside_dataset)
            if idx_inside_dataset >= len(dataset.datasets[dataset_ind])-1:
                idx_inside_dataset = -1
                dataset_ind += 1

        inds = n * gt_segm + res_segm
        inds = inds.flatten()
        mat = np.bincount(inds, minlength=n**2).reshape(n, n)
        confusion_matrix += mat
        prog_bar.update()
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix (%)',
                          color_theme='winter'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(2 * num_classes, 2 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    cbar = plt.colorbar(mappable=im, ax=ax)
    cbar.ax.tick_params(labelsize=30)

    title_font = {'weight': 'bold', 'size': 35}  # 12}
    # ax.set_title(title, fontdict=title_font)
    label_font = {'weight': 'bold', 'size': 35}  # 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=28)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    ax.tick_params(axis='y', labelsize=28)
    plt.setp(ax.get_yticklabels())

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}'.format(
                    round(confusion_matrix[i, j], 2
                          ) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='black',
                size=27)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix1.png'), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = mmcv.load(args.prediction_path)

    main_path = '/media/omek/Alta/experiments/arabella_test_annot_17082022/20220924_093847_all_sqrt/' \
                'train_Agamim_All_val_IrYamim_Kikar/all/segformer_mit-b3/sqrt/trial_1/'

    results += mmcv.load(os.path.join(main_path, 'conf_matrix/result_30_84_166.pkl'))
    results += mmcv.load(os.path.join(main_path, 'conf_matrix/result_50.pkl'))
    results += mmcv.load(os.path.join(main_path, 'conf_matrix/result_70.pkl'))
    results += mmcv.load(os.path.join(main_path, 'conf_matrix/result_100.pkl'))
    results += mmcv.load(os.path.join(main_path, 'conf_matrix/result_pilot.pkl'))

    # results += mmcv.load('/media/omek/Alta/experiments/arabella_test_post_sampling5/irrelevant/temp/result_30_84_166.pkl')
    # results += mmcv.load('/media/omek/Alta/experiments/arabella_test_post_sampling5/irrelevant/temp/result_50.pkl')
    # results += mmcv.load('/media/omek/Alta/experiments/arabella_test_post_sampling5/irrelevant/temp/result_70.pkl')
    # results += mmcv.load('/media/omek/Alta/experiments/arabella_test_post_sampling5/irrelevant/temp/result_100.pkl')
    # results += mmcv.load('/media/omek/Alta/experiments/arabella_test_post_sampling5/irrelevant/temp/result_pilot.pkl')

    assert isinstance(results, list)
    if isinstance(results[0], np.ndarray):
        pass
    else:
        raise TypeError('invalid type of prediction results')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)
    confusion_matrix = calculate_confusion_matrix(dataset, results)
    confusion_matrix = confusion_matrix[1:, 1:]  # remove the background class
    labels = dataset.CLASSES[1:]  # remove the background class
    labels[labels.index('transportation terrain')] = 'trans. terr.'
    labels[labels.index('rough terrain')] = 'rough terr.'
    labels[labels.index('soft terrain')] = 'soft terr.'
    labels[labels.index('walking terrain')] = 'walking terr.'
    plot_confusion_matrix(
        confusion_matrix,
        labels,
        save_dir=args.save_dir,
        show=args.show,
        title=args.title,
        color_theme=args.color_theme)


if __name__ == '__main__':
    main()
