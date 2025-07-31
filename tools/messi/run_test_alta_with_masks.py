# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes

import matplotlib.pyplot as plt
import numpy as np
import csv

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
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
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--load_pkl', type=int, default=0)  # <messi>

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    args.load_pkl = True  # <messi>
    if args.work_dir is None:  # <messi>
        args.work_dir = osp.join(osp.split(args.checkpoint)[0], 'test_results')
    if args.out is None:
        args.out = osp.join(osp.split(args.checkpoint)[0], 'test_results', 'results.pkl')
    args.eval = 'mIoU'

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    if not args.load_pkl or not os.path.isfile(args.out):  # <messi>
        if not distributed:
            warnings.warn(
                'SyncBN is only supported with DDP. To be compatible with DP, '
                'we convert SyncBN to BN. Please use dist_train.sh which can '
                'avoid this error.')
            if not torch.cuda.is_available():
                assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                    'Please use MMCV >= 1.4.4 for CPU training!'
            model = revert_sync_batchnorm(model)
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
            results = single_gpu_test(
                model,
                data_loader,
                args.show,
                args.show_dir,
                False,
                args.opacity,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)
        else:
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)
            results = multi_gpu_test(
                model,
                data_loader,
                args.tmpdir,
                args.gpu_collect,
                False,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:  # <messi>
            if args.load_pkl and os.path.isfile(args.out):
                results = mmcv.load(args.out)
            else:
                warnings.warn(
                    'The behavior of ``args.out`` has been changed since MMSeg '
                    'v0.16, the pickled outputs could be seg map as type of '
                    'np.array, pre-eval results or file paths for '
                    '``dataset.format_results()``.')
                print(f'\nwriting results to {args.out}')
                mmcv.dump(results, args.out)
        if args.eval:

            ind_end = 0
            for ds in dataset.datasets:
                scn_name = os.path.basename(ds.img_dir)
                os.makedirs(os.path.join(args.work_dir, scn_name), exist_ok=True)
                images_num_curr = len(ds.img_infos)
                ind_start = ind_end
                ind_end = ind_start + images_num_curr
                img_indices = np.zeros(images_num_curr)
                altitudes = np.zeros(images_num_curr)
                acc_vect = np.zeros(images_num_curr)
                mIoU_vect = np.zeros(images_num_curr)
                mAcc_vect = np.zeros(images_num_curr)
                for res_ind, result in enumerate(results[ind_start:ind_end]):
                    print(result)
                    total_area_intersect = result[0]
                    total_area_union = result[1]
                    total_area_pred_label = result[2]
                    total_area_label = result[3]
                    all_acc = total_area_intersect.sum() / total_area_label.sum()
                    iou = total_area_intersect / (total_area_union + 1e-15)
                    acc = total_area_intersect / (total_area_label + 1e-15)
                    mIoU = iou[1:].mean()  # without background
                    mAcc = acc[1:].mean()  # without background
                    img_ind = int(ds.img_infos[res_ind]['filename'].split('.')[0].split('_')[-1])
                    img_indices[res_ind] = img_ind
                    acc_vect[res_ind] = all_acc
                    mIoU_vect[res_ind] = mIoU
                    mAcc_vect[res_ind] = mAcc

                with open(os.path.join('/media/isl12/Alta/MESSI dataset full/Train and Val/6DOF/Agamim/Descend/',
                                       scn_name, '6DOF.csv')) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for line_count, row in enumerate(csv_reader):
                        if line_count > 0:
                            altitudes[line_count-1] = float(row[-1])

                mmcv.dump((img_indices, altitudes, acc_vect, mIoU_vect, mAcc_vect), os.path.join(args.work_dir, scn_name, 'results_scn.pkl'))

                plt.plot(altitudes, acc_vect, '.')
                plt.xlabel('Altitude')
                plt.ylabel('Accuracy')
                plt.ylim([0,1])
                plt.savefig(os.path.join(args.work_dir, scn_name, 'Accuracy.png'))
                plt.close()
                plt.plot(altitudes, mIoU_vect, '.')
                plt.xlabel('Altitude')
                plt.ylabel('mean IoU')
                plt.ylim([0,1])
                plt.savefig(os.path.join(args.work_dir, scn_name, 'mIoU.png'))
                plt.close()
                plt.plot(altitudes, mAcc_vect, '.')
                plt.xlabel('Altitude')
                plt.ylabel('mean Accuracy')
                plt.ylim([0,1])
                plt.savefig(os.path.join(args.work_dir, scn_name, 'mAcc.png'))
                plt.close()


if __name__ == '__main__':
    main()
