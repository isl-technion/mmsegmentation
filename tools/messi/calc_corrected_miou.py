import numpy as np
import mmcv

# results_file = '/media/omek/Alta/experiments/arabella_test_post_sampling5/20220829_151550_noB_equal/train_Agamim_All_val_IrYamim_Kikar/noB/deeplabv3plus_r18-d8/equal/trial_1/test_cfg_2/eval_single_scale_20220906_113906.pkl'
results_file = '/media/omek/Alta/experiments/arabella_test_post_sampling5/20220819_092331_equal/train_Agamim_All_val_IrYamim_Kikar/all/bisenetv1_r18-d32/equal/eval_avg_test_cfg_1.pkl'

results = mmcv.load(results_file)

mIoU = 0
count = 0
for k, v in results['metric'].items():
    if k.startswith('IoU.'):
        if 'background' in k:
            continue
        if 'building' in k:
            continue
        if np.isnan(v):
            val = 0
        else:
            val = v
        count += 1
        mIoU += val

print('count = {}, mIoU = {}'.format(count, mIoU/count))

mAcc = 0
count = 0
for k, v in results['metric'].items():
    if k.startswith('Acc.'):
        if 'background' in k:
            continue
        if 'building' in k:
            continue
        if np.isnan(v):
            val = 0
        else:
            val = v
        count += 1
        mAcc += val

print('count = {}, mAcc = {}'.format(count, mAcc/count))

aaa=1