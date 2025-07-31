# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import LoadAnnotations

import numpy as np
import mmcv

class LoadAnnotationsWithMask(LoadAnnotations):

    def __call__(self, results):
        results = super(LoadAnnotationsWithMask, self).__call__(results)
        scenario_name = osp.split(results['seg_prefix'])[-1]
        mask_name = results['ann_info']['seg_map'].replace('.png', '_mask.png')
        mask_path = osp.join('/media/isl12/Alta/Agamim_Descend_masks', scenario_name, mask_name)
        mask = mmcv.imread(mask_path)[..., 0]
        results['gt_semantic_seg'][mask == 0] = 255  # Ignore index -> no calculation is made
        return results

@DATASETS.register_module()
class MessiDataset(CustomDataset):
    """Messi dataset.
    """

    CLASSES = [
        'background',  # 0
        'bicycle',  # 1
        'building',  # 2
        'fence',  # 3
        'other objects',  # 4
        'person',  # 5
        'pole',  # 6
        'rough terrain',  # 7
        'shed',  # 8
        'soft terrain',  # 9
        'stairs',  # 10
        'transportation terrain',  # 11
        'vegetation',  # 12
        'vehicle',  # 13
        'walking terrain',  # 14
        'water',  # 15
    ]
    PALETTE = [
        [0, 0, 0],  # 0
        [255, 50, 50],  # 1
        [255, 127, 50],  # 2
        [255, 204, 50],  # 3
        [229, 255, 50],  # 4
        [153, 255, 50],  # 5
        [76, 255, 50],  # 6
        [50, 255, 101],  # 7
        [50, 255, 178],  # 8
        [50, 255, 255],  # 9
        [50, 178, 255],  # 10
        [50, 101, 255],  # 11
        [76, 50, 255],  # 12
        [153, 50, 255],  # 13
        [229, 50, 255],  # 14
        [255, 50, 204],  # 15
    ]

    class_scores = [
        0,  # 'background',  # 0
        0,  # 'bicycle',  # 1
        0,  # 'building',  # 2
        0,  # 'fence',  # 3
        0,  # 'other objects',  # 4
        0,  # 'person',  # 5
        0,  # 'pole',  # 6
        0.3,  # 'rough terrain',  # 7
        0,  # 'shed',  # 8
        1,  # 'soft terrain',  # 9
        0,  # 'stairs',  # 10
        0,  # 'transportation terrain',  # 11
        0,  # 'vegetation',  # 12
        0,  # 'vehicle',  # 13
        0.8,  # 'walking terrain',  # 14
        0,  # 'water',  # 15
    ]

    def __init__(self, use_mask=None, **kwargs):
        super(MessiDataset, self).__init__(
            img_suffix='.JPG',
            seg_map_suffix='.png',
            # reduce_zero_label=True,  # False  #remove the bkg class
            # ignore_index=1,
            classes=self.CLASSES[kwargs['reduce_zero_label']:],
            palette=self.PALETTE[kwargs['reduce_zero_label']:],
            **kwargs)
        # self.label_map = {0: 0, 1: 4, 2: 2, 3: 4, 4: 4, 5: 4, 6: 4, 7: 7, 8: 4, 9: 9, 10: 4, 11: 11, 12: 12, 13: 13, 14: 14, 15: 4}
        assert osp.exists(self.img_dir)
        if 'Descend' in self.img_dir:  # Reduce the number of Descend images
            # self.img_infos = self.img_infos[::5]
            if use_mask is not None:  # Then use the mask to define the ignore region
                self.gt_seg_map_loader = LoadAnnotationsWithMask()
            else:  # Normal mode - use every fifth image to reduce time
                self.img_infos = self.img_infos[::5]
