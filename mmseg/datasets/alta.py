# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AltaDataset(CustomDataset):
    """Alta dataset.
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

    # PALETTE = [
    #     [0, 0, 0],  # 0
    #     [255, 127, 50],  # 2
    #     [50, 101, 255],  # 11
    #     [50, 255, 101],  # 7
    #     [50, 255, 255],  # 9
    #     [76, 50, 255],  # 12
    #     [229, 50, 255],  # 14
    #     [153, 50, 255],  # 13
    #     [255, 204, 50],  # 3
    #     [229, 255, 50],  # 4
    #     [153, 255, 50],  # 5
    #     [76, 255, 50],  # 6
    #     [50, 255, 178],  # 8
    #     [50, 178, 255],  # 10
    #     [255, 50, 204],  # 15
    #     [255, 50, 50],  # 1
    # ]

    def __init__(self, **kwargs):
        super(AltaDataset, self).__init__(
            img_suffix='.JPG',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            classes=self.CLASSES,
            palette=self.PALETTE,
            **kwargs)
        self.label_map = {0: 0, 1: 4, 2: 2, 3: 4, 4: 4, 5: 4, 6: 4, 7: 7, 8: 4, 9: 9, 10: 4, 11: 11, 12: 12, 13: 13, 14: 14, 15: 4}
        assert osp.exists(self.img_dir)

    # def prepare_test_img(self, idx):  # uncomment if LoadAnnotations is needed during testing...
    #     """Get testing data after pipeline.
    #
    #     Args:
    #         idx (int): Index of data.
    #
    #     Returns:
    #         dict: Testing data after pipeline with new keys introduced by
    #             pipeline.
    #     """
    #
    #     img_info = self.img_infos[idx]
    #
    #     # results = dict(img_info=img_info)
    #     ann_info = self.get_ann_info(idx)
    #     results = dict(img_info=img_info, ann_info=ann_info)
    #
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)

# Implement sub-sampling per class
# Implement merging of several classes
# Train and test Segformer
# How are the segmentation numbers (during loading) set? Is it consistent for all images, even when some categories are missing?