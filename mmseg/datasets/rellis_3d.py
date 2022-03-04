# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
import numpy as np


@DATASETS.register_module()
class RELLIS3DDataset(CustomDataset):
    """Rellis-3D dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    # CLASSES = ('background', 'vessel')

    # CLASSES = tuple(['0', '1', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15', '17', '18', '19', '23', '27', '29', '30', '31', '33', '34'])
    CLASSES = tuple(
        ['0', '1', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15', '17', '18', '19', '23', '27', '31', '33', '34'])
    # PALETTE = [[120, 120, 120], [6, 230, 230]]
    # PALETTE = [list(np.random.randint((255, 255, 255))) for i in range(34)]
    PALETTE = [[0, 0, 0],
               [108, 64, 20],
               [0, 102, 0],
               [0, 255, 0],
               [0, 153, 153],
               [0, 128, 255],
               [0, 0, 255],
               [255, 255, 0],
               [255, 0, 127],
               [64, 64, 64],
               [255, 0, 0],
               [102, 0, 0],
               [204, 153, 255],
               [102, 0, 204],
               [255, 153, 204],
               [170, 170, 170],
               [41, 121, 255],
               [134, 255, 239],
               [99, 66, 34],
               [110, 22, 138]]

    CLASSES = tuple(['{}'.format(i) for i in range(0, 35)])
    PALETTE = [list(np.random.randint((255, 255, 255))) for i in range(35)]
    def __init__(self, **kwargs):
        super(RELLIS3DDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            # classes=self.CLASSES,
            # palette=self.PALETTE,
            **kwargs)
        # self.label_map = {0: 0, 1:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 12:10, 15:11, 17:12, 18:13, 19:14, 23:15, 27:16, 31:17, 33:18, 34:19}
        assert osp.exists(self.img_dir)
