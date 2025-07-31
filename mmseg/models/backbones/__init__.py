# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer

from .mboaz17.mit_histloss import MixVisionTransformerHistLoss  # <messi>
from .mboaz17.resnet_normless import (ResNetNormLess, ResNetV1cNormLess, ResNetV1dNormLess)  # <messi>
from .mboaz17.resnet_WN import (ResNetWeightNorm, ResNetV1cWeightNorm, ResNetV1dWeightNorm)  # <messi>

__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE'
]

__all__.append('MixVisionTransformerHistLoss')  # <messi>
__all__.append('ResNetNormLess')  # <messi>
__all__.append('ResNetV1cNormLess')  # <messi>
__all__.append('ResNetV1dNormLess')  # <messi>
__all__.append('ResNetWeightNorm')  # <messi>
__all__.append('ResNetV1cWeightNorm')  # <messi>
__all__.append('ResNetV1dWeightNorm')  # <messi>
