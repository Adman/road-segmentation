from .unet import unet
from .fcn_vgg16_32s import fcn_vgg16_32s
from .segnet import segnet, segnetsmall
from .resnet import resnet, resnetsmall
from .resnet_bnn import resnet_bnn

from .shuffleseg import shuffleseg
from .shufflenetv2 import shufflenetv2

from .mobilenetv2 import mobilenetv2
from .mobilenetv3 import mobilenetv3

from .segnet_mobilenet import segnet_mobilenet

from .unet_with_backbone import unet_resnet34, unet_resnet50, unet_vgg16, unet_mobilenetv2

from .linknet_with_backbone import (
    linknet_resnet50,
    linknet_vgg16,
    linknet_mobilenetv2,
    linknet_efficientnetb0,
    linknet_efficientnetb7,
)

from .fpn_with_backbone import fpn_resnet34, fpn_resnet50, fpn_vgg16
