from .unet import unet
from .fcn_vgg16_32s import fcn_vgg16_32s
from .segnet import segnet
from .resnet import resnet

from .unet_with_backbone import unet_resnet34, unet_resnet50, unet_vgg16
from .linknet_with_backbone import linknet_resnet50, linknet_vgg16
from .fpn_with_backbone import fpn_resnet34, fpn_resnet50, fpn_vgg16
