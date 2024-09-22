from segmentation_models import Unet
from .metrics import mean_iou


def unet_with_backbone(encoder, input_size=(480, 640, 3), loss='binary_crossentropy'):
    model = Unet(
        backbone_name=encoder,
        input_shape=input_size,
        encoder_weights='imagenet',
        encoder_freeze=True
    )
    model.compile('Adam', loss=loss, metrics=['accuracy', mean_iou])

    return model


def unet_resnet34(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return unet_with_backbone('resnet34', input_size=input_size, loss=loss)


def unet_resnet50(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return unet_with_backbone('resnet50', input_size=input_size, loss=loss)


def unet_vgg16(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return unet_with_backbone('vgg16', input_size=input_size, loss=loss)


def unet_mobilenetv2(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return unet_with_backbone('mobilenetv2', input_size=input_size, loss=loss)


