from segmentation_models import FPN
from .metrics import mean_iou


def fpn_with_backbone(encoder, input_size=(480, 640, 3), loss='binary_crossentropy'):
    model = FPN(
        backbone_name=encoder,
        encoder_weights='imagenet',
        classes=1,
        input_shape=input_size,
        activation='sigmoid',
        encoder_freeze=True
    )
    model.compile('Adam', loss=loss, metrics=['accuracy', mean_iou])

    return model


def fpn_resnet34(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return fpn_with_backbone('resnet34', input_size=input_size, loss=loss)


def fpn_resnet50(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return fpn_with_backbone('resnet50', input_size=input_size, loss=loss)


def fpn_vgg16(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return fpn_with_backbone('vgg16', input_size=input_size, loss=loss)
