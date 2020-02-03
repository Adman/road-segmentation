import segmentation_models as sm
from .metrics import mean_iou


def linknet_with_backbone(encoder, input_size=(480, 640, 3), loss='binary_crossentropy'):
    model = sm.Linknet(backbone_name=encoder, encoder_weights='imagenet',
                       encoder_freeze=True)
    model.compile('Adam', loss=loss, metrics=['accuracy', mean_iou])

    return model


def linknet_resnet50(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return linknet_with_backbone('resnet50', input_size=input_size, loss=loss)


def linknet_vgg16(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return linknet_with_backbone('vgg16', input_size=input_size, loss=loss)


def linknet_mobilenetv2(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return linknet_with_backbone('mobilenetv2', input_size=input_size, loss=loss)


def linknet_efficientnetb0(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return linknet_with_backbone('efficientnetb0', input_size=input_size, loss=loss)


def linknet_efficientnetb7(input_size=(480, 640, 3), loss='binary_crossentropy'):
    return linknet_with_backbone('efficientnetb7', input_size=input_size, loss=loss)

