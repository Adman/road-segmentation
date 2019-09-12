import segmentation_models as sm
from .metrics import mean_iou


def fpn_with_backbone(encoder, pretrained_weights=None,
                      input_size=(480, 640, 3), loss='binary_crossentropy'):
    model = sm.FPN(backbone_name=encoder, encoder_weights='imagenet',
                   classes=1, input_shape=input_size, activation='sigmoid',
                   freeze_encoder=True)
    model.compile('Adam', loss=loss, metrics=['accuracy', mean_iou])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def fpn_resnet34(pretrained_weights=None, input_size=(480, 640, 3), loss='binary_crossentropy'):
    return fpn_with_backbone('resnet34', pretrained_weights=pretrained_weights,
                             input_size=input_size, loss=loss)    


def fpn_resnet50(pretrained_weights=None, input_size=(480, 640, 3), loss='binary_crossentropy'):
    return fpn_with_backbone('resnet50', pretrained_weights=pretrained_weights,
                             input_size=input_size, loss=loss)    


def fpn_vgg16(pretrained_weights=None, input_size=(480, 640, 3), loss='binary_crossentropy'):
    return fpn_with_backbone('vgg16', pretrained_weights=pretrained_weights,
                              input_size=input_size, loss=loss)    
