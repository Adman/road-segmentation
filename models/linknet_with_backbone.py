import segmentation_models as sm
from .metrics import mean_iou


def linknet_with_backbone(encoder, pretrained_weights=None,
                          input_size=(480, 640, 3), loss='binary_crossentropy'):
    model = sm.Linknet(backbone_name=encoder, encoder_weights='imagenet',
                       freeze_encoder=True)
    model.compile('Adam', loss=loss, metrics=['accuracy', mean_iou])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def linknet_vgg16(pretrained_weights=None, input_size=(480, 640, 3), loss='binary_crossentropy'):
    return linknet_with_backbone('vgg16', pretrained_weights=pretrained_weights,
                                 input_size=input_size, loss=loss)

