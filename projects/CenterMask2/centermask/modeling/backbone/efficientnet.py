import torch
from torch.autograd import Variable

from efficientnet_pytorch import EfficientNet

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone 
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from .fpn import LastLevelP6P7, LastLevelP6

class EfficientBackbone(Backbone):
    def __init__(self, cfg, input_ch, out_features=None):
        super().__init__()
        assert input_ch == 3, "Colour images accepted only"

        self._out_features = ["stage2", "stage3", "stage4", "stage5"]
        self._out_features = out_features
        self._out_feature_channels = {"stage2": 24, "stage3": 40, "stage4": 112, "stage5": 1280}
        self._out_feature_strides = {"stage2": 4, "stage3": 8, "stage4": 16, "stage5": 32}
    
    def forward(self, image):
        #load pretrained model from efficientnet repo
        model = EfficientNet.from_pretrained('efficientnet-b0')
        #send model to cuda
        if torch.cuda.is_available():
            model.cuda()
        features = model.extract_features_midconv(image)

        return {
            'stage2': features[0],
            'stage3': features[1],
            'stage4': features[2],
            'stage5': features[3]
        }
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg, input_shape):
    """
    return layers
    return channels and strides for each layer
    """
    out_features = cfg.MODEL.EFFICIENTNET.OUT_FEATURES
    return EfficientBackbone(cfg, input_shape.channels, out_features=out_features)

@BACKBONE_REGISTRY.register()
def build_efficientnet_fpn_backbone(cfg, input_shape:ShapeSpec):
    bottom_up = build_efficientnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up = bottom_up,
        in_features = in_features,
        out_channels = out_channels,
        top_block = LastLevelMaxPool(),
        fuse_type = cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_fcos_efficientnet_fpn_backbone(cfg, input_shape: ShapeSpec):

    bottom_up = build_efficientnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES 
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels
    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    if top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

