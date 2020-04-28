import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone 
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from .fpn import LastLevelP6P7, LastLevelP6

__all__ = ["SimpleNet", "build_simplenet_backbone", "build_simplenet_fpn_backbone"]

class SimpleNet(Backbone):
    def __init__(self, cfg, input_ch, out_features=None):
    
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_ch, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1)
        self.conv12 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm5 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm6 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm9 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm11 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm12 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.batchnorm13 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        
        # self._out = nn.Linear(in_features=256*1*1, out_features=100)

        self._out_features = ["stage2", "stage3", "stage4", "stage5"]
        self._out_features = out_features
        self._out_feature_channels = {"stage2": 256, "stage3": 512, "stage4": 2048, "stage5": 256}
        self._out_feature_strides = {"stage2": 4, "stage3": 8, "stage4": 16, "stage5": 32}
    
    def forward(self, t):

        #(1) input layer
        t=t

        #(2) conv layers
        t=self.conv1(t)
        t=self.batchnorm1(t)
        t=F.relu(t, inplace=True)

        t=self.conv2(t)
        t=self.batchnorm2(t)
        t=F.relu(t, inplace=True)

        t=self.conv3(t)
        t=self.batchnorm3(t)
        t=F.relu(t, inplace=True)

        t=self.conv4(t)
        t=self.batchnorm4(t)
        t=F.relu(t, inplace=True)
        t=F.max_pool2d(t, kernel_size=2, stride=2)
        t=F.dropout2d(t, p=0.1)

        t=self.conv5(t)
        t=self.batchnorm5(t)
        t=F.relu(t, inplace=True)

        t=self.conv6(t)
        t=self.batchnorm6(t)
        t=F.relu(t, inplace=True)

        t=self.conv7(t)
        t=self.batchnorm7(t)
        t=F.relu(t, inplace=True)
        t=F.max_pool2d(t, kernel_size=2, stride=2)
        t=F.dropout2d(t, p=0.1)

        out1=self.conv8(t)
        t=self.batchnorm8(out1)
        t=F.relu(t, inplace=True)

        t=self.conv9(t)
        t=self.batchnorm9(t)
        t=F.relu(t, inplace=True)
        t=F.max_pool2d(t, kernel_size=2, stride=2)
        t=F.dropout2d(t, p=0.1)

        out2=self.conv10(t)
        t=self.batchnorm10(out2)
        t=F.relu(t, inplace=True)
        t=F.max_pool2d(t, kernel_size=2, stride=2)
        t=F.dropout2d(t)

        out3=self.conv11(t)
        t=self.batchnorm11(out3)
        t=F.relu(t, inplace=True)

        t=self.conv12(t)
        t=self.batchnorm12(t)
        t=F.relu(t, inplace=True)
        t=F.max_pool2d(t, kernel_size=2, stride=2)
        t=F.dropout2d(t, p=0.1)

        out4=self.conv13(t)
        # t=self.batchnorm13(out4)
        # t=F.relu(t, inplace=True)

        # t=t.reshape(-1, 256*1*1)
        # t=self.out(t)

        return {
            'stage1': out1,
            'stage2': out2,
            'stage3': out3,
            'stage4': out4
        }
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    

@BACKBONE_REGISTRY.register()
def build_simplenet_backbone(cfg, input_shape):
    '''
    create a SimpleNet instance from config

    returns:
        SimpleNet: a class `SimpleNet` instance
    '''
    out_features = cfg.MODEL.SIMPLENET.OUT_FEATURES 

    return SimpleNet(cfg, input_shape.channels, out_features=out_features)

@BACKBONE_REGISTRY.register()
def build_simplenet_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_simplenet_backbone(cfg, input_shape)
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
def build_fcos_simplenet_fpn_backbone(cfg, input_shape: ShapeSpec):

    bottom_up = build_simplenet_backbone(cfg, input_shape)
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
