import torch
import torch.nn as nn
import numpy as np
from backbone import *

class SSD(nn.Module):

    def __init__(
            self,
            backbone_version: str,
            num_classes: int,
    ):
        super(SSD, self).__init__()

        bb_model = EfficientNet(backbone_version, num_classes)
        self.backbone = nn.ModuleList([bb_model.stem_conv, bb_model.features])
        self.additional_layers = AdditionalLayers(1280)

    def forward(self, x):
        pass


class AdditionalLayers(nn.Module):

    def __init__(
            self,
            in_channels: int,
            channels: list = [1024, 1024, 256, 512, 128, 256, 128, 256, 128, 256],
            kernels: list = [3, 1, 1, 3, 1, 3, 1, 3],
            exes: list = [3, 5]
    ):
        super(AdditionalLayers, self).__init__()

        self.layers = []
        for i, c in enumerate(channels):
            if i == 0:
                self.layers.append(nn.Conv2d(1280, c, kernels[i]))
            else:
                if i in exes:
                    self.layers.append(nn.Conv2d(channels[i - 1], c, kernels[i], stride = 2))
                else:
                    self.layers.append(nn.Conv2d(channels[i - 1], c, kernels[i], stride = 1))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):

        detections = [x]

        for l in self.layers:
            detections.append(l(x))

        return detections
