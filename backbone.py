import torch
import torch.nn as nn
import numpy as np


class ConvLayer(nn.Module):
    """
    Layer for convolution operations. It aggregates also batch normalization and activation functions
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            groups: int = 1,
            bn: bool = True,
            act: bool = True
    ):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


def drop_connect(
        inputs: torch.Tensor,
        p: float = 0.1,
        training: bool = True
) -> torch.Tensor:
    """
    drop c
    :param inputs:
    :param p:
    :param training:
    :return:
    """
    assert 0 <= p <= 1, 'p must be in range of [0, 1]'
    if not training:
        return inputs

    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([inputs.shape[0], 1, 1, 1], dtype = inputs.dtype, device = inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class SEmodule(nn.Module):
    """
    Interdependencies between the channels of convolutional features
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(
            self,
            in_channels: int,
            scale_factor: int = 24
    ):
        super(SEmodule, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels // scale_factor,
            kernel_size = 1,
            stride = 1,
        )

        self.activation_1 = nn.SiLU()
        self.conv_2 = nn.Conv2d(
            in_channels = in_channels // scale_factor,
            out_channels = in_channels,
            kernel_size = 1,
            stride = 1,
        )
        self.activation_2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.activation_1(self.conv_1(y))
        y = self.activation_2(self.conv_2(y))

        return x * y


#
class MBConvBlock(nn.Module):
    """

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_f: int,
            kernel_size: int = 3,
            stride: int = 1,
            scale_f: int = 24,
            training: bool = True,
            drop_p: float = 0.2
    ):
        super(MBConvBlock, self).__init__()

        expanded_ch = in_channels * expansion_f

        padding = int(np.ceil((kernel_size - stride) / 2))

        self.skip_connection = (in_channels == out_channels) and (stride == 1)

        self.expansion = nn.Conv2d(
            in_channels = in_channels,
            out_channels = expanded_ch,
            kernel_size = 1
        )

        self.depthwise = ConvLayer(
            expanded_ch, expanded_ch,
            kernel_size, stride,
            padding, expanded_ch
        )

        self.se = SEmodule(expanded_ch, scale_factor = scale_f)

        self.reducer = ConvLayer(
            expanded_ch, out_channels,
            padding = 0,
            kernel_size = 1, act = False
        )

        self.training = training
        self.drop_p = drop_p

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        input = x

        x = self.expansion(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reducer(x)
        if self.skip_connection:
            x = drop_connect(x, self.drop_p)
            x += input

        return x


class MBConv1(MBConvBlock):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            scale_f: int = 24,
            training: bool = True,
            drop_p: float = 0
    ):
        super(MBConv1, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            expansion_f = 1,
            kernel_size = kernel_size,
            stride = stride,
            scale_f = scale_f,
            training = training,
            drop_p = drop_p
        )

class MBConv6(MBConvBlock):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            scale_f: int = 24,
            training: bool = True,
            drop_p: float = 0
    ):
        super(MBConv6, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            expansion_f = 6,
            kernel_size = kernel_size,
            stride = stride,
            scale_f = scale_f,
            training = training,
            drop_p = drop_p
        )

class EfficientNet(nn.Module):

    """

    """

    def __init__(
            self,
            stem_params: list = [3, 32, 2, 1],
            mb_params: dict,

    ):
        super(EfficientNet, self).__init__()

        # Stem
        self.stem_conv = nn.Conv2d(*stem_params)
        self.bn0 = nn.BatchNorm2d(stem_params[1])

        pass
