import torch
import torch.nn as nn
import numpy as np
from typing import Union


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

        self.expansion = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride
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


def scale_width(w, w_factor):
  """Scales width given a scale factor"""
  w *= w_factor
  new_w = (int(w+4) // 8) * 8
  new_w = max(8, new_w)
  if new_w < 0.9*w:
     new_w += 8
  return int(new_w)

class EfficientNet(nn.Module):

    """

    """

    def __init__(
            self,
            stem_params: list = [3, 32, 2, 1],
            num_of_mb6_blocks: int = 6,
            out_size: int = 1000,
            w_factor: Union[int, float] = 1,
            d_factor: Union[int, float] = 1,
            mb1_params: list,
            mb6_params: dict,
            last_conv_params: list

    ):
        super(EfficientNet, self).__init__()

        assert num_of_mb6_blocks == len(mb6_params.keys()), \
            'number of MBConv6 layers must be the same as amount of keys in dict'

        # Stem
        self.stem_conv = ConvLayer(*stem_params)

        # first MBConv1 block
        self.mbconv1 = MBConv1(*mb1_params)

        # second block
        self.mbconv6_blocks = nn.ModuleList([])
        for i in range(num_of_mb6_blocks):
            self.mbconv6_blocks.append(
                MBConv6(*mb6_params[i])
            )

        self.conv = ConvLayer(*last_conv_params)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_conv_params[1], out_size)
        )

        self._initialize_weights()


    def _initialize_weights(self) -> None:
        """
        Weights initialization.
        For convolutional blocks there is "He initialization".
        :return:
            None
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.stem_conv(x)
        x = self.mbconv1(x)
        for block in self.mbconv6_blocks:
            x = block(x)
        x = self.conv(x)
        x = self.head(x)

        return x


