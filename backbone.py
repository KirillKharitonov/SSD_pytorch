import torch
import torch.nn as nn
import numpy as np

base_model = [
    # expand ratio, channels, layers, kernel_size, stride
    [1, 16, 1, 3, 1],
    [6, 24, 2, 3, 2],
    [6, 40, 2, 5, 2],
    [6, 80, 3, 5, 1],
    [6, 112, 3, 5, 1],
    [6, 192, 4, 5, 2],
    [6, 320, 1, 3, 1]
]

phi_vals = {
    # model_version : (phi_value, resolution, drop_rate)
    'e0': (0, 224, 0.2),
    'e1': (0.5, 240, 0.2),
    'e2': (1, 260, 0.3),
    'e3': (2, 300, 0.3),
    'e4': (3, 380, 0.4),
    'e5': (4, 456, 0.4),
    'e6': (5, 528, 0.5),
    'e7': (6, 600, 0.5),
}

stem_params = [3, 32, 2, 1]


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
            padding = padding,
            groups = groups
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
            expanded_ch,
            kernel_size = 3,
            stride = 1,
            padding = 1
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
            drop_p = 0.1
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
            drop_p = 0.1
        )


class EfficientNet(nn.Module):
    """

    """

    def __init__(self,
                 version: str,
                 num_classes: int,
                 last_channels: int = 1280

                 ):
        super(EfficientNet, self).__init__()

        width_factor, depth_factor, drop_rate = self.calculate_factors(version)
        self.drop_rate = drop_rate
        self.last_channels = int(np.ceil(last_channels * width_factor))
        # Stem
        self.stem_conv = ConvLayer(*stem_params)

        # Features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, self.last_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, num_classes)
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

    def calculate_factors(
            self,
            version: str,
            alpha: float = 1.2,
            beta: float = 1.1
    ):
        phi, res, drop_ = phi_vals[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_

    def create_features(
            self,
            width_factor: float,
            depth_factor: float,
            last_channels: int
    ):

        in_channels = int(stem_params[1] * width_factor)
        features = []

        curr_block = 0
        for expand_ratio, channels, layers, kernel_size, stride in base_model:
            out_channels = int(4 * np.ceil((channels * width_factor) / 4))
            scaled_layers = int(np.ceil(layers * depth_factor))

            for layer in range(scaled_layers):
                if curr_block == 0:
                    features.append(
                        MBConv1(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride = stride if layer == 0 else 1,
                            drop_p = self.drop_rate
                        )
                    )

                    in_channels = out_channels
                else:
                    features.append(
                        MBConv6(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride = stride if layer == 0 else 1,
                            drop_p = self.drop_rate
                        )
                    )
                    in_channels = out_channels

                curr_block += 1

            features.append(
                ConvLayer(
                    in_channels,
                    last_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = 0
                )
            )

            return nn.Sequential(*features)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        x = self.stem_conv(x)
        print(x.shape)
        x = self.pool(self.features(x))
        x = self.classifier(x.view(x.shape[0], -1))
        return x
