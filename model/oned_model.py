#%%
import math
from typing import Tuple, Union

import numpy as np
from torch import nn


def init_layers(model: nn.Module):
    if getattr(model, "bias", None) is not None:
        nn.init.constant_(model.bias, 0)
    if isinstance(model, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(model.weight)
    for layer in model.children():
        init_layers(layer)


class Base1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        init_out_channel = self._nearest_pow2(in_channels * 9)
        self.model = nn.Sequential(
            ConvLayer(
                in_channels=in_channels,
                out_channels=init_out_channel,
                kernel_size=3,
                stride=1,
            ),  # (bs, init_out_channel, length >= 62)
            *[
                ConvLayer(
                    in_channels=init_out_channel * pow(2, i),
                    out_channels=init_out_channel * pow(2, i + 1),
                    kernel_size=3,
                    stride=2,
                )
                for i in range(5)
            ],
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                in_features=init_out_channel * pow(2, 5), out_features=4, bias=True,
            ),
        )
        init_layers(self.model)

    def _nearest_pow2(self, num: int):
        return pow(2, math.floor(np.log2(num)))

    def forward(self, x_batch):
        return self.model(x_batch)


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: Union[int, Tuple] = None,
        kernel_size: Union[int, Tuple] = 3,
        stride: Union[int, Tuple] = 2,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
    ):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x_batch):
        return super().forward(x_batch)
