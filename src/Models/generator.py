import torch
import torch.nn as nn
from .utils import *

class Generator(nn.Module):
    def __init__(self, in_features, kernel_size, maxpool_size, stride, padding):
        super(Generator, self).__init__()
        self.downsample_1 = DownSample(in_features, 64, kernel_size, maxpool_size, stride, padding)
        self.downsample_2 = DownSample(64, 128, kernel_size, maxpool_size, stride, padding)
        self.downsample_3 = DownSample(128, 256, kernel_size, maxpool_size, stride, padding)
        self.downsample_4 = DownSample(256, 512, kernel_size, maxpool_size, stride, padding)
        self.downsample_5 = DownSample(512, 1024, kernel_size, maxpool_size, stride, padding)

        self.upsample_1 = UpSample(deep_channels=1024, in_features=512, out_features=512, kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample_2 = UpSample(deep_channels=512, in_features=256, out_features=256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample_3 = UpSample(deep_channels=256, in_features=128, out_features=128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample_4 = UpSample(deep_channels=128, in_features=64,  out_features=64,  kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(64, in_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x1, skip_1 = self.downsample_1(x)
        x2, skip_2 = self.downsample_2(x1)
        x3, skip_3 = self.downsample_3(x2)
        x4, skip_4 = self.downsample_4(x3)
        x5, link = self.downsample_5(x4)

        x = self.upsample_1(link, skip_4)
        x = self.upsample_2(x, skip_3)
        x = self.upsample_3(x, skip_2)
        x = self.upsample_4(x, skip_1)

        return self.sigmoid(self.conv(x))