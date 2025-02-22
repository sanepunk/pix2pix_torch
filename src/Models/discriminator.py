import torch.nn as nn
from .utils import *

class Discriminator(nn.Module):
    def __init__(self, in_features, kernel_size, maxpool_size, stride, padding):
        super(Discriminator, self).__init__()
        self.residual1 = ResidualBlock(in_features, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu1 = nn.LeakyReLU()

        self.residual2 = ResidualBlock(64, 64 * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu2 = nn.LeakyReLU()

        self.residual3 = ResidualBlock(64 * 2, 64 * 3, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu3 = nn.LeakyReLU()

        self.residual4 = ResidualBlock(64 * 3, 64 *4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu4 = nn.LeakyReLU()

        self.residual5 = ResidualBlock(64 * 4, 64 * 5, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu5 = nn.LeakyReLU()

        self.dim_reduce_1 = nn.Conv2d(64 * 5, 128, kernel_size=(3, 3), stride = stride, padding=0)
        self.leaky_relu6 = nn.LeakyReLU()

        self.dim_reduce_2 = nn.Conv2d(128, 1, kernel_size=(4, 4), stride = (7, 7), padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("discriminator_r1", x.shape)
        x = self.leaky_relu1(self.residual1(x))
        # print("discriminator_r2")
        x = self.leaky_relu2(self.residual2(x))
        # print("discriminator_r3")
        x = self.leaky_relu3(self.residual3(x))
        # print("discriminator_r4")
        x = self.leaky_relu4(self.residual4(x))
        # print("discriminator_r5")
        x = self.leaky_relu5(self.residual5(x))

        x = self.leaky_relu6(self.dim_reduce_1(x))
        return self.sigmoid(self.dim_reduce_2(x))