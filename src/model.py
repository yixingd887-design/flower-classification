
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """基础残差块：两次卷积 + 一条跳跃连接。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 第一次卷积：提取特征
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二次卷积：进一步提取特征
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出形状不同，调整跳跃连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 核心：将卷积结果和跳跃连接相加
        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """用于五种鲜花分类的简化 ResNet。"""

    def __init__(self, num_classes=5):
        super().__init__()

        # 初始卷积层：提取基础特征
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 三组残差层，逐渐增加通道数
        self.layer1 = self._make_layer(32, 32, stride=1)
        self.layer2 = self._make_layer(32, 64, stride=2)
        self.layer3 = self._make_layer(64, 128, stride=2)

        # 将每个通道的特征图压缩成一个数
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 输出五个类别的分数
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x