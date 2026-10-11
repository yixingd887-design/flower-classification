
import torch
from model import ResidualBlock

# 构造一个假数据批次：8 张图片，每张 3 个通道，尺寸 32×32
x = torch.randn(8, 3, 32, 32)

# 输入输出通道数都是 3，尺寸不变
block1 = ResidualBlock(3, 3)
y1 = block1(x)
print("测试一：", x.shape, "->", y1.shape)

# 输出通道变为 16，空间尺寸缩小一半
block2 = ResidualBlock(3, 16, stride=2)
y2 = block2(x)
print("测试二：", x.shape, "->", y2.shape)

assert y1.shape == (8, 3, 32, 32)
assert y2.shape == (8, 16, 16, 16)

print("残差块测试全部通过！")
