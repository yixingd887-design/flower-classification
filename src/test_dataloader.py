
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 找到项目目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "split" / "train"

# 把图片统一转换成 224×224，并转成 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 读取文件夹中的图片
dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=transform,
)

# 每批读取 8 张图片
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

print("类别名称：", dataset.classes)
print("图片总数：", len(dataset))

# 取出第一批图片，检查数据格式
images, labels = next(iter(loader))

print("图片批次形状：", images.shape)
print("标签形状：", labels.shape)
print("标签内容：", labels.tolist())
print("数据类型：", images.dtype)
print("GPU 可用：", torch.cuda.is_available())
