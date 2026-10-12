from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "split" / "train"
VAL_DIR = PROJECT_DIR / "data" / "split" / "val"

# 将所有图片调整为 224×224，并转换为 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def get_dataloaders(batch_size=32):
    """创建训练集和验证集的数据加载器。"""

    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=transform,
    )

    # 确保训练集和验证集的类别顺序一致
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("训练集和验证集的类别名称或顺序不一致")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


if __name__ == "__main__":
    train_loader, val_loader, classes = get_dataloaders()

    print("类别名称：", classes)
    print("训练集图片数：", len(train_loader.dataset))
    print("验证集图片数：", len(val_loader.dataset))

    images, labels = next(iter(train_loader))

    print("图片批次形状：", images.shape)
    print("标签形状：", labels.shape)
    print("GPU 可用：", __import__("torch").cuda.is_available())