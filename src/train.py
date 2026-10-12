import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from model import ResNet


# 项目目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 训练参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# 选择运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
    """训练一轮，返回平均损失和准确率。"""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一批次的梯度
        optimizer.zero_grad()

        # 前向传播：计算预测分数
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播：计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """在验证集上评估模型，不更新参数。"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    print("运行设备：", device)

    train_loader, val_loader, classes = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    print("类别：", classes)

    model = ResNet(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2%} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2%}"
        )

        # 只保存验证准确率更高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "val_acc": val_acc,
                },
                MODEL_DIR / "best_model.pth",
            )

            print("已保存当前最佳模型。")

    
    print(f"训练完成，最佳验证准确率：{best_val_acc:.2%}")
    print("模型保存位置：", MODEL_DIR / "best_model.pth")

    # 创建结果目录
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=150)
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, EPOCHS + 1), val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    print("训练曲线已保存：", results_dir)


if __name__ == "__main__":
    main()
