from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ResNet


# 项目路径
PROJECT_DIR = Path(__file__).resolve().parent.parent

VAL_DIR = PROJECT_DIR / "data" / "split" / "val"

MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"


# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("运行设备:", device)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 验证集
dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)


classes = dataset.classes

print("类别:", classes)


# 加载模型
model = ResNet(num_classes=5)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(device)

model.eval()


# 统计
correct = 0
total = 0


class_correct = [0] * len(classes)
class_total = [0] * len(classes)


all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


        total += labels.size(0)
        correct += (predicted == labels).sum().item()


        for i in range(len(labels)):

            label = labels[i]

            class_total[label] += 1

            if predicted[i] == label:
                class_correct[label] += 1



accuracy = 100 * correct / total


print()

print("总体准确率:")
print(f"{accuracy:.2f}%")

print()


print("各类别准确率:")


for i, name in enumerate(classes):

    acc = 100 * class_correct[i] / class_total[i]

    print(
        f"{name}: {acc:.2f}%"
    )

    # 生成混淆矩阵

cm = confusion_matrix(
    all_labels,
    all_predictions
)


plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("True")


plt.xticks(
    range(len(classes)),
    classes,
    rotation=45
)

plt.yticks(
    range(len(classes)),
    classes
)


for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


save_path = PROJECT_DIR / "results" / "confusion_matrix.png"


plt.savefig(save_path)


print()

print("混淆矩阵已保存:")
print(save_path)