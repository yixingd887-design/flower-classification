from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import ResNet


# 项目路径
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"


# 类别名称
classes = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]


# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_path):

    # 设备
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("运行设备:", device)


    # 创建模型
    model = ResNet(
        num_classes=5
    )

    # 加载训练好的参数
    checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

    model.load_state_dict(
    checkpoint["model_state_dict"]
)

    model.to(device)

    model.eval()


    # 读取图片
    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    # 增加batch维度
    image = image.unsqueeze(0)

    image = image.to(device)


    # 预测
    with torch.no_grad():

        output = model(image)

        probability = torch.softmax(
            output,
            dim=1
        )

        confidence, index = torch.max(
            probability,
            dim=1
        )


    print(
        "预测类别:",
        classes[index.item()]
    )

    print(
        "置信度:",
        f"{confidence.item()*100:.2f}%"
    )



if __name__ == "__main__":

    image_path = input(
        "请输入图片路径:"
    )

    predict(image_path)