import torch
from model import ResNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(num_classes=5).to(device)
    x = torch.randn(8, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)

    print("运行设备：", device)
    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
    print("模型参数量：", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()