# Flower Classification 🌸

基于 PyTorch 实现的花卉图像分类项目。

本项目使用自定义 ResNet 网络，实现对五种花卉图像的自动分类：

- daisy（雏菊）
- dandelion（蒲公英）
- roses（玫瑰）
- sunflowers（向日葵）
- tulips（郁金香）


## 项目简介

本项目完整实现了一个深度学习图像分类流程：

- 数据集准备与划分
- 图像预处理
- Dataset 与 DataLoader 构建
- Residual Block 残差块实现
- 自定义 ResNet 网络搭建
- GPU 模型训练
- 模型保存与加载
- 单张图片预测


项目目标：

输入一张花卉图片，通过深度学习模型自动判断所属类别。


## 项目结构

flower-classification

├── data
│   └── split
│       ├── train
│       └── val

├── models
│   └── best_model.pth

├── results
│   ├── loss_curve.png
│   └── accuracy_curve.png

├── src
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py

├── docs

├── .gitignore

└── README.md