
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


# 项目根目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "split"
RESULTS_DIR = PROJECT_DIR / "results"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(directory):
    """统计指定目录下每个类别的图片数量。"""
    counts = {}

    if not directory.exists():
        raise FileNotFoundError(f"找不到数据目录：{directory}")

    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir():
            continue

        counts[class_dir.name] = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_SUFFIXES
        )

    return counts


def main():
    train_counts = count_images(DATA_DIR / "train")
    val_counts = count_images(DATA_DIR / "val")

    if not train_counts:
        raise RuntimeError("训练集中没有找到花卉类别文件夹。")

    if set(train_counts) != set(val_counts):
        raise ValueError("训练集与验证集的类别不一致，请检查数据目录。")

    # 使用 Pandas 整理统计表
    df = pd.DataFrame({
        "类别": list(train_counts.keys()),
        "训练集数量": list(train_counts.values()),
        "验证集数量": [
            val_counts[name] for name in train_counts
        ],
    })

    df["总数量"] = df["训练集数量"] + df["验证集数量"]

    # 使用 NumPy 进行统计分析
    train_array = np.asarray(df["训练集数量"], dtype=np.int64)
    val_array = np.asarray(df["验证集数量"], dtype=np.int64)
    total_array = np.asarray(df["总数量"], dtype=np.int64)

    print("各类别图片数量：")
    print(df.to_string(index=False))

    print("\n数据集统计：")
    print(f"训练集总数：{np.sum(train_array)}")
    print(f"验证集总数：{np.sum(val_array)}")
    print(f"图片总数：{np.sum(total_array)}")
    print(f"每类平均图片数：{np.mean(total_array):.1f}")
    print(f"类别数量范围：{np.min(total_array)} 至 {np.max(total_array)}")

    # 保存统计表
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "class_distribution.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 使用 Matplotlib 绘制类别分布图
    plot_df = df.set_index("类别")[
        ["训练集数量", "验证集数量"]
    ]

    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Flower Dataset Class Distribution")
    ax.set_xlabel("Flower Class")
    ax.set_ylabel("Number of Images")
    ax.legend(["Training Set", "Validation Set"])
    plt.xticks(rotation=30)
    plt.tight_layout()

    image_path = RESULTS_DIR / "class_distribution.png"
    plt.savefig(image_path, dpi=150)
    plt.close()

    print(f"\n统计表已保存：{csv_path}")
    print(f"类别分布图已保存：{image_path}")


if __name__ == "__main__":
    main()
