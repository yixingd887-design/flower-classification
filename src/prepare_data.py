
from pathlib import Path
import random
import shutil

# 项目根目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "data" / "flower_photos"
OUTPUT_DIR = PROJECT_DIR / "data" / "split"

# 固定随机种子，确保划分结果可复现
random.seed(42)

# 每类图片中，80% 用于训练，20% 用于验证
TRAIN_RATIO = 0.8

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"找不到原始数据集：{SOURCE_DIR}\n"
            "请先运行 python src/download_data.py"
        )

    class_dirs = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir()
    )

    if not class_dirs:
        raise RuntimeError("没有找到任何类别文件夹。")

    # 不自动覆盖已有划分，防止误删或混淆数据
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"目标目录已存在：{OUTPUT_DIR}\n"
            "为避免覆盖现有数据，本脚本已停止。\n"
            "如确实需要重新划分，请先备份并手动处理 data/split。"
        )

    total_train = 0
    total_val = 0

    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        )

        if len(images) < 2:
            raise RuntimeError(
                f"类别 {class_dir.name} 的图片不足 2 张。"
            )

        random.shuffle(images)

        split_index = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_index]
        val_images = images[split_index:]

        for split_name, split_images in (
            ("train", train_images),
            ("val", val_images),
        ):
            target_dir = OUTPUT_DIR / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                shutil.copy2(
                    image_path,
                    target_dir / image_path.name
                )

        total_train += len(train_images)
        total_val += len(val_images)

        print(
            f"{class_dir.name}: "
            f"训练集 {len(train_images)} 张，"
            f"验证集 {len(val_images)} 张"
        )

    print("-" * 40)
    print(f"训练集总数：{total_train}")
    print(f"验证集总数：{total_val}")
    print(f"划分完成，保存位置：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
