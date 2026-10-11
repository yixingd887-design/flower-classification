
from pathlib import Path
import random
import shutil

# 项目目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "data" / "flower_photos"
OUTPUT_DIR = PROJECT_DIR / "data" / "split"

# 固定随机种子，确保划分结果可复现
random.seed(42)

# 每类图片中，80%用于训练，20%用于验证
TRAIN_RATIO = 0.8

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"找不到数据集：{SOURCE_DIR}")

# 如果目标目录已存在，先停止，防止覆盖已有数据
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"目标目录已存在：{OUTPUT_DIR}\n"
        "请勿重复运行，以免混淆数据。"
    )

class_dirs = sorted(
    p for p in SOURCE_DIR.iterdir()
    if p.is_dir()
)

if not class_dirs:
    raise RuntimeError("没有找到任何花卉类别文件夹。")

total_train = 0
total_val = 0

for class_dir in class_dirs:
    images = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if len(images) < 2:
        raise RuntimeError(f"{class_dir.name} 的图片太少。")

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
    ]:
        target_dir = OUTPUT_DIR / split_name / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, target_dir / image_path.name)

    total_train += len(train_images)
    total_val += len(val_images)

    print(
        f"{class_dir.name}: "
        f"训练 {len(train_images)} 张，"
        f"验证 {len(val_images)} 张"
    )

print("-" * 40)
print(f"训练集总数：{total_train}")
print(f"验证集总数：{total_val}")
print(f"划分完成，保存位置：{OUTPUT_DIR}")
