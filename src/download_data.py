

from pathlib import Path
import tarfile
import urllib.request

# 项目根目录与数据目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "flower_photos.tgz"

# TensorFlow 官方示例花卉数据集
URL = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/example_images/flower_photos.tgz"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    dataset_dir = DATA_DIR / "flower_photos"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"数据集已存在，无需重复下载：{dataset_dir}")
        return

    try:
        if not ARCHIVE_PATH.exists():
            print("开始下载花卉数据集，请耐心等待……")
            urllib.request.urlretrieve(URL, ARCHIVE_PATH)

        if not tarfile.is_tarfile(ARCHIVE_PATH):
            raise RuntimeError("下载文件不是有效的 tar.gz 压缩包。")

        print("下载完成，开始解压……")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
            # 检查解压目标，避免路径穿越
            root = DATA_DIR.resolve()
            for member in archive.getmembers():
                target = (DATA_DIR / member.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"压缩包包含不安全路径：{member.name}")

            archive.extractall(DATA_DIR)

        if not dataset_dir.exists():
            raise RuntimeError("解压后未找到 flower_photos 目录。")

        print(f"数据集准备完成：{dataset_dir}")

    except Exception as exc:
        print(f"数据集准备失败：{exc}")
        print("请检查网络连接和磁盘空间，然后重新运行。")
        raise


if __name__ == "__main__":
    main()
