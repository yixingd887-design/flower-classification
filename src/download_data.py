
from pathlib import Path
import tarfile
import urllib.request

# 获取项目目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "flower_photos.tgz"

# 官方花卉数据集下载地址
URL = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/example_images/flower_photos.tgz"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

if (DATA_DIR / "flower_photos").exists():
    print("数据集已经存在，无需重复下载。")
else:
    print("开始下载花卉数据集，约 218 MB，请耐心等待……")

    urllib.request.urlretrieve(URL, ARCHIVE_PATH)

    print("下载完成，开始解压……")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        archive.extractall(DATA_DIR)

    print("解压完成！")
    print("数据集位置：", DATA_DIR / "flower_photos")
