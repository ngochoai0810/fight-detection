"""Tao notebooks/kaggle_train.ipynb, nhung san src/*.py de chi can tai mot file len Kaggle.

    python tools/build_kaggle_notebook.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT = Path(__file__).resolve().parents[1]

INTRO = """# Phát hiện đánh nhau: huấn luyện trên Kaggle

Trước khi chạy, trong phần **Settings** ở cột phải của notebook:

1. **Input → Add Input:** thêm bộ dữ liệu `vulamnguyen/rwf2000`.
2. **Accelerator:** chọn GPU.
3. **Internet:** bật On (cần xác minh số điện thoại Kaggle), để cài Ultralytics và tải trọng số.

Sau đó chọn **Save Version → Save & Run All** để notebook chạy nền, không cần giữ trình duyệt mở. Kết quả nằm ở tab **Output**, file `results.zip`.

File này được sinh bởi `tools/build_kaggle_notebook.py`, nên sửa code trong `src/` rồi sinh lại, không sửa trực tiếp ở đây."""

SETUP = """import os, shutil, subprocess, sys
from pathlib import Path

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)

def find_dataset():
    if os.environ.get('FIGHT_ROOT'):
        return Path(os.environ['FIGHT_ROOT'])
    for train in Path('/kaggle/input').rglob('train'):
        if (train / 'Fight').is_dir() and (train.parent / 'val').is_dir():
            return train.parent
    raise FileNotFoundError('Khong thay RWF-2000. Hay them bo vulamnguyen/rwf2000 vao Input.')

ROOT = find_dataset()
WORK = Path(os.environ.get('FIGHT_WORK', '/kaggle/working'))
# Frame chi la du lieu trung gian nen de ngoai thu muc output cua Kaggle.
DATA = Path(os.environ.get('FIGHT_DATA', '/tmp/fight'))
EPOCHS = int(os.environ.get('FIGHT_EPOCHS', 30))
IMAGE_SIZE = int(os.environ.get('FIGHT_IMAGE_SIZE', 224))
EXTRA = ['--no-pretrained'] if os.environ.get('FIGHT_NO_PRETRAINED') else []
SRC = WORK / 'src'
SRC.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
print('Du lieu:', ROOT)

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHONG CO, hay bat GPU trong Settings')

def run(script, *args):
    subprocess.run([sys.executable, str(SRC / script), *map(str, args)], check=True)
"""

STEPS = [
    ("## 1. Kiểm tra và chia dữ liệu, trích frame", "run('prepare_data.py', '--root', ROOT, '--frames', DATA / 'frames')\n"),
    (
        "## 2. EDA",
        "run('eda.py', '--root', ROOT, '--yolo-weights', 'yolo11n.pt')\n"
        "from IPython.display import Image, display\n"
        "import pandas as pd\n"
        "display(pd.read_csv('results/eda/by_label.csv'))\n"
        "for name in ['samples', 'motion', 'persons_mean']:\n"
        "    display(Image(filename=f'results/eda/{name}.png'))\n",
    ),
    (
        "## 3. M1: CNN baseline",
        "run('train_cnn.py', '--frames', DATA / 'frames', '--epochs', EPOCHS, '--image-size', IMAGE_SIZE, *EXTRA)\n",
    ),
    (
        "## 4. M2: YOLO11-cls",
        "run('train_yolo_cls.py', '--frames', DATA / 'frames', '--epochs', EPOCHS, '--image-size', IMAGE_SIZE)\n",
    ),
    (
        "## 5. M3: crop người bằng YOLO11, sau đó CNN + LSTM",
        "run('crop_persons.py', '--frames', DATA / 'frames', '--out', DATA / 'crops', '--size', IMAGE_SIZE)\n"
        "run('train_cnn_lstm.py', '--frames', DATA / 'crops', '--epochs', EPOCHS, '--image-size', IMAGE_SIZE, *EXTRA)\n",
    ),
    (
        "## 6. Đánh giá trên tập test\n\nNgưỡng được chọn trên validation. Bảng cuối là so sánh ba mô hình.",
        "common = ['--image-size', IMAGE_SIZE]\n"
        "run('evaluate.py', '--model', 'm1', '--frames', DATA / 'frames', *common)\n"
        "run('evaluate.py', '--model', 'm2', '--frames', DATA / 'frames', *common)\n"
        "run('evaluate.py', '--model', 'm3', '--frames', DATA / 'crops', '--full-frames', DATA / 'frames',\n"
        "    '--yolo-weights', 'yolo11n.pt', *common)\n"
        "pd.read_csv('results/comparison.csv')\n",
    ),
    (
        "## 7. Đóng gói kết quả\n\nTải `results.zip` ở tab Output, gồm cả checkpoint và `splits.csv`.",
        "shutil.copy('splits/splits.csv', 'results/splits.csv')\n"
        "shutil.make_archive('results', 'zip', 'results')\n"
        "print('Da tao results.zip')\n",
    ),
]


def main():
    cells = [new_markdown_cell(INTRO), new_code_cell(SETUP)]
    for path in sorted((PROJECT / "src").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        cells.append(new_code_cell(f"(SRC / {path.name!r}).write_text({source!r}, encoding='utf-8');"))
    for title, code in STEPS:
        cells += [new_markdown_cell(title), new_code_cell(code)]
    notebook = new_notebook(
        cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    )
    nbformat.write(notebook, PROJECT / "notebooks" / "kaggle_train.ipynb")


if __name__ == "__main__":
    main()
