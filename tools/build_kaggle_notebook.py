"""Tao notebooks/kaggle_train.ipynb, nhung san src/*.py de chi can tai mot file len Kaggle.

    python tools/build_kaggle_notebook.py
"""
from pathlib import Path
from hashlib import sha256
import argparse

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = (
    "prepare_data.py", "eda.py", "crop_persons.py", "fight_data.py", "models.py",
    "metrics.py", "train_utils.py", "train_cnn.py", "train_yolo_cls.py",
    "train_cnn_lstm.py", "prediction_analysis.py", "evaluate.py",
)


def training_requirements():
    packages = []
    for path in (PROJECT / "requirements.txt", PROJECT / "requirements/train.txt"):
        for raw in path.read_text(encoding="utf-8").splitlines():
            package = raw.split("#", 1)[0].strip()
            if package and not package.startswith("-r"):
                packages.append(package)
    return list(dict.fromkeys(packages))

INTRO = """# Phát hiện đánh nhau: huấn luyện trên Kaggle

Trước khi chạy, trong phần **Settings** ở cột phải của notebook:

1. **Input → Add Input:** thêm bộ dữ liệu `vulamnguyen/rwf2000`.
2. **Accelerator:** chọn GPU.
3. **Internet:** bật On (cần xác minh số điện thoại Kaggle), để cài Ultralytics và tải trọng số.

Sau đó chọn **Save Version → Save & Run All** để notebook chạy nền, không cần giữ trình duyệt mở. Kết quả nằm ở tab **Output**, file `results.zip`.

File này được sinh bởi `tools/build_kaggle_notebook.py`, nên sửa code trong `src/` rồi sinh lại, không sửa trực tiếp ở đây."""

SETUP = """import json, os, platform, shutil, subprocess, sys
from importlib.metadata import version
from pathlib import Path

# No --upgrade and no local constraints: preserve compatible Kaggle GPU packages.
TRAIN_REQUIREMENTS = __TRAIN_REQUIREMENTS__
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *TRAIN_REQUIREMENTS], check=True)

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
THRESHOLD_STRATEGY = os.environ.get('FIGHT_THRESHOLD_STRATEGY', 'f2')
MIN_RECALL = float(os.environ.get('FIGHT_MIN_RECALL', .9))
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
        "import pandas as pd\n"
        "common = ['--image-size', IMAGE_SIZE, '--threshold-strategy', THRESHOLD_STRATEGY, '--min-recall', MIN_RECALL]\n"
        "run('evaluate.py', '--model', 'm1', '--frames', DATA / 'frames', *common)\n"
        "run('evaluate.py', '--model', 'm2', '--frames', DATA / 'frames', *common)\n"
        "run('evaluate.py', '--model', 'm3', '--frames', DATA / 'crops', '--full-frames', DATA / 'frames',\n"
        "    '--yolo-weights', 'yolo11n.pt', *common)\n"
        "pd.read_csv('results/comparison.csv')\n",
    ),
    (
        "## 7. Đóng gói kết quả\n\nTải `results.zip` ở tab Output, gồm cả checkpoint và `splits.csv`.",
        "environment = {\n"
        "    'python': platform.python_version(), 'platform': platform.platform(),\n"
        "    'packages': {name: version(name) for name in TRAIN_REQUIREMENTS},\n"
        "    'torch_cuda': torch.version.cuda,\n"
        "    'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',\n"
        "}\n"
        "Path('results/environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')\n"
        "shutil.copy('splits/splits.csv', 'results/splits.csv')\n"
        "shutil.make_archive('results', 'zip', 'results')\n"
        "print('Da tao results.zip')\n",
    ),
]

EVALUATE_INTRO = """# Đánh giá lại checkpoint trên Kaggle — không huấn luyện

Thêm hai Input: **RWF-2000** và thư mục kết quả đã giải nén chứa
`m1/best.pt`, `m2/best.pt`, `m3/best.pt`, `splits.csv` từ cùng lần huấn luyện.
Bật GPU và Internet. Không cần chạy notebook huấn luyện lại.

Notebook giữ nguyên split cũ, xuất `preds_val.csv`, `preds_test.csv`, bảng so sánh
F2/F1/Precision với ràng buộc Recall và danh sách FP/FN cần xem lại.
Ngưỡng luôn chọn trên validation. Không dùng điểm test để chọn chính sách.

Có thể đặt `FIGHT_CHECKPOINTS` trong cell đầu nếu cần chỉ rõ thư mục kết quả;
`FIGHT_THRESHOLD_STRATEGY` chọn f2/f1/precision_at_recall (mặc định f2).
Tải `results.zip` ở Output sau khi hoàn tất.
"""

CHECKPOINT_SETUP = """
checkpoint_root = os.environ.get('FIGHT_CHECKPOINTS')
if checkpoint_root:
    CHECKPOINTS = Path(checkpoint_root)
else:
    CHECKPOINTS = next((p.parent.parent for p in Path('/kaggle/input').rglob('best.pt')
                        if p.parent.name == 'm1' and
                        all((p.parent.parent / m / 'best.pt').is_file() for m in ['m1', 'm2', 'm3'])), None)
if CHECKPOINTS is None or not (CHECKPOINTS / 'splits.csv').is_file():
    raise FileNotFoundError('Can thu muc checkpoint goc kem splits.csv; khong tu chia lai train/val/test.')
for name in ['m1', 'm2', 'm3']:
    target = WORK / 'results' / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in ['best.pt', 'config.json', 'history.csv']:
        source = CHECKPOINTS / name / filename
        if source.is_file():
            shutil.copy2(source, target / filename)
print('Checkpoint source:', CHECKPOINTS)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    setup = SETUP.replace("__TRAIN_REQUIREMENTS__", repr(training_requirements()))
    cells = [new_markdown_cell(EVALUATE_INTRO if args.evaluate_only else INTRO), new_code_cell(setup + (CHECKPOINT_SETUP if args.evaluate_only else ""))]
    for name in TRAINING_SCRIPTS:
        path = PROJECT / "src" / name
        source = path.read_text(encoding="utf-8")
        cells.append(new_code_cell(f"(SRC / {path.name!r}).write_text({source!r}, encoding='utf-8');"))
    steps = STEPS
    if args.evaluate_only:
        steps = [
            ("## 1. Trích frame theo split gốc", "run('prepare_data.py', '--root', ROOT, '--frames', DATA / 'frames', '--reuse-splits', CHECKPOINTS / 'splits.csv')\n"),
            ("## 2. Crop người cho M3", "run('crop_persons.py', '--frames', DATA / 'frames', '--out', DATA / 'crops', '--size', IMAGE_SIZE)\n"),
            (STEPS[-2][0].replace("## 6.", "## 3."), STEPS[-2][1]),
            (STEPS[-1][0].replace("## 7.", "## 4."), STEPS[-1][1]),
        ]
    for title, code in steps:
        cells += [new_markdown_cell(title), new_code_cell(code)]
    notebook = new_notebook(
        cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    )
    for cell in notebook.cells:
        cell["id"] = sha256(f"{cell.cell_type}\n{cell.source}".encode("utf-8")).hexdigest()[:12]
    filename = "kaggle_evaluate.ipynb" if args.evaluate_only else "kaggle_train.ipynb"
    with (PROJECT / "notebooks" / filename).open("w", encoding="utf-8", newline="\n") as stream:
        nbformat.write(notebook, stream)


if __name__ == "__main__":
    main()
