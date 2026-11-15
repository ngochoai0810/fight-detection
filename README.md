# Phát hiện đánh nhau qua video giám sát

Dự án Python/PyTorch phân loại **Fight / NonFight**, hỗ trợ phân tích video và camera trực tiếp trên dashboard.

| Model | Phương pháp |
| --- | --- |
| M1 | ResNet18 phân loại từng frame |
| M2 | YOLO11 classification |
| M3 | YOLO crop vùng người → ResNet18 → LSTM |

## Cài đặt

Chạy lệnh tại thư mục gốc dự án. Môi trường đã kiểm thử: Python 3.11.

```powershell
cd D:\fight-detection
python -m pip install -r requirements.txt
```

`requirements.txt` dùng cho dashboard và nhận diện. Các nhóm bổ sung:

| Mục đích | File cài đặt |
| --- | --- |
| Huấn luyện, EDA, đánh giá | `requirements/train.txt` |
| Kiểm thử | `requirements/test.txt` |
| Tạo/chạy notebook | `requirements/notebooks.txt` |

Phiên bản thư viện local đã kiểm thử được ghi trong `requirements/constraints-local.txt`.
Để cài theo các phiên bản này, thêm `-c requirements/constraints-local.txt` vào lệnh pip.
Đây là danh sách phiên bản dependency trực tiếp, không phải bản khóa toàn bộ môi trường.

Chuẩn bị checkpoint đã huấn luyện cho model muốn sử dụng:

```text
results/
├── m1/best.pt
├── m2/best.pt
└── m3/best.pt
yolo11n.pt          # Bắt buộc cho M3; dùng khoanh vùng người cho M1/M2 nếu có
```

Đặt `metrics.json` tương ứng cùng thư mục checkpoint để tự đọc ngưỡng Fight. Nếu thiếu, ngưỡng mặc định là `0.5`. Dữ liệu, trọng số và kết quả trong `results/`, `runs/` được bỏ qua bởi Git; cần chuẩn bị riêng trên máy.

## Chạy dashboard

```powershell
python src/dashboard.py
```

Mở **http://127.0.0.1:8501**. Dashboard tự chạy model, không cần mở terminal thứ hai.

- **Video:** chọn file tối đa 1 GB, chọn model rồi bấm **Phân tích video**. Khi hoàn tất, xem video kết quả và các clip cảnh báo.
- **Camera:** tại **Camera trực tiếp**, chọn camera `0` hoặc `1` nếu dùng USB, chọn model rồi bấm **Bật camera**. Hình trực tiếp hiển thị khung nhận diện, điểm Fight và FPS. Bấm **Tắt camera** để kết thúc phiên.
- **Bằng chứng:** xem clip cảnh báo, duyệt đúng/báo nhầm, thêm ghi chú, xác thực SHA-256 và xuất CSV/JSON.

Mỗi lần chạy camera hoặc phân tích một file video. Camera là thiết bị gắn với máy chạy dashboard và tiếp tục chạy khi tải lại/đóng tab; bấm **Tắt camera** hoặc dừng dashboard bằng `Ctrl+C` để kết thúc.

M1 phù hợp để thử trên CPU. M3 cần đủ 16 frame lấy mẫu trước khi dự đoán. Tốc độ thực tế phụ thuộc cấu hình máy.

Đổi cổng hoặc thư mục lưu:

```powershell
python src/dashboard.py --port 8502 --output-dir runs/demo
```

Mặc định `runs/live/` lưu video kết quả, clip bằng chứng, ảnh thumbnail, `history.sqlite` và nhật ký. Các file này không tự xóa.

## Chạy nhận diện bằng terminal (tùy chọn)

```powershell
python src/inference.py --source 0 --model m1 --weights results/m1/best.pt --device cpu
```

Thay `--source 0` bằng chỉ số camera khác hoặc đường dẫn video. Dùng `--threshold 0.5` để đặt ngưỡng, `--no-voice` để tắt giọng nói, `--save output.mp4` để ghi video có kết quả. Nhấn `q` trong cửa sổ video để thoát. Xem toàn bộ tùy chọn bằng `python src/inference.py --help`.

## Huấn luyện và đánh giá

Huấn luyện và đánh giá trên **Kaggle GPU** bằng `notebooks/kaggle_train.ipynb`;
local chỉ cần checkpoint để chạy dashboard. Notebook dùng nhóm dependency huấn luyện,
không ép phiên bản PyTorch/CUDA của local lên Kaggle, và lưu phiên bản thực tế vào
`results/environment.json` cùng kết quả. Khi hoàn tất, tải `results.zip` và giải nén vào `results/`.

Nếu sửa mã huấn luyện hoặc dependency, cài `requirements/notebooks.txt` rồi sinh lại notebook:

```powershell
python tools/build_kaggle_notebook.py
```

Các lệnh bên dưới mô tả quy trình mà notebook thực hiện; không cần chạy lại trên máy local.

Chuẩn bị [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) theo cấu trúc:

```text
RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

Chuẩn bị dữ liệu và huấn luyện từng model:

```powershell
python src/prepare_data.py --root D:/RWF-2000 --frames data/frames
python src/train_cnn.py
python src/train_yolo_cls.py
python src/crop_persons.py --weights yolo11n.pt
python src/train_cnn_lstm.py
```

Chỉ huấn luyện model cần sử dụng; M3 cần chạy crop trước. GPU được khuyến nghị cho huấn luyện. Dữ liệu được chia theo video; thư mục `val` gốc của RWF-2000 được giữ làm tập test.

Đánh giá sau khi chốt model; ngưỡng được chọn trên validation theo F2, các chỉ số được tính ở cấp clip trên test:

```powershell
python src/evaluate.py --model m1
python src/evaluate.py --model m2
python src/evaluate.py --model m3 --yolo-weights yolo11n.pt
```

Checkpoint và báo cáo nằm trong `results/m1/`, `results/m2/`, `results/m3/`; bảng so sánh được ghi vào `results/comparison.csv`.

## Kiểm thử

```powershell
python -m pip install -r requirements/test.txt
python -m pytest
```

Mặc định bỏ qua smoke test. Chạy toàn bộ luồng trên dữ liệu tổng hợp bằng `python -m pytest -m smoke`; lần đầu có thể cần tải trọng số YOLO. Smoke test kiểm tra luồng chạy, không đo độ chính xác trên dữ liệu thực.

## Xử lý lỗi thường gặp

- **Không mở được camera:** đóng ứng dụng khác đang dùng camera, kiểm tra quyền Camera của Windows và thử chỉ số camera khác. Với CLI, có thể thử `--camera-backend dshow` hoặc `msmf`.
- **Thiếu checkpoint:** kiểm tra đường dẫn `results/<model>/best.pt`; M3 cần thêm `yolo11n.pt` tại thư mục gốc.
- **Không thấy giao diện mới:** khởi động lại dashboard và nhấn `Ctrl+F5` trên trình duyệt.

## Cấu trúc mã dashboard

- `src/dashboard.py`: HTTP API, dữ liệu và điều khiển tác vụ.
- `src/templates/dashboard.html`: bố cục giao diện.
- `src/assets/dashboard.css`: kiểu dáng; `detection.js`, `camera.js`, `evidence.js`: các luồng giao diện.
- `src/live_camera.py`, `src/inference.py`, `src/monitoring.py`: camera, nhận diện và lưu bằng chứng.
