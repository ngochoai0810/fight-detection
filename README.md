# 🎯 Phát hiện Hành vi Đánh nhau qua Video Giám sát

**Đề tài 19:** Phát hiện hành vi bất thường/khả nghi trong video giám sát an ninh  
**Nhóm thực hiện:** Hoài, Ngọc  
**Ngôn ngữ:** Python, PyTorch  
**Phiên bản:** 1.0.0 (Hoàn thành)

## 📋 Mô tả dự án

Hệ thống phân loại tự động các clip video có chứa hành vi **đánh nhau** hoặc **không đánh nhau** sử dụng kỹ thuật deep learning. Dự án huấn luyện và so sánh ba mô hình với độ phức tạp tăng dần, từ CNN baseline đến YOLO11 kết hợp LSTM để học thông tin thời gian.

**Phạm vi:** Chỉ phát hiện đánh nhau. Té ngã, xâm nhập vùng cấm nằm ngoài phạm vi dự án.

### 🏆 Kết quả chính

| Mô hình | Độ chính xác | Recall | Precision | ROC-AUC | FPS |
|---------|-------------|--------|-----------|---------|-----|
| M1: CNN Baseline | 68.0% | **99%** | 61.1% | 0.843 | **378** |
| M2: YOLO11-cls | 72.25% | 98% | 64.7% | 0.832 | 87.5 |
| **M3: YOLO11+LSTM** | **73.25%** | 92.5% | **66.79%** | **0.866** ⭐ | 92.1 |

**Mô hình tốt nhất:** M3 với ROC-AUC **0.866**, đạt hiệu suất cao nhất trong phân biệt giữa hai lớp.

---

## 📊 Bộ dữ liệu (RWF-2000)

### Nguồn
[RWF-2000: Real World Fighting Video Database](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) - ICPR 2020

**Tham khảo:** Cheng, M., Cai, Z., Li, B. (2020). "RWF-2000: An Open Large Scale Video Database for Violence Detection". In *International Conference on Pattern Recognition (ICPR)*.

### Thống kê dữ liệu

```
Tổng cộng: 2,000 clip video
├── Train:      1,392 clip (70%)
│   ├── Fight:     695
│   └── NonFight:  697
├── Validation:   208 clip (10%)
│   ├── Fight:     105
│   └── NonFight:  103
└── Test:         400 clip (20%)
    ├── Fight:     200
    └── NonFight:  200
```

### Đặc điểm
- **Độ dài:** 5 giây/clip (150 frame ở 30 FPS)
- **Độ phân giải:** 640×360 đến 1280×720
- **Tỷ lệ lớp:** Cân bằng 1:1
- **Chia tập:** Theo video (không chia frame để tránh rò rỉ dữ liệu)

### ⚠️ Lưu ý pháp lý

RWF-2000 yêu cầu xin phép từ **SMIIP Lab** trước khi sử dụng hoặc phân phối lại. 

**Các tùy chọn:**
1. **Gửi form xin phép:** https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
2. **Sử dụng Kaggle mirror:** `vulamnguyen/rwf2000` (nhanh hơn, đã được cấp phép)

Dữ liệu không được commit lên repository này.

---

## 🧠 Các mô hình

### M1: CNN Baseline (ResNet18)

**Kiến trúc:**
```
16 frames/clip → Resize 224×224 → ResNet18 (pretrained) → MLP head → Fight/NonFight
```

**Đặc điểm:**
- Backbone pretrained trên ImageNet
- Xử lý từng frame độc lập
- Dự đoán cuối: trung bình xác suất các frame

**Kết quả:**
- Accuracy: 68.0%
- Recall: **99%** (bắt được gần tất cả fight)
- Precision: 61.1%
- **FPS: 378** (nhanh nhất) ⚡
- Model size: 44.8 MB

**Ưu điểm:** Nhanh, đơn giản  
**Nhược điểm:** Precision thấp (nhiều cảnh báo sai), không học thông tin thời gian

---

### M2: YOLO11-cls (Phân loại)

**Kiến trúc:**
```
16 frames/clip → YOLO11-cls head → Fight/NonFight (từng frame)
                                ↓
                          Aggregate theo clip
```

**Đặc điểm:**
- YOLOv11 nano trong chế độ classification
- Xử lý frame độc lập
- Model nhỏ gọn (3.2 MB)

**Kết quả:**
- Accuracy: 72.25%
- Recall: 98% (gần tuyệt đối)
- Precision: 64.7%
- FPS: 87.5
- Model size: **3.2 MB** (nhỏ nhất)

**Ưu điểm:** Model rất nhẹ, dễ deploy  
**Nhược điểm:** Không học thời gian

---

### M3: YOLO11 + CNN+LSTM ⭐ (Mô hình cải tiến)

**Kiến trúc:**
```
Video clip
    ↓
[YOLO11 Detection] → Crop vùng người (16 frames)
    ↓
[ResNet18 CNN] → Extract features (16 × 512)
    ↓
[2-layer LSTM] → Học thông tin thời gian (256 units)
    ↓
[FC layer] → Fight/NonFight
```

**Kỹ thuật cải tiến:**
- **Crop người:** Sử dụng YOLO11 phát hiện người, crop vùng có người → loại bỏ nền
- **Temporal learning:** LSTM học sự thay đổi theo thời gian (chuyển động)
- **Freeze backbone:** Giữ trọng số ResNet pretrained, chỉ train LSTM

**Kết quả:**
- Accuracy: **73.25%** ✓
- Recall: 92.5% (185/200 clip)
- Precision: **66.79%** ✓
- **ROC-AUC: 0.866** ⭐ (tốt nhất)
- FPS: 92.1 (đủ thời gian thực)
- Model size: 47.9 MB

**Ưu điểm:** 
- ROC-AUC cao nhất (phân biệt tốt)
- Học được chuyển động
- Cân bằng precision/recall

**Nhược điểm:**
- Model lớn hơn (47.9 MB)
- Bỏ sót 15 clip (bạn chuyển động chậm, nhân vật nhỏ)

---

## 📈 Đánh giá

### Chỉ số
- **Accuracy:** Tỷ lệ dự đoán đúng
- **Precision:** Trong những clip dự đoán là Fight, bao nhiêu % đúng
- **Recall:** Trong tất cả clip Fight, bắt được bao nhiêu %
- **F1-Score:** Cân bằng Precision và Recall
- **ROC-AUC:** Diện tích dưới đường cong ROC (0-1, cao hơn tốt hơn)

### Đặc điểm đánh giá
- **Cấp độ:** Clip-level (không phải frame-level)
- **Tập test:** 400 clip (200 Fight + 200 NonFight)
- **Threshold:** Chọn trên validation bằng F2-score (ưu tiên Recall)
- **Early stopping:** 
  - M1, M3: Dừng theo F1 cấp clip
  - M2: Dừng theo top-1 cấp frame (Ultralytics)

### Phân tích lỗi M3 (mô hình tốt nhất)
- **15 False Negative:** Bỏ sót Fight
  - Clip có nhân vật nhỏ
  - Chuyển động chậm/tinh tế
  
- **92 False Positive:** Cảnh báo sai
  - NonFight với chuyển động nhanh (ví dụ: thể dục)
  - Nền phức tạp

---

## 📁 Cấu trúc thư mục

```
fight-detection/
├── src/                      # Mã nguồn chính
│   ├── prepare_data.py       # Chuẩn bị & chia dữ liệu
│   ├── eda.py                # Phân tích dữ liệu (EDA)
│   ├── train_cnn.py          # Huấn luyện M1 (CNN baseline)
│   ├── train_yolo_cls.py     # Huấn luyện M2 (YOLO11-cls)
│   ├── train_cnn_lstm.py     # Huấn luyện M3 (YOLO11+LSTM)
│   ├── crop_persons.py       # Crop vùng người (M3 preprocessing)
│   ├── evaluate.py           # Đánh giá mô hình
│   ├── metrics.py            # Tính toán chỉ số
│   ├── models.py             # Định nghĩa kiến trúc
│   ├── fight_data.py         # DataLoader, augmentation
│   └── train_utils.py        # Utilities
├── tests/                    # Unit tests
│   ├── test_prepare_data.py
│   ├── test_metrics.py
│   ├── test_eda.py
│   ├── test_crop_persons.py
│   ├── test_torch_parts.py
│   ├── test_smoke_pipeline.py # Smoke test (end-to-end)
│   └── conftest.py           # Pytest config
├── notebooks/                # Jupyter notebooks
│   ├── eda.ipynb             # EDA interactif
│   └── kaggle_train.ipynb    # Notebook chạy trên Kaggle
├── configs/                  # Cấu hình
├── data/                     # Dữ liệu (không commit)
├── splits/                   # Danh sách clip train/val/test (CSV)
├── eda/                      # Output EDA (biểu đồ, thống kê)
├── m1/, m2/, m3/             # Output huấn luyện (checkpoint, metrics)
│
├── RESULTS.md                # Báo cáo kết quả chi tiết
├── README.md                 # File này
├── requirements.txt          # Dependencies
└── pytest.ini                # Pytest config
```

---

## ⚙️ Yêu cầu & Cài đặt

### Yêu cầu
- Python 3.8+
- GPU (khuyến nghị: CUDA 11.8+)
- Dung lượng: ~50 GB (RWF-2000 + frames + models)

### Cài đặt

```bash
# Clone repository
git clone https://github.com/ngochoai0810/fight-detection.git
cd fight-detection

# Cài đặt dependencies
pip install -r requirements.txt
```

**Lưu ý:** Nên sử dụng **GPU** (Kaggle hoặc Colab) để huấn luyện. CPU chỉ phù hợp cho debugging.

---

## 🚀 Hướng dẫn chạy

### 1️⃣ Chuẩn bị dữ liệu

**Tùy chọn A: Kaggle (Khuyến nghị)**
```bash
# Chạy notebook Kaggle: notebooks/kaggle_train.ipynb
# Hoặc tải dữ liệu từ: https://www.kaggle.com/datasets/vulamnguyen/rwf2000
```

**Tùy chọn B: Local**
```bash
# Tải RWF-2000 (cần xin phép hoặc dùng Kaggle mirror)
# Giải nén theo cấu trúc:
# RWF-2000/
# ├── train/
# │   ├── Fight/      *.avi
# │   └── NonFight/   *.avi
# └── val/
#     ├── Fight/      *.avi
#     └── NonFight/   *.avi

python src/prepare_data.py --root D:/RWF-2000 --frames data/frames
```

**Output:**
- `splits/splits.csv` — Danh sách clip với split (train/val/test)
- `data/frames/` — Frames được trích xuất

### 2️⃣ Phân tích dữ liệu (EDA)

```bash
python src/eda.py --root D:/RWF-2000 --yolo-weights yolo11n.pt --out results/eda
```

**Output:**
- `eda/summary.csv` — Thống kê chung
- `eda/by_label.csv` — Thống kê theo lớp
- `eda/samples.png` — Grid mẫu clip
- `eda/motion.png` — Biểu đồ optical flow (tính chuyển động)

Xem chi tiết: `notebooks/eda.ipynb`

### 3️⃣ Huấn luyện mô hình

#### M1: CNN Baseline
```bash
python src/train_cnn.py \
  --frames data/frames \
  --epochs 30 \
  --image-size 224 \
  --batch-size 32
```

#### M2: YOLO11-cls
```bash
python src/train_yolo_cls.py \
  --frames data/frames \
  --epochs 30 \
  --image-size 224 \
  --batch-size 64
```

#### M3: YOLO11 + LSTM
```bash
# Bước 1: Crop vùng người
python src/crop_persons.py \
  --frames data/frames \
  --out data/crops \
  --size 224

# Bước 2: Huấn luyện
python src/train_cnn_lstm.py \
  --frames data/crops \
  --epochs 30 \
  --image-size 224 \
  --hidden-size 128 \
  --freeze-backbone
```

**Output mỗi mô hình:** `results/m1/`, `results/m2/`, `results/m3/`
- `best.pt` — Checkpoint tốt nhất
- `history.csv` — Loss/metrics theo epoch
- `metrics.json` — Metrics cuối cùng

### 4️⃣ Đánh giá trên Test Set

⚠️ **Chỉ chạy sau khi chốt xong mô hình** (dùng tập test)

```bash
python src/evaluate.py --model m1 --image-size 224
python src/evaluate.py --model m2 --image-size 224
python src/evaluate.py --model m3 --frames data/crops --yolo-weights yolo11n.pt --image-size 224
```

**Output:**
- `results/m{1,2,3}/metrics.json` — Chỉ số (accuracy, precision, recall, etc.)
- `results/m{1,2,3}/preds_test.csv` — Dự đoán từng clip
- `results/m{1,2,3}/errors.csv` — Phân tích lỗi (FP, FN)
- `results/m{1,2,3}/confusion_matrix.png` — Confusion matrix
- `results/m{1,2,3}/roc.png` — ROC curve
- `results/comparison.csv` — **Bảng so sánh 3 mô hình** ⭐

### 5️⃣ Kiểm thử (Testing)

```bash
# Chạy tất cả test
python -m pytest

# Chạy test nhanh (loại bỏ smoke test)
python -m pytest -m "not smoke"

# Chạy smoke test (end-to-end trên dữ liệu tổng hợp)
python -m pytest -m smoke
```

**Lưu ý:** Smoke test chứng minh quy trình chạy được, không đo độ chính xác.

---

## 📊 Kết quả chi tiết

Xem đầy đủ tại **[RESULTS.md](RESULTS.md)**

Tóm tắt:

| Mô hình | Accuracy | Recall | Precision | ROC-AUC | FPS | Model |
|---------|----------|--------|-----------|---------|-----|-------|
| **M1** | 68.0% | 99% | 61.1% | 0.843 | 378 | 44.8 MB |
| **M2** | 72.25% | 98% | 64.7% | 0.832 | 87.5 | 3.2 MB |
| **M3** ⭐ | 73.25% | 92.5% | 66.79% | 0.866 | 92.1 | 47.9 MB |

---

## 🔍 Quy ước & Lưu ý

1. **Chia dữ liệu:** Theo video, không theo frame (tránh rò rỉ)
2. **Threshold:** Chọn trên validation bằng F2-score (ưu tiên Recall)
3. **Early stopping:**
   - M1, M3: Dừng theo F1 cấp clip
   - M2: Dừng theo top-1 cấp frame (Ultralytics)
4. **Clip cùng nguồn:** Không tách giữa train và val

---

## 📚 Tài liệu tham khảo

1. **RWF-2000 Dataset**
   - Cheng, M., Cai, Z., Li, B. (2020). "RWF-2000: An Open Large Scale Video Database for Violence Detection". *ICPR 2020*
   - GitHub: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

2. **YOLOv11**
   - Ultralytics. "YOLOv11: A New Generation of State-of-the-Art Real-Time Detectors"
   - Docs: https://docs.ultralytics.com/

3. **Two-Stream CNN**
   - Wang, L., et al. (2016). "Two-Stream Convolutional Networks for Action Recognition in Videos". *CVPR 2016*

---

## 📝 License & Attribution

**Dữ liệu RWF-2000:** Yêu cầu xin phép từ SMIIP Lab trước khi sử dụng/phân phối  
**Mã nguồn:** MIT License (xem [LICENSE](LICENSE) nếu có)

---

## 👥 Thông tin liên hệ

**Nhóm thực hiện:** Hoài, Ngọc  
**Repository:** https://github.com/ngochoai0810/fight-detection
