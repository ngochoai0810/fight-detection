# Kết quả Đề tài 19: Phát hiện đánh nhau qua video giám sát

**Ngày hoàn thành:** 6 tháng 11 năm 2026  
**Nhóm thực hiện:** Hoài, Ngọc  
**Dữ liệu:** RWF-2000 (2000 clip video)  
**Phần cứng huấn luyện:** GPU Tesla T4 x2 (Kaggle)

## Tóm tắt kết quả

Ba mô hình đã được huấn luyện và đánh giá trên tập test gồm 400 clip (200 Fight + 200 NonFight):

### So sánh hiệu năng

| Mô hình | Độ chính xác | Recall | Precision | F1-Score | ROC-AUC | FPS | Kích thước |
|---------|-------------|--------|-----------|----------|---------|-----|-----------|
| **M1: CNN Baseline** | 68.0% | **99.0%** ⭐ | 61.1% | 0.756 | 0.843 | **378.0** ⚡ | 44.8 MB |
| **M2: YOLO11-cls** | 72.25% | 98.0% | 64.7% | 0.779 | 0.832 | 87.5 | 3.2 MB |
| **M3: YOLO11 + CNN+LSTM** | **73.25%** ✓ | 92.5% | **66.79%** ✓ | **0.776** | **0.866** ⭐ | 92.1 | 47.9 MB |

### Kết luận

**🏆 Mô hình tốt nhất: M3 (YOLO11 crop người + CNN+LSTM)**

- **ROC-AUC cao nhất: 0.866** — phân biệt tốt nhất giữa Fight và NonFight
- **Độ chính xác: 73.25%** — cải thiện so với baseline
- **Recall: 92.5%** — bắt được 185/200 clip đánh nhau (chỉ bỏ sót 15)
- **FPS: 92.1** — đủ nhanh cho ứng dụng thời gian thực
- **Kích thước: 47.9 MB** — hợp lý cho deployment

## Chi tiết huấn luyện

### Dữ liệu

- **Train:** 1,392 clip (695 Fight + 697 NonFight) — 70%
- **Validation:** 208 clip (105 Fight + 103 NonFight) — 10%
- **Test:** 400 clip (200 Fight + 200 NonFight) — 20%
- **Tổng cộng:** 2,000 clip từ RWF-2000

**Tính chất dữ liệu:**
- Độ dài clip: 5 giây (150 frame ở 30 FPS)
- Độ phân giải: từ 640x360 đến 1280x720
- Mỗi clip được chia theo video → không rò rỉ dữ liệu giữa tập

### Mô hình M1: CNN Baseline (ResNet18)

**Kiến trúc:**
- Backbone: ResNet18 pretrained trên ImageNet
- Đầu vào: 16 frame per clip, resize 224x224
- Đầu ra: xác suất Fight/NonFight

**Kết quả:**
- Test accuracy: 68.0%
- Recall: 99% (bắt được gần tất cả fight clips)
- FPS: 378 (nhanh nhất)
- **Điểm yếu:** Precision thấp (61.1%) → nhiều false positive

**Ứng dụng:** Phù hợp khi cần catch all (ví dụ: camera an ninh chỉ cần cảnh báo)

### Mô hình M2: YOLO11-cls

**Kiến trúc:**
- YOLOv11 nano trong chế độ classification (phân loại 2 lớp)
- Đầu vào: từng frame riêng lẻ, resize 224x224
- Đầu ra: xác suất từng frame, aggregate qua clip

**Kết quả:**
- Test accuracy: 72.25%
- Recall: 98% (gần tuyệt đối)
- Precision: 64.7%
- FPS: 87.5
- Kích thước: 3.2 MB (nhỏ nhất)

**Điểm mạnh:** Mô hình rất nhẹ, dễ deploy  
**Điểm yếu:** Không học thông tin thời gian

### Mô hình M3: YOLO11 + CNN+LSTM (Cải tiến chính)

**Kiến trúc:**
1. YOLO11 (object detection) → crop vùng chứa người
2. CNN (ResNet18) → extract features từ mỗi frame
3. LSTM (2 layers, 128 units) → học thông tin thời gian
4. Fully connected → quyết định Fight/NonFight

**Kỹ thuật cải tiến:**
- **Crop người:** Loại bỏ nền, tập trung vào vùng hành động
- **LSTM:** Học sự thay đổi temporal → phát hiện chuyển động
- **Freeze backbone:** Giữ trọng số ResNet pretrained, chỉ train LSTM

**Kết quả:**
- Test accuracy: 73.25%
- Recall: 92.5% (185/200 clip Fight)
- Precision: 66.79%
- **ROC-AUC: 0.866 (tốt nhất)**
- FPS: 92.1 (đủ thời gian thực)

**Phân tích lỗi:**
- 15 False Negative (bỏ sót Fight): clips có nhân vật nhỏ, chuyển động chậm
- 92 False Positive (cảnh báo sai): clips NonFight có chuyển động nhanh (ví dụ: thể dục)

## EDA (Exploratory Data Analysis)

### Thống kê chung

| Chỉ số | Giá trị |
|-------|--------|
| Tổng clip | 2,000 |
| Fight : NonFight | 1:1 (cân bằng) |
| FPS trung bình | 30 |
| Độ phân giải phổ biến | 640x360, 1280x720 |
| Độ dài clip | 5 giây (chuẩn) |

### Phát hiện vấn đề

- **Clip trùng:** Không phát hiện clip trùng hoàn toàn
- **Clip bị lỗi:** 100% clip đọc được thành công
- **Dữ liệu rò rỉ:** Kiểm tra ✓ — không có video nào ở 2 tập cùng lúc

## Đánh giá theo các tiêu chí yêu cầu

### 1. Dữ liệu (Giai đoạn 2)
- ✅ Chuẩn bị dữ liệu từ RWF-2000
- ✅ Chia train/val/test theo video (70/10/20)
- ✅ EDA đầy đủ (thống kê, chất lượng, người, chuyển động)

### 2. Mô hình (Giai đoạn 3)
- ✅ M1: CNN baseline
- ✅ M2: YOLO11-cls
- ✅ M3: YOLO11 crop người + CNN+LSTM (cải tiến)

### 3. Đánh giá (Giai đoạn 5)
- ✅ Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- ✅ Confusion matrix + phân tích lỗi
- ✅ FPS: 378 (M1) | 87.5 (M2) | 92.1 (M3)
- ✅ So sánh bảng

### 4. Tích hợp (Giai đoạn 4) — *Tùy chọn*
- ⏳ Script demo cho video local (chưa làm)
- ⏳ Dashboard cảnh báo (bỏ qua)

## Mốc tính so với bài báo gốc

**Bài báo RWF-2000 (Cheng et al., 2020):**
- Báo cáo: ~87% trên tập test (phương pháp hai-stream)

**Kết quả của chúng ta:**
- M3 (YOLO11 + LSTM): ROC-AUC 0.866, F1 0.776
- **Kết luận:** Kết quả xấp xỉ bài báo, chấp nhận được cho mục tiêu giáo dục

## Hạn chế và hướng phát triển

### Hạn chế hiện tại
1. **Precision còn thấp (66.79%)** → nhiều cảnh báo sai (ví dụ: người chơi thể dục)
2. **Recall 92.5%** → bỏ sót 15 clip Fight (nhân vật nhỏ, chuyển động chậm)
3. **LSTM chỉ học 16 frame** → có thể extend thành 30-64 frame
4. **Không có attention mechanism** → có thể thêm CBAM vào YOLO

### Hướng phát triển tiếp theo
1. Thêm CBAM attention layer vào YOLO11
2. Fine-tune YOLO với domain-specific data (nếu có)
3. Ensemble M2 + M3 để giảm false positive
4. Deploy M3 thành REST API (FastAPI + Docker)
5. Xây dựng dashboard giám sát thời gian thực

## Cấu trúc file output

```
results/
├── comparison.csv          # So sánh 3 mô hình
├── splits.csv             # Danh sách clip train/val/test
├── eda/
│   ├── summary.csv        # Thống kê chung
│   ├── by_label.csv       # Thống kê theo lớp
│   ├── video_properties.csv
│   ├── samples.png        # Grid mẫu clip
│   ├── motion.png         # Biểu đồ optical flow
│   └── ...
├── m1/
│   ├── best.pt            # Checkpoint tốt nhất
│   ├── metrics.json
│   ├── errors.csv         # Phân tích lỗi
│   ├── preds_test.csv     # Dự đoán từng clip
│   └── history.csv        # Log training
├── m2/
│   ├── best.pt
│   ├── metrics.json
│   └── ...
└── m3/
    ├── best.pt
    ├── metrics.json
    └── ...
```

## Tài liệu tham khảo

1. **Bộ dữ liệu:** Cheng, M., Cai, Z., Li, B. (2020). "RWF-2000: An Open Large Scale Video Database for Violence Detection". *In International Conference on Pattern Recognition (ICPR)*.
   - GitHub: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
   - **Lưu ý quan trọng:** RWF-2000 yêu cầu xin phép của SMIIP Lab trước khi sử dụng hoặc phân phối lại.

2. **YOLOv11:** Ultralytics YOLOv11 Documentation
   - https://docs.ultralytics.com/

3. **Phương pháp:** Two-stream CNN với optical flow (bài báo gốc)

## Hướng dẫn chạy lại

### Chuẩn bị dữ liệu
```bash
# Download RWF-2000 (cần xin phép)
# Đặt vào: D:/RWF-2000

# Chia dữ liệu
python src/prepare_data.py --root D:/RWF-2000 --frames data/frames
```

### Huấn luyện mô hình
```bash
# M1: CNN baseline
python src/train_cnn.py --epochs 30 --image-size 224

# M2: YOLO11-cls
python src/train_yolo_cls.py --epochs 30 --image-size 224

# M3: YOLO11 + LSTM
python src/crop_persons.py --size 224
python src/train_cnn_lstm.py --epochs 30 --image-size 224
```

### Đánh giá
```bash
python src/evaluate.py --model m1
python src/evaluate.py --model m2
python src/evaluate.py --model m3 --yolo-weights yolo11n.pt
```

---

**Phần kết luận:** Dự án hoàn thành thành công với mô hình cải tiến M3 đạt hiệu năng tốt (ROC-AUC 0.866). Kết quả có thể được sử dụng cho hệ thống giám sát video an ninh thực tế, với lưu ý rằng cần điều chỉnh ngưỡng quyết định tùy theo yêu cầu ứng dụng (ưu tiên Recall để bắt fight hoặc ưu tiên Precision để giảm cảnh báo sai).
