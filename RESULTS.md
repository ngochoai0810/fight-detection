# Kết quả phát hiện đánh nhau qua video giám sát

Dự án so sánh M1 (ResNet18), M2 (YOLO11 classification) và M3 (YOLO crop người → ResNet18 → LSTM) trên RWF-2000.

## Dữ liệu và cách đánh giá

Thống kê từ `splits.csv`; bộ `results/splits.csv` cục bộ có cùng số lượng theo lớp và tập:

| Tập | Fight | NonFight | Tổng |
| --- | ---: | ---: | ---: |
| Train | 695 | 697 | 1.392 |
| Validation | 105 | 103 | 208 |
| Test | 200 | 200 | 400 |

- Đánh giá ở cấp clip, với Fight là lớp dương.
- Chọn ngưỡng trên validation theo F2 để ưu tiên Recall; tính chỉ số cuối trên test.
- M1/M2 tổng hợp xác suất từng frame; M3 xử lý chuỗi frame đã crop vùng người.
- Giữ tập `val` gốc làm test, chia train/validation theo nhóm nguồn video. Mã chuẩn bị dữ liệu có cảnh báo nguồn trùng với test; việc chia theo video không tự bảo đảm không có nguồn trùng.

## Bộ kết quả đã lưu trong Git

Nguồn: `comparison.csv`, `m1/metrics.json`, `m2/metrics.json`, `m3/metrics.json`. Số TP/FP/FN/TN đã được đối chiếu với `preds_test.csv` trong từng thư mục.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 68,00% | 61,11% | 99,00% | 0,756 | 0,843 | 126 | 2 |
| M2 | 72,25% | 64,69% | 98,00% | 0,779 | 0,832 | 107 | 4 |
| M3 | 73,25% | 66,79% | 92,50% | 0,776 | 0,866 | 92 | 15 |

Trong bộ này, M3 có Accuracy, Precision và ROC-AUC cao nhất; M2 có F1 cao nhất; M1 có Recall cao nhất. Không có model dẫn đầu mọi chỉ số.

## Bộ kết quả cục bộ trong `results/`

Nguồn: `results/comparison.csv` và `results/m{1,2,3}/metrics.json`. Số TP/FP/FN/TN cũng khớp với các file `preds_test.csv` tương ứng. Thư mục này được bỏ qua bởi Git nên không có sẵn khi clone dự án.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 78,75% | 70,91% | 97,50% | 0,821 | 0,852 | 80 | 5 |
| M2 | 72,25% | 64,69% | 98,00% | 0,779 | 0,832 | 107 | 4 |
| M3 | 70,00% | 62,74% | 98,50% | 0,767 | 0,876 | 117 | 3 |

Trong bộ này, M1 có Accuracy, Precision và F1 cao nhất; M3 có Recall và ROC-AUC cao nhất. Cần xác định checkpoint và cấu hình tương ứng trước khi chọn một bộ làm kết quả báo cáo chính thức.

## Ngưỡng và tốc độ được ghi nhận

Các file metrics ghi thiết bị Tesla T4. Bảng dưới giữ riêng số đo của từng bộ:

| Model | Ngưỡng · Git | FPS · Git | Ngưỡng · Cục bộ | FPS · Cục bộ |
| --- | ---: | ---: | ---: | ---: |
| M1 | 0,002001 | 378,0 | 0,045277 | 341,6 |
| M2 | 0,079018 | 87,5 | 0,079018 | 79,4 |
| M3 | 0,331839 | 92,1 | 0,121257 | 81,9 |

FPS benchmark không phải FPS toàn tuyến của camera/dashboard. Kết quả thực tế còn phụ thuộc thiết bị, model, stride, độ phân giải và bước khoanh vùng/ghi bằng chứng. Chưa có số đo webcam thật được xác nhận trong báo cáo này.

## Hạn chế và trạng thái tích hợp

- Cả hai bộ đều có nhiều cảnh báo nhầm trên 200 clip NonFight. Việc ưu tiên Recall cần được cân nhắc cùng Precision và số FP.
- Ngưỡng cho clip test không mặc nhiên phù hợp camera thực tế; cần kiểm tra trên dữ liệu độc lập gần với môi trường sử dụng.
- Đối chiếu CSV xác nhận tính nhất quán của các chỉ số, chưa xác nhận checkpoint hiện có tái lập được toàn bộ kết quả. File `dummy_best.pt` không dùng để chứng minh chất lượng model.
- Dashboard đã có phân tích video, camera trực tiếp, lưu bằng chứng, duyệt cảnh báo và xác thực SHA-256. Luồng camera đã kiểm thử bằng dữ liệu mô phỏng; chưa xác nhận hiệu năng webcam thật.

Hướng dẫn cài đặt, chạy, huấn luyện và đánh giá nằm trong [README.md](README.md).
