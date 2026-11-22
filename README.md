# Phân loại cảm xúc phản hồi sinh viên trên UIT-VSFC

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/volethanhbinh/ml-uit-vsfc/blob/main/notebooks/main_uit_vsfc_pipeline.ipynb)

## 1. Mục tiêu bài tập lớn & Bộ dữ liệu UIT-VSFC

### 1.1. Mục tiêu
Dự án xây dựng một pipeline Học máy và Học sâu hoàn chỉnh, có khả năng tái lập (`seed=42`) cho bài toán **phân loại cảm xúc văn bản tiếng Việt 3 nhãn** trên dữ liệu phản hồi sinh viên:
1. Thực hiện **Phân tích khám phá dữ liệu (EDA)** định lượng và trực quan hóa độ dài câu, phân phối nhãn mất cân bằng và các token nhiễu đặc thù.
2. Xây dựng bộ **tiền xử lý văn bản tiếng Việt cấu hình được** (`VietnameseTextPreprocessor`), khảo sát thực nghiệm (ablation study) tác động của tách từ tiếng Việt (`pyvi.ViTokenizer`) và lọc stopwords có bảo toàn từ phủ định/chuyển ý.
3. So sánh hệ thống **3 nhóm mô hình**:
   - **Pipeline ML truyền thống:** Bag-of-Words (`CountVectorizer`) và `TF-IDF` (Unigram/Bigram) kết hợp `MultinomialNB`, `LogisticRegression`, `LinearSVC` (có đối chiếu `class_weight='balanced'`).
   - **Pipeline Lai (Hybrid):** Trích xuất đặc trưng ngữ nghĩa bằng `vinai/phobert-base-v2` (Mean Pooling trên GPU, lưu thành `features/*.npy`), sau đó nạp lại để huấn luyện `LogisticRegression`, `LinearSVC` và `MLP (256-64)`.
   - **Pipeline Học sâu End-to-End (Mở rộng +5%):** Fine-tuning trực tiếp `vinai/phobert-base-v2` với `AdamW`, Linear Warmup Scheduler và `CrossEntropyLoss` có trọng số lớp.
4. Đánh giá bằng thước đo chính **Macro F1-score** (phù hợp với dữ liệu mất cân bằng), trực quan hóa **Confusion Matrix** và thực hiện **Phân tích lỗi (Error Analysis)**.

### 1.2. Tập dữ liệu UIT-VSFC
- **Nguồn công khai:** [`uitnlp/vietnamese_students_feedback`](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) trên Hugging Face Hub.
- **Nhãn phân loại (3 lớp):**
  - `0` — **Tiêu cực**
  - `1` — **Trung tính** *(lớp thiểu số, chỉ chiếm khoảng ~5% toàn bộ tập dữ liệu — nguyên nhân chính đòi hỏi dùng Macro F1 và chiến lược xử lý mất cân bằng)*
  - `2` — **Tích cực**
- **Đặc điểm văn bản:** Câu phản hồi ngắn của sinh viên về giảng viên và môn học, chứa một số ký tự/token mã hóa đặc trưng của bộ UIT-VSFC như `colon`, `dot`, `doublespace`, `fraction`, `wzjwz...`.

---

## 2. Cấu trúc thư mục dự án

```text
ml-uit-vsfc/
├── features/                       # Chứa các file embedding PhoBERT (.npy) sau khi trích xuất
│   ├── .gitkeep
│   ├── phobert_train.npy           # (Sinh tự động khi chạy Notebook)
│   ├── phobert_val.npy             # (Sinh tự động khi chạy Notebook)
│   └── phobert_test.npy            # (Sinh tự động khi chạy Notebook)
├── modules/                        # Các module Python lõi được import vào Notebook
│   ├── __init__.py
│   ├── data_loader.py              # Tải UIT-VSFC qua Hugging Face (parquet + fallback)
│   ├── eda.py                      # Phân tích EDA & xuất biểu đồ chất lượng cao (300 DPI)
│   ├── preprocessing.py            # VietnameseTextPreprocessor cấu hình bật/tắt linh hoạt
│   ├── feature_extraction.py       # BoW/TF-IDF & trích xuất/lưu/đọc PhoBERT Mean Pooling (.npy)
│   ├── models.py                   # Pipeline ML truyền thống, Lai (.npy) & PhoBERT Fine-tuning
│   └── evaluation.py               # Macro F1, Classification Report, Confusion Matrix & Error Analysis
├── notebooks/
│   └── main_uit_vsfc_pipeline.ipynb # Notebook chính chạy một mạch từ đầu đến cuối trên Colab
├── reports/                         # Thư mục báo cáo và biểu đồ thực nghiệm
│   ├── figures/                     # Các biểu đồ EDA, Model Comparison & Confusion Matrix (.png, 300 DPI)
│   │   ├── .gitkeep
│   │   ├── label_distribution.png
│   │   ├── length_distribution.png
│   │   ├── special_tokens_frequency.png
│   │   ├── top_unigrams_by_label.png
│   │   ├── top_bigrams_by_label.png
│   │   ├── model_comparison.png
│   │   ├── confusion_matrix_traditional_best.png
│   │   └── confusion_matrix_best.png
│   ├── report.tex                   # Mã nguồn LaTeX báo cáo khoa học
│   └── report.pdf                   # File báo cáo PDF đã biên dịch
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 3. Hướng dẫn chạy thực nghiệm

### 3.1. Chạy trên Google Colab (Khuyến nghị — `Runtime -> Run all`)
Notebook được thiết kế để chạy độc lập trên môi trường Google Colab sạch (**không mount Google Drive hay dịch vụ cloud cá nhân**):
1. Mở liên kết Colab: [Chạy `main_uit_vsfc_pipeline.ipynb` trên Google Colab](https://colab.research.google.com/github/volethanhbinh/ml-uit-vsfc/blob/main/notebooks/main_uit_vsfc_pipeline.ipynb).
2. Chọn `Runtime` -> `Run all`.
3. Notebook sẽ tự động:
   - Clone repository `https://github.com/volethanhbinh/ml-uit-vsfc.git` và thêm vào `sys.path`.
   - Cài đặt đầy đủ các gói phụ thuộc.
   - Tải dữ liệu UIT-VSFC công khai từ Hugging Face.
   - Chạy lần lượt EDA, Thí nghiệm 1 (Ablation tiền xử lý), Thí nghiệm 2 (ML truyền thống), Thí nghiệm 3 (trích xuất và lưu `features/phobert_*.npy` + mô hình lai), Thí nghiệm 4 (Fine-tuning PhoBERT End-to-End), và xuất bảng tổng hợp, biểu đồ và mẫu phân tích lỗi.

