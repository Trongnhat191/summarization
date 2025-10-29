# Các Tập Dữ Liệu Quan Trọng Để Fine-Tuning FLAN-T5-Large Cho Tóm Tắt (Summarization)

---

## 1. Các Tập Dữ Liệu Đa Ngôn Ngữ (Multilingual Datasets)

| Dataset | Ngôn Ngữ Chính | Đặc Điểm | Phù Hợp Cho | Link |
| :--- | :--- | :--- | :--- | :--- |
| **XL-Sum** | 44 ngôn ngữ (Bao gồm Trung, Nhật, Tây Ban Nha, và **Việt**) | Tập dữ liệu **tóm tắt trừu tượng** (Abstractive Summarization) quy mô lớn (~1.35 triệu cặp bài báo-tóm tắt) được thu thập từ BBC News. | **Fine-tuning giai đoạn đầu** để củng cố khả năng tóm tắt đa ngôn ngữ (M2MS - Many-to-Many Summarization), đặc biệt cho các ngôn ngữ mục tiêu như Trung Quốc (zh-CN, zh-TW) và Nhật Bản (ja). | [Link XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) |
| **WikiLingua** | Đa ngôn ngữ | Được tạo ra để nghiên cứu các phương pháp **tóm tắt chéo ngôn ngữ** và **đa ngôn ngữ**. | Cung cấp nguồn dữ liệu đa dạng về chủ đề và ngôn ngữ để tăng cường khả năng chuyển giao kiến thức (transferability) của mô hình T5. | [Link WikiLingua](https://huggingface.co/datasets/GEM/wiki_lingua) |

---

## 2. Các Tập Dữ Liệu Tiếng Việt Chuyên Biệt (Vietnamese Datasets)

| Dataset | Đặc Điểm | Nguồn Dữ Liệu | Số Lượng (Xấp xỉ) | Link |
| :--- | :--- | :--- | :--- | :--- |
| **VietNews-Abs-Sum** (còn gọi là **VNDS**) | Dành cho **tóm tắt trừu tượng** tiếng Việt (Vietnamese Abstractive Summarization). | Thu thập từ các báo điện tử lớn của Việt Nam như *tuoitre.vn*, *vnexpress.net*, và *nguoiduatin.vn*. | Hơn **105.000** mẫu huấn luyện. | [Link VietNews-Abs-Sum](https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum) |
| **Vietnews** | Tập dữ liệu **tóm tắt trừu tượng** tiếng Việt, được xây dựng để cải thiện mô hình tóm tắt T5/BRIO. | Cung cấp tài nguyên chất lượng cao cho fine-tuning. | Dữ liệu tiêu chuẩn cho nghiên cứu tóm tắt tiếng Việt. | [Link Vietnews](https://huggingface.co/datasets/nam194/vietnews) |
| **Vietnamese Multiple Document Summarization (MDS)** | Tập dữ liệu **tóm tắt đa văn bản** tiếng Việt (Multi-Document Summarization). | Thu thập từ Baomoi và Google News, bao gồm 300 cụm tin tức được tóm tắt thủ công. | **300** cụm tài liệu. | [Link Vietnamese MDS](https://www.kaggle.com/datasets/vtrnanh/sust-feature-data-new) |

---
