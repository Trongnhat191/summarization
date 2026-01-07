# Vietnamese Text Summarization with LLMs (Tóm tắt văn bản tiếng Việt)

Dự án này tập trung nghiên cứu, huấn luyện (fine-tune) và triển khai các mô hình ngôn ngữ lớn (LLM) để thực hiện tác vụ tóm tắt văn bản tiếng Việt. Dự án bao gồm quy trình từ chuẩn bị dữ liệu, huấn luyện các mô hình khác nhau (BART, FLAN-T5, Qwen) đến xây dựng giao diện demo tương tác.

## 📂 Cấu trúc Dự án

Repository này bao gồm 6 file chính, được chia thành các nhóm chức năng sau:

### 1. Chuẩn bị Dữ liệu (Data Preparation)
* **`datasets_suggestion.md`**: Tài liệu tổng hợp và đánh giá các bộ dữ liệu (dataset) tiềm năng cho bài toán tóm tắt văn bản, bao gồm các nguồn đa ngôn ngữ (như XL-Sum, WikiLingua) và các nguồn tiếng Việt chuyên biệt (VietNews-Abs-Sum).
* **`gen_data.py`**: Script Python dùng để sinh dữ liệu hoặc chuẩn hóa dữ liệu huấn luyện. Script này sử dụng API (ví dụ: GPT qua FPT Marketplace) để tóm tắt lại các bài báo, tạo ra cặp dữ liệu `Document` - `Summary` chất lượng cao, phục vụ cho quá trình fine-tuning.

### 2. Huấn luyện Mô hình (Model Fine-tuning)
Mỗi notebook dưới đây thực hiện quy trình fine-tuning cho một kiến trúc mô hình cụ thể:
* **`fine_tune_bart.ipynb`**: Notebook huấn luyện mô hình **BART**. BART là mô hình Seq2Seq kinh điển, hoạt động tốt cho các tác vụ tóm tắt.
* **`fine_tune_flan.ipynb`**: Notebook huấn luyện mô hình **FLAN-T5** (phiên bản XL). Đây là mô hình Encoder-Decoder mạnh mẽ của Google, có khả năng zero-shot/few-shot tốt.
* **`fine_tune_qwen.ipynb`**: Notebook huấn luyện mô hình **Qwen** (sử dụng kỹ thuật PEFT/LoRA). Qwen là dòng mô hình Decoder-only hiện đại với hiệu năng cao trên tiếng Việt.

### 3. Giao diện Demo (Interface)
* **`interface.ipynb`**: Notebook chứa mã nguồn xây dựng giao diện người dùng (GUI) bằng thư viện **Gradio**. 
    * Cho phép người dùng nhập văn bản hoặc URL.
    * Tùy chọn mô hình tóm tắt (Model Selector).
    * Tùy chỉnh tham số (Độ dài tóm tắt, Beam size).
    * Hiển thị kết quả tóm tắt trực quan.

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu hệ thống
* Python 3.10+
* GPU (Khuyến nghị NVIDIA T4/A100 nếu chạy training hoặc demo các mô hình lớn).

### Cài đặt thư viện
Bạn cần cài đặt các thư viện cần thiết được sử dụng trong các notebook (xem chi tiết trong từng file), ví dụ:
```bash
pip install torch transformers datasets peft trl gradio openai python-dotenv newspaper3k
