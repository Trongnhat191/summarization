# Vietnamese Text Summarization with LLMs 🇻🇳

Dự án này tập trung vào việc nghiên cứu, huấn luyện (fine-tune) và so sánh hiệu quả của các mô hình ngôn ngữ lớn (LLM) khác nhau trong tác vụ **Tóm tắt văn bản tiếng Việt**. Dự án bao gồm quy trình khép kín từ khâu chuẩn bị dữ liệu, huấn luyện mô hình đến triển khai giao diện demo tương tác.

## 📂 Cấu trúc Repository

Repository bao gồm 6 file chính, được chia thành 3 nhóm chức năng:

### 1. Dữ liệu (Data Preparation)
* **`datasets_suggestion.md`**: Tài liệu tổng hợp các nguồn dữ liệu tóm tắt chất lượng cao (như *VietNews-Abs-Sum*, *XL-Sum*, *WikiLingua*) để định hướng cho việc huấn luyện.
* **`gen_data.py`**: Script Python dùng để sinh và làm sạch dữ liệu huấn luyện. 
    * Sử dụng API (ví dụ: GPT qua FPT Marketplace) để tạo ra các bản tóm tắt chuẩn (gold standard) từ các bài báo gốc.
    * Hỗ trợ xử lý đa luồng (multithreading) để tăng tốc độ sinh dữ liệu.
    * Kết quả được lưu thành file JSON để đưa vào các notebook huấn luyện.

### 2. Huấn luyện Mô hình (Model Fine-tuning)
Mỗi notebook tương ứng với một kiến trúc mô hình khác nhau được thử nghiệm:
* **`fine_tune_bart.ipynb`**: Huấn luyện mô hình **BART** (Bidirectional and Auto-Regressive Transformers). Đây là mô hình Seq2Seq kinh điển, nhẹ và hiệu quả cho tóm tắt.
* **`fine_tune_flan.ipynb`**: Huấn luyện mô hình **FLAN-T5** (phiên bản XL). Đây là mô hình Encoder-Decoder mạnh mẽ của Google, có khả năng zero-shot tốt và hiểu ngữ nghĩa sâu.
* **`fine_tune_qwen.ipynb`**: Huấn luyện mô hình **Qwen** (sử dụng kỹ thuật PEFT/LoRA). Qwen là đại diện cho dòng mô hình Decoder-only hiện đại, hiệu năng cao trên tiếng Việt.

### 3. Ứng dụng Demo (Interface)
* **`interface.ipynb`**: Giao diện người dùng (GUI) được xây dựng bằng **Gradio**.
    * Cho phép người dùng chọn mô hình (Model Selector).
    * Hỗ trợ nhập văn bản trực tiếp hoặc dán URL bài báo.
    * Tùy chỉnh tham số sinh văn bản (Beam size, Length penalty...).
    * So sánh trực quan kết quả tóm tắt.

---

## 🛠 Cài đặt

### Yêu cầu hệ thống
* Python 3.10 trở lên.
* GPU: Khuyến nghị sử dụng NVIDIA T4, V100 hoặc A100 (đặc biệt cần thiết cho `fine_tune_flan` và `fine_tune_qwen`).

### Cài đặt thư viện
Chạy lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install torch transformers datasets peft trl gradio openai python-dotenv newspaper3k lxml[html_clean]