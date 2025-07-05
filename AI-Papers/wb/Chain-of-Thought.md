# Chain-of-Thought trong Neural Code Generation

## Paper nói về điều gì?

Paper này nghiên cứu về việc áp dụng kỹ thuật Chain-of-Thought (CoT) để cải thiện khả năng sinh code của các Lightweight Language Models (ℓLMs - các mô hình có dưới 10 tỷ tham số). Nghiên cứu đề xuất phương pháp COTTON (Chain Of Thought cOde geNeration) - một cách tiếp cận mới cho phép ℓLMs tự động sinh ra CoT chất lượng cao để hướng dẫn quá trình sinh code, đạt hiệu suất tương đương với các LLMs lớn hơn nhiều.

## Các khái niệm quan trọng trong paper

### 1. Chain-of-Thought (CoT)

**What is it?**
- Chuỗi các bước suy luận trung gian bằng ngôn ngữ tự nhiên dẫn đến kết quả cuối cùng
- Giúp model chia nhỏ vấn đề phức tạp thành các bước đơn giản hơn
- Ví dụ: Thay vì sinh code trực tiếp, CoT sẽ mô tả từng bước cần thực hiện

**Why we need it?**
- Cải thiện độ chính xác của code được sinh ra
- Tăng khả năng giải thích và hiểu được của model
- Giúp ℓLMs với tài nguyên hạn chế đạt hiệu suất cao hơn
- Trong ví dụ của paper: CodeGen-350M với CoT đạt độ chính xác từ 0% lên 100%

### 2. Lightweight Language Models (ℓLMs)

**What is it?**
- Các mô hình ngôn ngữ có dưới 10 tỷ tham số
- Có thể chạy trên GPU người dùng thông thường (RTX 3090/4090)
- Ví dụ: CodeGen (350M-6B), StarCoder (1B-7B), CodeT5+ (220M-6B)

**Why we need it?**
- Chi phí thấp hơn nhiều so với LLMs (GPT-3.5 cần 10,000 GPU V100)
- Dễ triển khai cho cá nhân và tổ chức nhỏ
- Có thể fine-tune với tài nguyên hạn chế
- Phù hợp cho các ứng dụng software engineering thực tế

### 3. COTTON Framework

**What is it?**
- Phương pháp tự động sinh CoT cho code generation
- Sử dụng CodeLlama-7B làm base model
- Được huấn luyện trên dataset CodeCoT-9k (9,264 mẫu)
- Áp dụng LoRA để giảm chi phí huấn luyện

**Why we need it?**
- ℓLMs không thể tự sinh CoT chất lượng cao (chỉ đạt <60% consistency)
- Viết CoT thủ công tốn thời gian và chi phí
- COTTON giúp ℓLMs đạt hiệu suất tương đương LLMs với chi phí thấp hơn
- Cải thiện Pass@1 lên đến 95% cho một số models

### 4. Few-shot vs Zero-shot Learning

**What is it?**
- **Zero-shot**: Chỉ cung cấp instruction và input, không có ví dụ
- **Few-shot**: Cung cấp một số ví dụ (3 trong paper) kèm instruction

**Why we need it?**
- Zero-shot phù hợp khi model đã được fine-tune
- Few-shot cải thiện performance 46-659% so với zero-shot khi không fine-tune
- Paper khuyến nghị dùng 3 ví dụ (đủ hiệu quả, không cần nhiều hơn)
- Sử dụng BM25 để chọn ví dụ phù hợp nhất

### 5. Multi-agent Alignment

**What is it?**
- Sử dụng 3 agents (based on GPT-3.5) để xây dựng dataset:
  - Quality Checker: Đánh giá giá trị giáo dục của code
  - CoT Generator: Sinh CoT từ code và comment
  - Consistency Checker: Kiểm tra tính nhất quán giữa CoT và code

**Why we need it?**
- Đảm bảo chất lượng cao của dataset huấn luyện
- Loại bỏ code không có giá trị học tập
- Đảm bảo CoT phản ánh đúng logic của code
- Tạo dataset đa dạng và chất lượng

### 6. Parameter-Efficient Fine-tuning (LoRA)

**What is it?**
- Low-Rank Adaptation - phương pháp fine-tune hiệu quả
- Chỉ cập nhật một phần nhỏ tham số (low-rank matrices)
- Giữ nguyên pre-trained weights, thêm trainable matrices B và A

**Why we need it?**
- Giảm đáng kể tài nguyên cần thiết cho fine-tuning
- Có thể train trên single GPU (RTX 3090)
- Vẫn đạt hiệu suất cao
- Phù hợp cho các tổ chức có ngân sách hạn chế

## Các thuật ngữ quan trọng cần nhớ

1. **Pass@1**: Tỷ lệ code được sinh ra pass test cases ngay lần đầu
2. **CoT-Pass@1**: Tỷ lệ pass test khi có hướng dẫn của CoT
3. **BM25**: Thuật toán ranking để chọn ví dụ cho few-shot learning
4. **Greedy Search**: Thuật toán decoding chọn token có xác suất cao nhất
5. **RMSNorm**: Root Mean Square Layer Normalization trong CodeLlama
6. **GQA (Group Query Attention)**: Kỹ thuật attention được tối ưu trong CodeLlama
7. **Consistency Metric**: Đo lường độ nhất quán giữa CoT và code
8. **Educational Value**: Giá trị học tập/hướng dẫn của CoT cho developers

## Key Findings

1. **ℓLMs không thể tự sinh CoT chất lượng cao**: Hầu hết ℓLMs đạt consistency < 60%
2. **COTTON vượt trội các baselines**: Đạt 93.29% consistency trên HumanEval
3. **Hiệu suất cải thiện đáng kể**: CodeT5+ 6B từ 26.22% lên 42.68% Pass@1
4. **Chi phí-hiệu quả tốt**: StarCoder-7B + COTTON ≈ StarCoder-16B performance
5. **Few-shot hiệu quả hơn persona prompting**: Persona làm giảm performance 1-54%