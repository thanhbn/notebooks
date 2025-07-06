# Evaluating Large Language Models Trained on Code

## Paper nói về điều gì?

Bài báo giới thiệu **Codex** - một mô hình ngôn ngữ GPT được fine-tune trên code từ GitHub, và đánh giá khả năng viết code Python của nó. Điểm nổi bật chính:

- **Codex** là nền tảng cho GitHub Copilot - công cụ gợi ý code phổ biến
- Giới thiệu **HumanEval** - bộ dataset mới với 164 bài toán lập trình viết tay để đánh giá tính đúng đắn chức năng
- Codex-12B giải quyết được 28.8% bài toán với 1 mẫu, và 70.2% với 100 mẫu
- So sánh hiệu suất với các mô hình khác như GPT-3, GPT-J, GPT-Neo

## Chi tiết các khái niệm cơ bản cần nắm được để hiểu paper

### 1. Language Models for Code Generation
- **What is it**: Mô hình ngôn ngữ được huấn luyện để hiểu và sinh code thay vì văn bản tự nhiên
- **Why we need it**: Tự động hóa việc viết code, tăng năng suất lập trình viên, giải quyết các bài toán lập trình phức tạp

### 2. Fine-tuning on Code
- **What is it**: Quá trình huấn luyện thêm một mô hình ngôn ngữ đã được pre-train trên dữ liệu code cụ thể
- **Why we need it**: Chuyển đổi khả năng hiểu ngôn ngữ tự nhiên sang khả năng hiểu và sinh code chính xác

### 3. Functional Correctness vs Match-based Metrics
- **What is it**: Đánh giá code dựa trên việc code có chạy đúng (pass unit tests) thay vì so khớp văn bản
- **Why we need it**: Code có thể có nhiều cách viết khác nhau nhưng vẫn đúng về mặt chức năng

## Các khái niệm quan trọng trong paper

### 1. HumanEval Dataset
- **What is it**: Bộ dataset 164 bài toán lập trình được viết tay, mỗi bài gồm function signature, docstring, và unit tests
- **Why we need it**: Đánh giá chính xác khả năng giải quyết vấn đề của mô hình, tránh data leakage từ GitHub

### 2. Pass@k Metric
- **What is it**: Phương pháp đánh giá xem có ít nhất 1 trong k mẫu code sinh ra pass được unit tests không
- **Why we need it**: Phản ánh thực tế sử dụng - developer có thể thử nhiều giải pháp và chọn cái đúng

### 3. Codex vs Codex-S
- **What is it**: 
  - Codex: Fine-tune GPT trên raw GitHub code
  - Codex-S: Fine-tune thêm trên standalone functions với unit tests
- **Why we need it**: Codex-S đạt hiệu suất cao hơn (37.7% vs 28.8% pass@1) do được train trên data gần với task hơn

### 4. Temperature Sampling
- **What is it**: Tham số điều khiển độ ngẫu nhiên khi sinh code (T=0.2 cho pass@1, T=0.8 cho pass@100)
- **Why we need it**: Temperature thấp cho code chính xác hơn, temperature cao cho đa dạng hơn khi cần nhiều mẫu

### 5. Sandbox Execution Environment
- **What is it**: Môi trường cách ly an toàn sử dụng gVisor để chạy và test code được sinh ra
- **Why we need it**: Bảo vệ hệ thống khỏi code độc hại hoặc lỗi khi thực thi code tự động sinh

## Các thuật ngữ quan trọng cần nhớ

1. **Codex**: Mô hình GPT fine-tuned trên 159GB Python code từ GitHub
2. **HumanEval**: Benchmark dataset với 164 programming problems  
3. **Pass@k**: Metric đánh giá functional correctness với k samples
4. **Docstring-to-code**: Task sinh code từ mô tả hàm (docstring)
5. **Nucleus sampling**: Phương pháp sampling với top-p = 0.95
6. **Mean log-probability**: Heuristic để chọn best sample khi không có unit tests
7. **APPS dataset**: Dataset khác với 5000 coding challenges
8. **Competitive programming problems**: 10,000 problems từ coding contest sites
9. **Continuous Integration (CI) tracing**: Thu thập 40,000 problems từ test suites
10. **Power law scaling**: Test loss giảm theo quy luật lũy thừa với model size