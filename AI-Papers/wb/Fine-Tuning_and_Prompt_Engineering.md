# Fine-Tuning và Prompt Engineering cho Code Review với Large Language Models

## Paper nói về điều gì?

Paper này nghiên cứu về việc tối ưu hóa Large Language Models (LLMs) cho tự động hóa code review thông qua hai phương pháp chính:
- **Fine-tuning**: Huấn luyện thêm LLMs trên dữ liệu code review cụ thể
- **Prompt Engineering**: Sử dụng các kỹ thuật prompting để hướng dẫn LLMs mà không cần huấn luyện thêm

Nghiên cứu so sánh hiệu suất của GPT-3.5 và Magicoder khi áp dụng các kỹ thuật khác nhau trên 3 dataset code review thực tế.

## Các khái niệm quan trọng trong paper

### 1. Code Review Automation

**What is it?**
- Quá trình tự động hóa việc xem xét và cải thiện code bằng AI
- LLMs được huấn luyện để hiểu mối quan hệ giữa code gốc và code đã được cải thiện

**Why we need it?**
- Code review thủ công tốn thời gian và chi phí cao
- Developers thường phải chờ phản hồi từ reviewers
- Tự động hóa giúp tăng tốc độ phát triển và đảm bảo chất lượng code

### 2. Fine-tuning

**What is it?**
- Quá trình huấn luyện thêm một pre-trained LLM trên dataset cụ thể
- Model học mối quan hệ trực tiếp giữa input (code cần review) và output (code đã cải thiện)

**Why we need it?**
- Pre-trained models thiếu kiến thức domain-specific về code review
- Fine-tuning giúp model đạt hiệu suất cao hơn 73-74% so với không fine-tune
- Phù hợp khi có đủ dữ liệu training

### 3. Zero-shot Learning

**What is it?**
- Sử dụng LLM để giải quyết task mà không cần examples
- Chỉ cung cấp instruction và input cho model

**Why we need it?**
- Không cần dữ liệu training cụ thể
- Nhanh chóng và dễ triển khai
- Phù hợp cho các task đơn giản hoặc khi không có dữ liệu

### 4. Few-shot Learning

**What is it?**
- Cung cấp một số ví dụ (3 examples trong paper) cùng với instruction
- Model học pattern từ các ví dụ để áp dụng cho input mới

**Why we need it?**
- Cải thiện hiệu suất 46-659% so với zero-shot
- Không cần fine-tuning (tiết kiệm tài nguyên)
- Hiệu quả cho cold-start problem

### 5. Persona Prompting

**What is it?**
- Thêm "vai trò" vào prompt (VD: "Bạn là expert developer...")
- Hướng dẫn model generate output theo phong cách của persona

**Why we need it?**
- Giúp model hiểu context và mục đích rõ hơn
- Tuy nhiên, paper phát hiện persona có thể làm giảm hiệu suất 1-54%
- Không recommended cho code review automation

### 6. BM25 Algorithm

**What is it?**
- Thuật toán ranking để chọn các ví dụ phù hợp nhất cho few-shot learning
- Tính toán độ tương đồng giữa query và documents

**Why we need it?**
- Chọn được examples relevant nhất từ training set
- Cải thiện chất lượng của few-shot learning
- Được chứng minh hiệu quả hơn các phương pháp khác

## Các thuật ngữ quan trọng cần nhớ

1. **Exact Match (EM)**: Metric đo % output giống hoàn toàn với ground truth
2. **CodeBLEU**: Metric đánh giá similarity về syntax và semantics của code
3. **DiffHunk**: Đơn vị code change trong version control (một phần của diff)
4. **Cold-start Problem**: Vấn đề khi không có đủ data để train model
5. **Inference**: Quá trình sử dụng trained model để generate output
6. **Granularity**: Mức độ chi tiết của code (function-level vs diffhunk-level)
7. **Parameter-Efficient Fine-tuning (DoRA)**: Kỹ thuật fine-tune hiệu quả cho large models

## Key Findings và Recommendations

### Kết quả chính:
1. **Fine-tuned GPT-3.5** đạt hiệu suất cao nhất (EM cao hơn 73-74% so với baseline)
2. **Few-shot learning** hiệu quả khi không thể fine-tune (cải thiện 46-659% so với zero-shot)
3. **Persona prompting** không hiệu quả cho code review (giảm performance 1-54%)

### Khuyến nghị thực tiễn:
- **Ưu tiên fine-tuning** khi có đủ data và resources
- **Sử dụng few-shot learning KHÔNG có persona** khi gặp cold-start problem
- **Chọn 3 examples** cho few-shot learning (đủ hiệu quả, không cần nhiều hơn)
- **Sử dụng BM25** để chọn examples cho few-shot learning