# Parameter Efficient Fine Tuning: A Comprehensive Analysis

## Paper nói về điều gì?

Bài báo này cung cấp một phân tích toàn diện về **Parameter Efficient Fine-Tuning (PEFT)** - các kỹ thuật fine-tuning hiệu quả tham số cho các mô hình deep learning. Các điểm chính:

- **PEFT** cho phép fine-tune chỉ một phần nhỏ tham số của mô hình pre-trained lớn thay vì toàn bộ
- So sánh hiệu quả của các phương pháp PEFT khác nhau trên nhiều ứng dụng: text generation, medical imaging, protein modeling, code review, speech synthesis
- Chứng minh PEFT có thể giảm 70-95% số lượng tham số cần train mà vẫn duy trì hoặc cải thiện hiệu suất
- Đặc biệt hiệu quả khi dữ liệu ít hoặc tài nguyên tính toán hạn chế

## Chi tiết các khái niệm cơ bản cần nắm được để hiểu paper

### 1. Fine-tuning Traditional (Full Fine-tuning)
- **What is it**: Điều chỉnh toàn bộ tham số của mô hình pre-trained cho task mới
- **Why we need it**: Adapt mô hình tổng quát thành mô hình chuyên biệt cho task cụ thể

### 2. Pre-trained Models
- **What is it**: Các mô hình đã được train trên lượng data khổng lồ (BERT, GPT, T5, LLaMA)
- **Why we need it**: Chứa knowledge tổng quát có thể transfer sang nhiều tasks khác nhau

### 3. Parameter Efficiency
- **What is it**: Tỷ lệ giữa hiệu suất đạt được và số lượng tham số cần fine-tune
- **Why we need it**: Giảm chi phí tính toán, memory, và thời gian training

### 4. Catastrophic Forgetting
- **What is it**: Mô hình quên knowledge đã học từ pre-training khi fine-tune
- **Why we need it**: Hiểu để tránh mất performance trên cả task mới và task gốc

## Các khái niệm quan trọng trong paper

### 1. LoRA (Low-Rank Adaptation)
- **What is it**: Decompose weight updates thành low-rank matrices, chỉ train matrices nhỏ này
- **Why we need it**: Giảm 90% parameters, memory efficient (3.3B vs 33B params), widely applicable

### 2. LoReFT (Low-rank Linear Subspace ReFT)
- **What is it**: Modify internal representations qua projection matrix R với formula DII(b,s,R) = b + R⊤(Rs - Rb)
- **Why we need it**: Cải thiện 10-50x parameter efficiency so với PEFT methods khác

### 3. Adapter Modules
- **What is it**: Insert small trainable modules giữa các layers của pre-trained model
- **Why we need it**: Modular design, flexible, giữ nguyên model gốc, dễ switch giữa tasks

### 4. BitFit
- **What is it**: Chỉ fine-tune bias terms của model (0.22% parameters)
- **Why we need it**: Extremely parameter efficient, đơn giản implement, hiệu quả với limited data

### 5. Prefix Tuning
- **What is it**: Thêm trainable prefix tokens vào input để guide model behavior
- **Why we need it**: Không modify model weights, task-specific adaptation qua learned prefixes

### 6. Freezing Layers
- **What is it**: Freeze một số layers (thường bottom layers) và chỉ train top layers
- **Why we need it**: Giữ general features ở bottom, adapt task-specific features ở top

## Các thuật ngữ quan trọng cần nhớ

1. **PEFT (Parameter Efficient Fine-Tuning)**: Họ các phương pháp fine-tune với ít parameters
2. **Trainable Parameters**: Tham số được update trong quá trình training
3. **Frozen Parameters**: Tham số giữ nguyên từ pre-trained model
4. **Parameter Reduction Percentage**: Phần trăm giảm parameters so với full fine-tuning
5. **Adapter Layers**: Small neural networks inserted between transformer layers
6. **Rank Decomposition**: Phân tích matrix thành tích của low-rank matrices
7. **Knowledge Distillation**: Transfer knowledge từ model lớn sang model nhỏ
8. **Structural Redundancy**: Dư thừa trong architecture có thể exploit để giảm parameters
9. **Message Passing**: Cơ chế update trong graph neural networks
10. **Intervention Functions**: Functions modify internal representations (như DII trong LoReFT)