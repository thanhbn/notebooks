# Parameter Efficient Fine Tuning - Advanced Concepts & Architectures

## Khái niệm chuyên sâu

### 1. LoReFT Deep Dive - Distributed Interchange Intervention

**Mathematical Foundation**:
```
DII(b, s, R) = b + R⊤(Rs - Rb)
```
Trong đó:
- b: Hidden states hiện tại
- s: Target state mong muốn  
- R: Projection matrix (learned)

**Key Insights**:
- Intervention diễn ra trong subspace thấp chiều
- R học cách project và modify representations
- Hiệu quả hơn LoRA 10-50x về parameters

**Performance trên Commonsense Reasoning**:
- LLaMA-7B: 80.2% avg accuracy (0.031% params)
- LLaMA-13B: 83.3% avg accuracy (0.025% params)
- Vượt trội ChatGPT (77.0%) với fraction của parameters

### 2. Adapter Architecture Analysis

**Standard Adapter Module**:
```python
class Adapter(nn.Module):
    def __init__(self, dim, reduction_factor=16):
        self.down_project = nn.Linear(dim, dim // reduction_factor)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(dim // reduction_factor, dim)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual
```

**Variations**:
- **AdapterS (Sequential)**: 0.99% params, 70.8% accuracy
- **AdapterP (Parallel)**: 3.54% params, 72.3% accuracy
- **Task-specific Adapters**: Different adapters cho different tasks

### 3. LoRA Mathematical Formulation

**Weight Update Decomposition**:
```
W' = W + ΔW = W + BA
```
Trong đó:
- W: Original weight matrix (d × k)
- B: Low-rank matrix (d × r)
- A: Low-rank matrix (r × k)
- r << min(d, k): Rank của decomposition

**Training Strategy**:
- Freeze W, chỉ train B và A
- Merge khi inference: W' = W + BA
- No additional inference latency

### 4. Advanced BitFit Strategy

**Bias-only Fine-tuning**:
- Chỉ update bias terms trong:
  - Linear layers
  - LayerNorm layers
  - Attention projections
- Typically 0.1-0.5% total parameters

**Why it works**:
- Bias terms control activation shifts
- Small changes có large effects
- Preserve learned representations

### 5. Comparative Performance Analysis

**Medical Imaging Results**:
```
Method          | Params (%) | Performance
----------------|------------|-------------
Full FT         | 100%       | Baseline
Adapter         | 1.18%      | -0.5%
BitFit          | 0.22%      | -1.2%
LoRA            | 0.81%      | -0.3%
BitFit + LoRA   | 1.03%      | +0.2%
```

### 6. Multi-method Combinations

**Synergistic Approaches**:
- BitFit + LoRA: Combine bias và low-rank updates
- Adapter + Prefix: Structural và input modifications
- Layer Freezing + Selective Updates: Hybrid strategies

## KIẾN TRÚC THAM KHẢO CHO CÁC SCENARIO THỰC TẾ

### Scenario 1: Enterprise LLM Deployment for Multiple Tasks

**Architecture**:
```
Base LLaMA-70B → Task Router → Task-Specific PEFT Modules → Output
       ↓              ↓                    ↓                    ↓
   Frozen Base    Classify Task      Load LoRA/Adapter      Inference
                                    (Customer Service,
                                     Code Gen, QA)
```

**Implementation Details**:
- Base model: Shared across all tasks (frozen)
- Task modules: Separate LoRA weights per task
- Storage: 0.8% × N tasks vs N × 100% for full models
- Switching: < 100ms task switch time

**Optimization**:
```python
# Dynamic LoRA loading
def load_task_lora(task_name):
    lora_weights = torch.load(f"loras/{task_name}.pt")
    model.inject_lora(lora_weights)
    return model
```

### Scenario 2: Medical Multi-Modal Analysis System

**Architecture**:
```
Vision Encoder → Frozen ViT → BitFit Layers → Feature Fusion
Text Encoder → Frozen BERT → LoRA Modules → Cross-Attention → Diagnosis
                                  ↓
                            Adapter Layers
```

**Key Components**:
- Vision: ViT with BitFit (0.22% params)
- Text: BERT with LoRA (0.81% params)
- Fusion: Trainable cross-attention (2% params)
- Total trainable: < 3.5% of full model

**Performance Metrics**:
- Accuracy: 94.3% (vs 95.1% full FT)
- Training time: 8 hours (vs 72 hours)
- GPU memory: 16GB (vs 80GB)

### Scenario 3: Real-time Code Review System

**Architecture**:
```
Code Input → Tokenizer → LLaMA-7B → LoRA Review Module → Review Output
     ↓           ↓           ↓              ↓                  ↓
  AST Parse   Special     Frozen      Task-specific      Format as
             Tokens       Base         (0.8% params)     Comments
```

**LoRA Configuration**:
```python
lora_config = {
    "r": 16,  # rank
    "alpha": 32,  # scaling
    "dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

**Results**:
- Review accuracy: 70.49% F1
- Comment BLEU-4: 5.70
- Inference speed: 50ms/review

### Scenario 4: Protein Function Prediction Pipeline

**Architecture**:
```
Protein Sequence → ESM-2 Encoder → Multi-Task PEFT → Predictions
        ↓               ↓               ↓              ↓
   Amino Acids     Frozen Base    Task Adapters   Function,
                                  (PPI, Symmetry,   Structure,
                                   Localization)    Interaction
```

**PEFT Distribution**:
- PPI prediction: Adapter (1.18%)
- Symmetry: BitFit (0.22%)
- Localization: LoRA (0.81%)
- Combined: < 2.5% total params

### Scenario 5: Multi-lingual Speech Emotion Recognition

**Architecture**:
```
Audio → Wav2Vec 2.0 → Language Router → PEFT Modules → Emotion
  ↓          ↓              ↓               ↓            ↓
MFCC    Frozen Base    Detect Lang    Lang-specific   7 Classes
                                      LoRA (0.8%)
```

**Advanced Features**:
- Per-language LoRA modules
- Fairness constraints in training
- Multi-task learning objectives

**Performance**:
- UAR: 67.3% with LoRA
- Fairness score: 0.89
- Languages: 12 supported

## Performance Optimization Strategies

### 1. Memory-Efficient Training
```python
# Gradient checkpointing + PEFT
model.enable_gradient_checkpointing()
model.enable_lora(r=8, alpha=16)

# Mixed precision
with autocast():
    outputs = model(inputs)
```

### 2. Multi-Task Learning
```python
# Shared base + task-specific heads
class MultiTaskPEFT(nn.Module):
    def __init__(self, base_model, tasks):
        self.base = base_model
        self.task_adapters = nn.ModuleDict({
            task: Adapter(dim=768) for task in tasks
        })
```

### 3. Hyperparameter Guidelines

**LoRA**:
- Rank r: 4-64 (start with 8)
- Alpha: r × 2 (scaling factor)
- Target modules: attention layers first

**Adapters**:
- Reduction factor: 16-64
- Placement: After FFN preferred
- Initialization: Near-identity

**BitFit**:
- Learning rate: 10x của normal FT
- Warmup: Longer (2000 steps)
- Regularization: Higher weight decay

### 4. Deployment Optimization

**Inference Speed**:
- LoRA merge: Pre-compute W' = W + BA
- Adapter caching: Keep in GPU memory
- Quantization: Compatible với 8-bit/4-bit

**Storage Efficiency**:
```
Full model: 70GB
Base + 10 LoRA tasks: 70GB + 10×0.56GB = 75.6GB
Savings: 624.4GB (89%)
```

## Cost-Benefit Analysis

### Computational Savings
- Training time: 5-10x faster
- GPU memory: 80-95% reduction
- Storage: < 1% per task vs 100%
- Energy consumption: 70-90% lower

### Performance Trade-offs
- Typical accuracy drop: 0-2%
- Some tasks see improvement (regularization effect)
- Better generalization with limited data
- Faster iteration and experimentation

### ROI Metrics
- Development velocity: 3x faster
- Infrastructure cost: 75% reduction
- Model serving: 10x more tasks per GPU
- Time to market: 50% reduction