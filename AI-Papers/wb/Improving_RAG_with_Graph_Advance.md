# Improving RAG with Graph - Advanced Concepts & Architectures

## Khái niệm chuyên sâu

### 1. Two-Stage Graph Construction Pipeline

**Stage 1: AMR Graph Generation**
- Mỗi question-document pair được concat: `"question:<question text><document text>"`
- AMRBART parse thành AMR graph Gqp = {V, E}
- Nodes V: concepts/entities
- Edges E: relations (source, relation, destination)

**Stage 2: Document Graph Construction**
- Nodes: Top 100 documents từ DPR
- Edges: Connect nếu 2 AMR graphs có common nodes
- Remove isolated nodes
- Undirected graph để information flow 2 chiều

### 2. Advanced Node Feature Engineering

**Traditional Approach Problems**:
- Wang et al. integrate toàn bộ AMR tokens → high computational cost
- Redundant information → overfitting risk
- GPU memory explosion

**G-RAG Solution**:
```python
# Node feature generation
xi = Encode(concat(pi, ai))

# Where ai is extracted via:
1. Find SSSPs from "question" node
2. Extract node concepts along paths
3. Concatenate concepts as text sequence
```

**Key Insight**: Negative documents có 2 patterns:
- Không establish adequate connections với question
- Quá nhiều question-related info nhưng thiếu gold answers

### 3. Edge Feature Design

**Edge features ÊRⁿˣⁿˣˡ với l=2**:
```
Êij1 = # common nodes between Gqpi and Gqpj
Êij2 = # common edges between Gqpi and Gqpj
```

**Normalization Strategy**:
- Normalize on first và second dimensions riêng biệt
- Tránh explosive scale khi multiply trong GCN operations
- Similar to techniques trong paper [8]

### 4. GNN Architecture Details

**Node Update Rule**:
```
x^ℓ_v = g(x^(ℓ-1)_v, ∑_{u∈N(v)} f(x^(ℓ-1)_u, e^(ℓ-1)_uv))
```

**Edge-weighted Feature Computation**:
```
f(x^(ℓ-1)_u, e^(ℓ-1)_uv) = ∑_{m=1}^l e^(ℓ-1)_uv(m) × x^(ℓ-1)_u
```

**Design Choices**:
- Mean aggregator for stability
- 2-layer GCN (deeper không improve)
- Hidden dimensions: {8, 64, 128}
- Dropout rates: {0.1, 0.2, 0.4}

### 5. Training Strategy Optimization

**Cross-entropy Loss Issues**:
- Imbalanced data (few positive, many negative documents)
- Not designed for ranking tasks
- Poor performance on hard negatives

**Pairwise Ranking Loss Advantages**:
```
RL_q(si, sj, r) = max(0, -r(si - sj) + 1)
```
- Directly optimize for ranking
- Better handle imbalanced data
- Significant performance improvement

## KIẾN TRÚC THAM KHẢO CHO CÁC SCENARIO THỰC TẾ

### Scenario 1: Enterprise Knowledge Base QA System

**Architecture**:
```
User Query → DPR Retrieval → AMR Parser → G-RAG Reranker → LLM Reader
     ↓            ↓              ↓              ↓              ↓
  Embedding    Top-100      Question+Doc    Document      Final
  Search       Docs         AMR Graphs      Graph         Answer
```

**Implementation Details**:
- DPR với custom embeddings cho domain
- Parallel AMR parsing với batching
- Redis cache cho AMR graphs
- GPU cluster cho GNN inference
- Latency target: < 500ms end-to-end

**Optimizations**:
- Pre-compute AMR cho frequent documents
- Graph sparsification cho large document sets
- Ensemble với fast MLP ranker fallback

### Scenario 2: Multi-Document Reasoning System

**Architecture**:
```
Complex Query → Multi-hop Retrieval → Cross-doc Graph → Reasoning
      ↓               ↓                    ↓             ↓
  Decompose      Iterative            G-RAG with      Chain
  Sub-queries    Retrieval          Extended Features  Results
```

**Key Components**:
- Query decomposition module
- Iterative retrieval với feedback
- Extended edge features:
  - Temporal relations
  - Entity coreference
  - Causal connections
- Multi-hop reasoning chains

### Scenario 3: Real-time News QA Platform

**Architecture**:
```
News Stream → Incremental Indexing → Dynamic G-RAG → Live QA
     ↓              ↓                     ↓            ↓
  Crawling     Update DPR +          Streaming      WebSocket
  Pipeline     AMR Cache            Graph Update     Delivery
```

**Technical Challenges & Solutions**:
- Incremental graph updates
- Sliding window cho recency
- Approximate AMR cho speed
- Edge pruning strategies
- Distributed GNN inference

### Scenario 4: Educational Tutoring System

**Architecture**:
```
Student Question → Concept Extraction → Knowledge Graph RAG → Explanation
        ↓                ↓                     ↓                ↓
   Difficulty       Curriculum           G-RAG with         Adaptive
   Assessment       Alignment          Learning Paths      Response
```

**Advanced Features**:
- Prerequisite concept edges
- Difficulty-weighted ranking
- Student model integration
- Progressive disclosure
- Multi-modal support (text + diagrams)

### Scenario 5: Legal Document Analysis

**Architecture**:
```
Legal Query → Citation Network → Precedent Graph → Argument Mining
     ↓             ↓                  ↓               ↓
  NER for      Build Legal       G-RAG with       Extract
  Entities     Doc Graph        Case Relations    Holdings
```

**Domain Adaptations**:
- Legal AMR parsing rules
- Citation as strong edges
- Temporal precedence weights
- Jurisdiction-aware ranking
- Contradiction detection

## Performance Optimization Strategies

### 1. Embedding Model Selection
```
Best performers (by MRR on NQ/TQA):
- Ember: 29.0/19.8 (most consistent)
- GTE: 29.9/19.2 (good balance)
- BGE: 28.7/18.7 (memory efficient)
- BERT: 27.3/19.8 (baseline)
```

### 2. Hyperparameter Guidelines
```
Optimal settings:
- Hidden dim: 8 (sufficient for most cases)
- Dropout: 0.1 (avoid over-regularization)
- Learning rate: 1e-4
- Batch size: 5 (memory constraints)
- Warmup steps: 1000
```

### 3. Scalability Considerations
- Graph sparsification for > 1000 documents
- Approximate nearest neighbor cho initial retrieval
- Distributed GNN training với DGL
- Model quantization cho edge deployment

### 4. LLM Integration Lessons
**Why PaLM 2 underperforms**:
- Coarse-grained scores (divisible by 5)
- Excessive tied rankings
- Not optimized for ranking tasks
- Better as reader than reranker

**Recommendation**: Use specialized rerankers (G-RAG) + LLM readers

## Cost-Benefit Analysis

### Computational Efficiency
- G-RAG on A100 40GB: Handles full pipeline
- BART-GST: Requires more memory, prone to overfitting
- Inference time: ~50ms per query (after caching)
- Training time: 50k steps ≈ 10 hours

### Performance Gains
- NQ dataset: +7.1% MRR over baseline
- TQA dataset: +7.7% MRR improvement
- Exact match improvement: +15-20% in downstream QA
- Especially effective on "weak connection" documents

### ROI Metrics
- Reduced LLM calls: 30% (better document selection)
- User satisfaction: +25% (more relevant answers)
- System latency: Comparable to BERT reranker
- Infrastructure cost: 40% less than BART-GST