# Bảng Tóm Tắt Benchmark từ các Paper arXiv

| **Tên File** | **Nội Dung Benchmark Chính** |
|--------------|----------------------------|
| **1.2.3.LeetCodeDataset.A Temporal Dataset for Robust Evaluation and_2504.14655v1.txt** | - LeetCodeDataset: high-quality benchmark for evaluating LLMs<br>- Thiếu reasoning-focused coding benchmarks<br>- LiveCodeBench: commonly used benchmark<br>- Evaluated across: HumanEval, MBPP, và các benchmarks khác<br>- Benchmark temporal để tránh data leakage |
| **1.3.2.AgentCoder. Multi-Agent-based Code Generation with Iterative Testing and Optimisation2312.13010v3.txt** | - OpenAI series set benchmark for performance<br>- Sử dụng CodeGeeX benchmark execution<br>- Multi-agent approach for code generation |
| **1.3.3. Chain-of-Thought in Neural Code Generation.2312.05562v2.txt** | - COTTON outperform trên various benchmarks<br>- HumanEval benchmark để test generalizability<br>- HumanEval-plus, OpenEval benchmarks<br>- Machine learning benchmark dataset cho code understanding |
| **1.3.4. StarCoder 2 and The Stack v2- The Next Generation. 2402.19173v1.txt** | - Comprehensive evaluation trên multiple benchmarks<br>- HumanEval, MBPP: most widely studied benchmarks<br>- HumanEval+, MBPP+: enhanced với 80×/35× more tests<br>- MultiPL-E, DS-1000, HumanEvalFix, CanItEdit<br>- CRUXEval: 800 samples benchmark<br>- GSM8K math-reasoning benchmark<br>- RepoBench, CrossCodeEval cho repository-level<br>- "Asleep at the Keyboard" security benchmark<br>- BOLD bias benchmark |
| **1.4.2. LLMDFA- Analyzing Dataflow in Code with - 2402.10754v2.txt** | - Juliet Test Suite: widely used benchmark<br>- SecBench.js: real-world JavaScript benchmark<br>- TaintBench: malware benchmarking |
| **1.4.3. AST-T5- Structure-Aware Pretraining for Code Generation and Understanding - 2401.03003v4.txt** | - CodeT5+ across various benchmarks<br>- HumanEval và MBPP benchmarks<br>- EvalPlus: more rigorous benchmark<br>- MBXP benchmarks cho multilingual evaluation |
| **2004.13820v2.txt** | - STS benchmark datasets cho semantic similarity<br>- SICK dataset và STS benchmark<br>- Pearson's Correlation trên STS benchmark |
| **2102.04664v2.txt** | - CodeXGLUE: benchmark dataset cho machine learning<br>- BigCloneBench: large code clone benchmark<br>- GLUE: multi-task benchmark cho natural language |
| **2105.12655v2.txt** | - CodeNet: rich set of code samples<br>- POJ-104 benchmark và GCJ-297<br>- Benchmark datasets từ CodeNet (C++, Python, Java)<br>- C++1000, C++1400, Java250 benchmarks |
| **2203.09095v2.txt** | - CodeReviewer: GitHub code review benchmark<br>- High-quality benchmark dataset cho 9 tasks<br>- Pre-training và benchmark cho evaluation |
| **2212.09132v1.txt** | - CodeXGLUE benchmark cho code completion<br>- Synthetic vs real-world benchmarks<br>- ManyTypes4Py: benchmark Python dataset |
| **2305.06161v2.txt** | - StarCoder evaluation trên multiple benchmarks<br>- HumanEval, MBPP: widely-used benchmarks<br>- DS-1000: natural và reliable benchmark<br>- ODEX benchmark<br>- MultiPL-E: scalable approach to benchmarking<br>- Security benchmarks, FIM benchmarks<br>- GSM8K math-reasoning, MMLU language understanding |
| **2308.10462v3.txt** | - HumanEval extensively used to benchmark code generation<br>- Challenging benchmark với test cases<br>- CodeXGLUE: machine learning benchmark |
| **2406.04712v1.txt** | - AICoderEval: benchmark cho AI-oriented code generation<br>- 2,000 code files dataset<br>- Domain-specific tasks benchmark construction |
| **2406.11931v1.txt** | - DeepSeek-Coder-V2 trên math và code benchmarks<br>- HumanEval, MBPP benchmarks<br>- LiveCodeBench (LCB), USACO benchmarks<br>- SWE-bench: comprehensive benchmark cho real-world issues<br>- CRUXEval: code reasoning benchmark<br>- Mathematical benchmarks: GSM8K, MATH, AIME<br>- Standard benchmarks: BBH, MMLU, Arena-Hard |
| **2407.02485v1.txt** | - RankRAG: state-of-the-art performance trên RAG benchmarks<br>- KILT benchmark cho fact verification<br>- FEVER, Natural Questions benchmarks<br>- Biomedical RAG benchmarks |
| **2407.08275v1.txt** | - Massive Text Embedding benchmarks<br>- BEIR benchmark datasets<br>- Performance benchmarks vs embedding similarity |
| **2407.15462v4.txt** | - ANN Benchmarks cho nearest neighbor search<br>- Common heterogeneous benchmarks<br>- State-of-the-art across benchmark datasets |
| **2409.10959v1.txt** | - CodeReviewer: widely employed benchmark dataset<br>- GitHub code review benchmark cho evaluation |
| **2410.20424v3.txt** | - ML-Bench: benchmark cho language agents<br>- Machine learning benchmark applications |
| **2502.02757v2.txt** | - CodeReviewer benchmark cleaning<br>- Chỉ 64% của benchmark là valid<br>- Benchmark dataset quality issues |
| **2503.13505v1.txt** | - RouterBench: benchmark cho routing<br>- Instruction-following benchmarks<br>- Cost-effective methods benchmarking |
| **2504.10046v1.txt** | - DevEval benchmark effectiveness<br>- EvoCodeBench: evolving code generation benchmark |
| **2505.24581v1.txt** | - MTEB benchmark cho embedding models<br>- STS benchmarks improvement<br>- Arabic NLP benchmarks<br>- Matryoshka vs base models trên MTEB |

## **Tóm Tắt Thống Kê:**

- **Tổng số file:** 25+ papers
- **Benchmark phổ biến nhất:** HumanEval, MBPP, CodeXGLUE, GSM8K
- **Lĩnh vực chính:** Code Generation, Math Reasoning, NLP, Security
- **Xu hướng:** Từ synthetic → real-world, multilingual, multimodal