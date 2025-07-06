# Improving RAG with Graph-based Reranking

## Paper nói về điều gì?

Bài báo giới thiệu **G-RAG** - một phương pháp reranking dựa trên Graph Neural Networks (GNNs) để cải thiện Retrieval Augmented Generation (RAG). Điểm nổi bật chính:

- **G-RAG** sử dụng kết nối giữa các documents và thông tin ngữ nghĩa (qua AMR graphs) để rerank documents
- Vượt trội hơn các phương pháp state-of-the-art với computational footprint nhỏ hơn
- PaLM 2 (LLM) hoạt động kém hiệu quả khi làm reranker do vấn đề tied ranking scores
- Giải quyết vấn đề khi documents có partial information hoặc weak connections với câu hỏi

## Chi tiết các khái niệm cơ bản cần nắm được để hiểu paper

### 1. Retrieval Augmented Generation (RAG)
- **What is it**: Kỹ thuật kết hợp retrieval và generation để cải thiện LLM responses bằng cách grounding với context từ documents
- **Why we need it**: Giúp LLM truy cập thông tin cập nhật, giảm hallucination, và tăng độ chính xác của câu trả lời

### 2. Document Reranking
- **What is it**: Quá trình sắp xếp lại thứ tự documents được retrieve để đưa documents chứa câu trả lời lên top
- **Why we need it**: Retriever có thể miss documents quan trọng hoặc xếp hạng sai, reranker giúp filter và prioritize documents hiệu quả hơn

### 3. Graph Neural Networks (GNNs)
- **What is it**: Mạng neural xử lý dữ liệu dạng graph, update node features thông qua message passing từ neighbor nodes
- **Why we need it**: Capture được relationships và connections giữa documents mà traditional methods không làm được

### 4. Abstract Meaning Representation (AMR)
- **What is it**: Biểu diễn ngữ nghĩa của text dưới dạng directed graph với nodes là concepts/entities và edges là relations
- **Why we need it**: Cung cấp structured semantic information, giúp hiểu complex semantics tốt hơn natural language

## Các khái niệm quan trọng trong paper

### 1. Document Graph Construction
- **What is it**: Xây dựng graph với mỗi node là một document, edges kết nối documents có common concepts từ AMR
- **Why we need it**: Cho phép leverage connections giữa documents để identify positive documents với weak connections

### 2. Single Source Shortest Paths (SSSPs)
- **What is it**: Shortest paths từ node "question" trong AMR graph đến các concepts khác
- **Why we need it**: Xác định key factors giúp reranker identify relevant documents mà không add redundant AMR information

### 3. Tied Ranking Problem
- **What is it**: Khi nhiều documents có cùng relevance score (common với LLMs), gây khó khăn cho ranking
- **Why we need it**: Cần metrics mới (MTRR, TMHits@10) để evaluate fairly khi có tied scores

### 4. Two-Graph Architecture
- **What is it**: 
  - AMR graphs: Semantic representation của question-document pairs
  - Document graph: Connection graph giữa documents
- **Why we need it**: Kết hợp semantic information và cross-document connections cho better reranking

### 5. Edge Features Normalization
- **What is it**: Normalize edge features (số common nodes/edges) để tránh explosive scale trong GNN operations
- **Why we need it**: Đảm bảo stable training và proper information propagation trong graph

## Các thuật ngữ quan trọng cần nhớ

1. **G-RAG**: Graph-based Reranking for RAG framework
2. **Positive documents**: Documents chứa gold answers cho câu hỏi
3. **DPR (Dense Passage Retrieval)**: Initial retriever lấy top 100 documents
4. **ODQA (Open-Domain Question Answering)**: Task trả lời câu hỏi không giới hạn domain
5. **Mean Tied Reciprocal Ranking (MTRR)**: Metric mới xử lý tied ranking scores
6. **TMHits@10**: Tied Mean Hits@10 - variant của MHits@10 cho tied rankings
7. **AMRBART**: Model để parse text thành AMR graphs
8. **Message passing**: Cơ chế update node features trong GNN
9. **Pairwise ranking loss**: Loss function phù hợp cho ranking tasks hơn cross-entropy
10. **Weak connections**: Documents có relevant information nhưng không obvious connection với question