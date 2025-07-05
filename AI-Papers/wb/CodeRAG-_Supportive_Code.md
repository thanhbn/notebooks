# CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation

## Paper nói về điều gì?

Paper này đề xuất CodeRAG - một framework Retrieval-Augmented Generation (RAG) mới để truy xuất toàn diện các đoạn code hỗ trợ cho việc sinh code trong môi trường thực tế (repo-level code generation). Khác với các phương pháp RAG truyền thống chỉ dựa vào độ tương đồng văn bản hoặc cấu trúc đơn giản, CodeRAG xây dựng một hệ thống bigraph (gồm Requirement Graph và DS-Code Graph) để mô hình hóa mối quan hệ phức tạp giữa các yêu cầu và code trong repository, từ đó tìm kiếm các đoạn code hỗ trợ một cách toàn diện hơn.

## Chi tiết các khái niệm cơ bản cần nắm để hiểu paper

### 1. Repo-level Code Generation

**What is it?**
- Nhiệm vụ sinh code trong ngữ cảnh của toàn bộ repository
- Khác với sinh code độc lập, repo-level generation phải xem xét dependencies, domain knowledge và integration với codebase hiện có
- Ví dụ: Tạo một function mới cần gọi các functions/classes đã định nghĩa trong repo

**Why we need it?**
- Phản ánh quy trình phát triển phần mềm thực tế
- Developers thường làm việc trong môi trường có sẵn codebase
- Code mới phải tích hợp seamlessly với existing framework
- Sản phẩm thương mại như Copilot, Cursor hoạt động ở level này

### 2. Retrieval-Augmented Generation (RAG)

**What is it?**
- Kỹ thuật kết hợp retrieval (tìm kiếm thông tin liên quan) với generation (sinh nội dung mới)
- Cung cấp context phù hợp cho LLMs để sinh code chính xác hơn
- Giải quyết vấn đề knowledge gap của LLMs về specific codebase

**Why we need it?**
- LLMs có giới hạn context window
- Không thể input toàn bộ repository vào LLMs
- Cần chọn lọc thông tin relevant nhất để augment generation process
- Cải thiện accuracy và relevance của generated code

### 3. Graph-based Code Modeling

**What is it?**
- Biểu diễn code repository dưới dạng đồ thị
- Nodes: Code elements (functions, classes, methods, modules)
- Edges: Relationships (calls, imports, inheritance, contains)
- Cho phép capture complex dependencies

**Why we need it?**
- Text-based retrieval bỏ qua structural dependencies
- Code có quan hệ phức tạp không thể biểu diễn bằng text similarity
- Graph traversal cho phép explore related code systematically
- Support reasoning về code relationships

## Các khái niệm quan trọng trong paper

### 1. Requirement Graph

**What is it?**
- Đồ thị biểu diễn mối quan hệ giữa các requirements (functional descriptions)
- Nodes: Requirements của các functions/classes trong repo
- Edges: Parent-child relationships và similarity relationships
- Được xây dựng bằng LLM (DeepSeek-V2.5) để generate descriptions và annotate relationships

**Why we need it?**
- Bridge gap giữa natural language requirement và programming code
- Tìm sub-requirements của target requirement
- Identify semantically similar requirements
- Reasoning từ góc độ requirements thay vì chỉ code structure

### 2. DS-Code Graph (Dependency-Semantic Code Graph)

**What is it?**
- Đồ thị mô hình hóa code repository với cả dependency và semantic relationships
- 4 loại nodes: Module, Class, Method, Function
- 5 loại edges: Import, Contain, Inherit, Call, Similarity
- Mở rộng của traditional code graph với semantic relationships

**Why we need it?**
- Capture cả structural dependencies và semantic similarities
- Cho phép reasoning về code từ nhiều góc độ
- Support traversal để tìm indirectly related code
- Lưu trữ hiệu quả trong Neo4j database

### 3. Bigraph Mapping

**What is it?**
- Cơ chế mapping giữa Requirement Graph và DS-Code Graph
- Map sub-requirement nodes → corresponding code nodes
- Map similar requirement nodes → their code implementations
- Tạo code anchor set cho further reasoning

**Why we need it?**
- Connect requirement-level reasoning với code-level retrieval
- Sub-requirement codes thường được invoke bởi target code
- Similar requirement codes provide implementation patterns
- Enable comprehensive supportive code retrieval

### 4. Code-oriented Agentic Reasoning

**What is it?**
- Quá trình cho phép LLMs tự động điều chỉnh strategy và search supportive codes
- 3 programming tools: WebSearch, GraphReason, CodeTest
- Sử dụng ReAct strategy để guide tool usage
- Iterative process based on LLM's needs

**Why we need it?**
- Static retrieval không đủ cho complex generation tasks
- LLMs cần different information ở different stages
- Mimics human programming process
- Allows dynamic knowledge acquisition

### 5. Supportive Codes (4 loại)

**What is it?**
1. APIs được invoke bởi target code (predefined functions/classes)
2. Code snippets semantically similar với target
3. Source codes indirectly related (qua graph traversal)
4. External domain knowledge (qua web search)

**Why we need it?**
- Mỗi loại cung cấp different type của useful information
- APIs: Direct dependencies cần được satisfy
- Similar codes: Implementation patterns và examples
- Indirect codes: Context và helper functions
- External knowledge: Domain-specific theorems và concepts

## Các thuật ngữ quan trọng cần nhớ

1. **Pass@1**: Metric đo tỷ lệ code sinh ra pass test suite ngay lần đầu
2. **DevEval**: Dataset benchmark cho repo-level code generation (1,825 samples từ 117 repos)
3. **Tree-sitter**: Static analysis tool để parse code và extract AST
4. **BM25**: Thuật toán ranking dựa trên textual similarity
5. **Neo4j**: Graph database dùng để store DS-Code Graph
6. **ReAct**: Reasoning strategy kết hợp reasoning traces và actions
7. **Code anchors**: Initial retrieved code nodes dùng làm starting points cho graph reasoning
8. **DuckDuckGo**: Search engine được dùng trong web search tool
9. **Black**: Code formatting và testing tool
10. **Standalone vs Non-standalone**: Phân loại code dựa trên dependencies
11. **Local-file, Cross-file, Local&Cross-file**: Các dependency types khác nhau

## Key Results

- CodeRAG cải thiện 40.90 Pass@1 trên GPT-4o và 37.79 trên Gemini-Pro so với no RAG
- Vượt trội hơn các commercial products như GitHub Copilot và Cursor
- Đặc biệt hiệu quả cho cross-file dependencies (khó nhất)
- Graph reasoning tool đóng góp nhiều nhất (6.31 points khi ablate)