# 🚀 LangChain Learning Roadmap - Complete Guide

## 📋 Tổng quan

Roadmap học LangChain từ cơ bản đến nâng cao dành cho developers có 2+ năm kinh nghiệm Python. Được thiết kế để học một cách có hệ thống và thực tế.

---

## 🎯 **Level 1: Core Foundations (1-2 tuần)**

### **1.1 LangChain Basics**

#### **Khái niệm cốt lõi cần nắm vững:**

- **Document Loaders**: Load dữ liệu từ nhiều nguồn
  - `TextLoader`, `PyPDFLoader`, `WebBaseLoader`
  - `CSVLoader`, `JSONLoader`, `DirectoryLoader`
  - Custom loaders cho data sources đặc biệt

- **Text Splitters**: Chia nhỏ documents thành chunks
  - `RecursiveCharacterTextSplitter` (recommended)
  - `CharacterTextSplitter`, `TokenTextSplitter`
  - Language-specific splitters

- **Embeddings**: Chuyển text thành vectors
  - `OpenAIEmbeddings` (commercial)
  - `HuggingFaceEmbeddings` (free)
  - Custom embedding models

- **Vector Stores**: Lưu trữ và tìm kiếm vectors
  - `Chroma` (local development)
  - `FAISS` (high performance)
  - `Pinecone` (production scale)

- **Retrievers**: Tìm kiếm documents liên quan
  - Similarity search
  - MMR (Maximum Marginal Relevance)
  - Threshold-based retrieval

- **LLMs/Chat Models**: Tích hợp các mô hình AI
  - `ChatOpenAI`, `ChatAnthropic`
  - `ChatOllama` (local models)
  - Model fallbacks và error handling

- **Prompts & Templates**: Quản lý prompts hiệu quả
  - `PromptTemplate`, `ChatPromptTemplate`
  - Few-shot prompting
  - Dynamic prompt generation

#### **Thực hành:**
```python
# Tạo basic RAG pipeline
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# Load → Split → Embed → Store → Retrieve → Generate
```

### **1.2 LCEL (LangChain Expression Language)**

#### **Tại sao LCEL quan trọng:**
- Cú pháp hiện đại nhất của LangChain (từ v0.1.0+)
- Dễ debug và compose chains
- Native support cho streaming và async
- Better error handling

#### **Core concepts:**

- **Runnable Interface**: Base class cho tất cả components
  - `.invoke()`: Sync execution
  - `.ainvoke()`: Async execution
  - `.stream()`: Streaming responses
  - `.batch()`: Batch processing

- **Chain Composition với `|`**:
  ```python
  chain = prompt | llm | output_parser
  ```

- **RunnablePassthrough**: Pass input unchanged
  ```python
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
  ```

- **RunnableParallel**: Execute multiple chains parallel
  ```python
  RunnableParallel(context=retriever, question=RunnablePassthrough())
  ```

#### **Thực hành:**
```python
# Modern RAG với LCEL
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 🔧 **Level 2: RAG Mastery (2-3 tuần)**

### **2.1 RAG Components Deep Dive**

#### **Document Processing:**

- **PDF Processing**:
  - `PyPDFLoader` vs `PyMuPDFLoader`
  - Handle scanned PDFs với OCR
  - Extract tables và images

- **Web Scraping**:
  - `WebBaseLoader` với custom parsing
  - Handle dynamic content (JavaScript)
  - Respect robots.txt và rate limiting

- **Database Integration**:
  - SQL databases với SQLDatabaseLoader
  - NoSQL databases (MongoDB, etc.)
  - Real-time data integration

#### **Chunking Strategies:**

- **Smart Chunking**:
  - Document structure aware
  - Preserve semantic boundaries
  - Overlap strategies

- **Content-Type Specific**:
  - Code splitting (language-aware)
  - Academic papers (section-based)
  - Conversations (speaker-aware)

#### **Vector Databases Comparison:**

| Database | Best For | Performance | Cost |
|----------|----------|-------------|------|
| Chroma | Development, Local | Medium | Free |
| FAISS | High Performance, Local | High | Free |
| Pinecone | Production, Scale | High | Paid |
| Weaviate | Complex Metadata | High | Mixed |
| Qdrant | Full-text + Vector | High | Mixed |

### **2.2 Advanced RAG Techniques**

#### **Multi-Query Retrieval:**
- Generate multiple variations of user query
- Retrieve for each variation
- Combine and deduplicate results
```python
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
```

#### **Contextual Compression:**
- Remove irrelevant information from retrieved docs
- Extract only relevant passages
- Reduce token usage và improve accuracy
```python
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

#### **Hybrid Search:**
- Combine semantic search (vectors) with keyword search (BM25)
- Better handling of specific terms và names
- Weighted combination of results
```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

#### **Re-ranking:**
- Post-process retrieved documents
- Use specialized models for relevance scoring
- Cross-encoder models for better ranking

#### **Metadata Filtering:**
- Filter by document properties
- Time-based filtering
- Source-based filtering
```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "specific_document.pdf"}
    }
)
```

---

## 🤖 **Level 3: Agents & Tools (2-3 tuần)**

### **3.1 LangChain Agents**

#### **Agent Types:**

- **ReAct (Reasoning + Acting)**:
  - Most common pattern
  - Observation → Thought → Action loop
  - Great for tool usage

- **Plan-and-Execute**:
  - High-level planning
  - Step-by-step execution
  - Better for complex tasks

- **Conversational Agents**:
  - Maintain conversation context
  - Memory integration
  - Multi-turn interactions

#### **Agent Components:**

- **Agent Executor**: Runs the agent với safety controls
- **Tools**: Functions agent có thể call
- **Memory**: Maintain state across interactions
- **Callbacks**: Monitor agent execution

#### **Custom Agent Development:**
```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
```

### **3.2 Tools Ecosystem**

#### **Built-in Tools:**

- **Web Search**:
  - `TavilySearchResults`: Professional search API
  - `DuckDuckGoSearchResults`: Free alternative
  - `GoogleSearchResults`: Requires API key

- **Database Tools**:
  - `SQLDatabaseToolkit`: Query SQL databases
  - `QuerySQLDataBaseTool`: Execute specific queries
  - Custom database connectors

- **File Operations**:
  - `ReadFileTool`, `WriteFileTool`
  - `ListDirectoryTool`
  - Cloud storage integration

- **API Tools**:
  - `RequestsGetTool`, `RequestsPostTool`
  - Custom API wrappers
  - Authentication handling

- **Math & Code**:
  - `PythonREPLTool`: Execute Python code
  - `LLMMathTool`: Mathematical calculations
  - `ShellTool`: System commands

#### **Custom Tools Development:**
```python
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class CustomCalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CustomCalculatorInput

    def _run(self, a: int, b: int) -> str:
        return f"{a} + {b} = {a + b}"

    async def _arun(self, a: int, b: int) -> str:
        return self._run(a, b)
```

#### **Tool Integration Patterns:**

- **Tool Chaining**: Sequential tool usage
- **Parallel Tool Execution**: Multiple tools simultaneously
- **Conditional Tool Usage**: Based on context
- **Error Handling**: Graceful failure recovery

---

## 🧠 **Level 4: Memory & Conversation (1-2 tuần)**

### **4.1 Memory Types**

#### **ConversationBufferMemory:**
- Stores all conversation messages
- Simple but can grow large
- Good for short conversations
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

#### **ConversationBufferWindowMemory:**
- Keep only last N interactions
- Fixed memory size
- Good for long conversations
```python
memory = ConversationBufferWindowMemory(
    k=5,  # Keep last 5 interactions
    return_messages=True
)
```

#### **ConversationSummaryMemory:**
- Summarize old conversations
- Dynamic memory management  
- Best for very long conversations
```python
memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)
```

#### **VectorStoreRetrieverMemory:**
- Store conversations in vector database
- Semantic search through history
- Great for contextual recall
```python
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs=dict(k=1))
)
```

### **4.2 Conversation Patterns**

#### **History-Aware Retrieval:**
- Reformulate questions based on chat history
- Maintain context across turns
- Handle references và pronouns
```python
from langchain.chains import create_history_aware_retriever

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

#### **Session Management:**
- Multiple users/sessions
- Session isolation
- Persistent storage
```python
# Session-based memory
sessions = {}
def get_session_memory(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferMemory()
    return sessions[session_id]
```

#### **Context-Aware Responses:**
- Reference previous messages
- Maintain conversation flow
- Handle topic switches

---

## 🏗️ **Level 5: Production & Advanced (2-3 tuần)**

### **5.1 LangSmith & Monitoring**

#### **Tracing & Debugging:**
- Trace chain execution step-by-step
- Debug failed runs
- Performance profiling
```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"

from langsmith import traceable

@traceable
def my_rag_function(question: str):
    # Your RAG logic here
    pass
```

#### **Evaluation & Metrics:**
- RAGAS framework integration
- Custom evaluation metrics
- A/B testing setup
```python
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

# Evaluate RAG system
results = evaluate(
    dataset=eval_dataset,
    metrics=[answer_relevancy, faithfulness]
)
```

#### **Production Monitoring:**
- Error tracking
- Performance metrics
- Cost monitoring
- Usage analytics

### **5.2 LangGraph (Advanced State Management)**

#### **Why LangGraph:**
- Complex workflows với state
- Conditional routing
- Human-in-the-loop patterns
- Multi-agent coordination

#### **Core Concepts:**
- **State**: Shared data structure
- **Nodes**: Processing functions
- **Edges**: Transitions between nodes
- **Conditional Edges**: Dynamic routing

```python
from langgraph.graph import StateGraph, MessagesState

# Define workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

#### **Advanced Patterns:**
- **Multi-Agent Systems**: Agents collaborate
- **Human-in-the-Loop**: User approval steps
- **Error Recovery**: Automatic retry logic
- **Parallel Processing**: Concurrent execution

### **5.3 Production Deployment**

#### **LangServe (API Deployment):**
```python
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI()
add_routes(app, rag_chain, path="/rag")

# Deploy với uvicorn app:app --host 0.0.0.0 --port 8000
```

#### **Performance Optimization:**

- **Caching Strategies**:
  - Response caching
  - Embedding caching
  - Query result caching

- **Async & Streaming**:
  - Non-blocking operations
  - Real-time responses
  - Better user experience

- **Load Balancing**:
  - Multiple model endpoints
  - Request distribution
  - Failover handling

#### **Security Best Practices:**

- **API Security**:
  - Authentication & authorization
  - Rate limiting
  - Input sanitization

- **Data Privacy**:
  - PII handling
  - Data encryption
  - Audit logging

- **Model Security**:
  - Prompt injection prevention
  - Output filtering
  - Content moderation

---

## 📊 **Specialized Areas**

### **6.1 Document Analysis**

#### **Document QA:**
- Multi-document question answering
- Citation và source tracking
- Confidence scoring

#### **Summarization:**
- Extractive vs abstractive
- Multi-document summarization
- Hierarchical summarization

#### **Classification & Extraction:**
- Document classification
- Named entity recognition
- Information extraction

### **6.2 Code Generation & Analysis**

#### **Code Understanding:**
- Parse và analyze codebases
- Generate documentation
- Code review automation

#### **Code Generation:**
- Generate code từ requirements
- Test generation
- Code refactoring suggestions

### **6.3 Multimodal Applications**

#### **Vision Integration:**
- Image analysis với LLMs
- OCR và document processing
- Chart và diagram understanding

#### **Audio Processing:**
- Speech-to-text integration
- Audio analysis
- Multimodal conversations

---

## 🗺️ **Learning Path Timeline**

### **Tuần 1-2: Foundations**
- [ ] Document loaders & text splitters
- [ ] Embeddings & vector stores setup
- [ ] Basic RAG pipeline implementation
- [ ] LCEL syntax mastery
- [ ] **Project**: Simple document QA system

### **Tuần 3-4: RAG Deep Dive**
- [ ] Advanced retrieval strategies
- [ ] Multi-query & contextual compression
- [ ] Hybrid search implementation
- [ ] Performance optimization
- [ ] **Project**: Advanced RAG với multiple techniques

### **Tuần 5-6: Agents & Tools**
- [ ] Agent patterns (ReAct, Plan-Execute)
- [ ] Built-in tools integration
- [ ] Custom tools development
- [ ] Agent memory & state management
- [ ] **Project**: AI assistant với multiple tools

### **Tuần 7-8: Memory & Conversation**
- [ ] Conversation memory types
- [ ] History-aware retrieval
- [ ] Session management
- [ ] Context preservation
- [ ] **Project**: Conversational RAG system

### **Tuần 9-10: Production & Advanced**
- [ ] LangSmith tracing & evaluation
- [ ] Error handling & monitoring
- [ ] API deployment với LangServe
- [ ] Performance optimization
- [ ] **Project**: Production-ready RAG API

### **Tuần 11-12: Specialization**
- [ ] Choose specialization area
- [ ] Advanced techniques exploration
- [ ] LangGraph for complex workflows
- [ ] Multi-agent systems
- [ ] **Project**: Specialized application

---

## 📚 **Learning Resources**

### **Official Documentation**
- [LangChain Python Docs](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangSmith Documentation](https://docs.smith.langchain.com/)

### **Video Resources**
- [LangChain YouTube Channel](https://www.youtube.com/@LangChain)
- [Harrison Chase Presentations](https://www.youtube.com/results?search_query=harrison+chase+langchain)
- [AI Engineer Summit Talks](https://www.youtube.com/@aiengineersummit)

### **Community & Practice**
- [LangChain Discord](https://discord.com/invite/langchain)
- [GitHub Examples](https://github.com/langchain-ai/langchain/tree/master/templates)
- [Reddit r/LangChain](https://www.reddit.com/r/LangChain/)

### **Books & Papers**
- "Building LLM Applications" papers
- RAG research papers
- Agent-based systems research

---

## 🎯 **Project Ideas by Level**

### **Beginner Projects:**
1. **Personal Document Assistant**: RAG trên personal documents
2. **Website QA Bot**: Scrape website và answer questions
3. **PDF Analyzer**: Upload PDF và extract insights

### **Intermediate Projects:**
1. **Multi-Source Knowledge Assistant**: Combine multiple data sources
2. **Code Documentation Generator**: Auto-generate docs từ code
3. **Research Assistant**: Academic paper analysis

### **Advanced Projects:**
1. **Customer Support Automation**: Multi-step problem resolution
2. **Content Creation Pipeline**: Research → outline → writing
3. **Business Intelligence Assistant**: Data analysis + natural language

### **Expert Projects:**
1. **Multi-Agent Research Team**: Specialized agents collaborate
2. **Enterprise Knowledge Platform**: Company-wide knowledge system
3. **AI-Powered Workflow Automation**: Complex business processes

---

## 💡 **Best Practices & Tips**

### **Development Tips:**
- Start simple, add complexity gradually
- Always handle errors gracefully
- Monitor token usage và costs
- Test with diverse inputs
- Version control your prompts

### **Performance Optimization:**
- Use appropriate chunk sizes (experiment!)
- Implement caching strategies
- Monitor retrieval quality
- Optimize prompt length
- Use async when possible

### **Production Readiness:**
- Comprehensive error handling
- Monitoring và alerting
- Security best practices
- Scalability planning
- User feedback collection

### **Learning Strategy:**
- Build projects while learning
- Join community discussions
- Contribute to open source
- Stay updated với releases
- Practice prompt engineering

---

## 🚀 **Next Steps**

1. **Set Learning Goals**: Define specific objectives
2. **Choose Starting Point**: Based on current knowledge
3. **Create Learning Schedule**: Consistent daily practice
4. **Build Portfolio**: Document projects và learnings
5. **Join Community**: Connect with other learners
6. **Stay Updated**: Follow LangChain releases

---

## 📞 **Support & Questions**

Nếu có câu hỏi trong quá trình học:

1. **Check Documentation**: Official docs first
2. **Search Community**: Discord, Reddit, Stack Overflow
3. **GitHub Issues**: For bugs và feature requests
4. **Practice Projects**: Learn by building
5. **Code Review**: Share code for feedback

---

**Happy Learning! 🎉**

*Remember: Consistency beats intensity. 30 minutes daily is better than 3 hours once a week.*