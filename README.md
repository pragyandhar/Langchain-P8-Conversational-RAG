# Conversational RAG with LangChain

A intelligent document Q&A system that combines **Retrieval-Augmented Generation (RAG)** with **Conversational Memory** to understand context and answer questions strictly based on your documents.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)

---

## Overview

This project implements a sophisticated conversational retrieval system that:
- **Loads and processes** PDF and TXT documents
- **Understands context** from previous conversations
- **Intelligently retrieves** the most relevant information
- **Answers questions** exclusively from your provided documents
- **Maintains conversation history** for seamless multi-turn dialogue

Perfect for building intelligent chatbots, document analysis tools, and context-aware Q&A systems.

---

## Key Features

### Smart Question Understanding
- **Context-Aware Processing**: Questions are automatically rewritten to resolve pronouns and implicit references
- **History-Aware Retrieval**: The system understands follow-up questions by condensing them into standalone queries
- **Conversation Memory**: Full conversation history is maintained across sessions

### Document Processing
- **Multi-Format Support**: Seamlessly handle PDF and TXT files
- **Intelligent Chunking**: Documents are split into manageable chunks with overlap for context preservation
- **Vector Embeddings**: Uses OpenAI's embeddings for semantic search

### Retrieval & Answering
- **FAISS Vector Store**: Ultra-fast similarity search across documents
- **Document-Grounded**: Answers strictly from provided documents—no hallucination
- **Source Tracking**: Each answer includes citations to source documents

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL RAG PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. DOCUMENT LOADING                                             │
│     ├─ PDF Files (PyPDFLoader)                                  │
│     └─ TXT Files (TextLoader)                                   │
│            ↓                                                      │
│  2. TEXT SPLITTING                                               │
│     └─ RecursiveCharacterTextSplitter (chunks + overlap)         │
│            ↓                                                      │
│  3. VECTORIZATION                                                │
│     └─ OpenAI Embeddings (text-embedding-3-small)              │
│            ↓                                                      │
│  4. VECTOR STORE                                                 │
│     └─ FAISS (Fast similarity search)                            │
│            ↓                                                      │
│  5. CONVERSATIONAL CHAIN                                         │
│     ├─ History-Aware Retriever (question condensing)            │
│     ├─ Retrieved Documents Combination                           │
│     ├─ LLM Response Generation (GPT-4o-mini)                    │
│     └─ Message History Management                               │
│            ↓                                                      │
│  6. OUTPUT WITH SOURCES                                          │
│     └─ Answer + Source Documents                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Langchain-P8-Conversational-RAG/
├── main.py              # Main entry point & orchestration
├── loader.py            # Document loading (PDF & TXT)
├── splitter.py          # Text chunking with overlap
├── vectorstore.py       # Vector store & retriever setup
├── chain.py             # RAG chain with conversation memory
├── sample.txt           # Sample document (AI/ML concepts)
├── assets/              # Additional resources
├── README.md            # This file
└── .env                 # Environment variables (create locally)
```

### File Descriptions

| File | Purpose |
|------|---------|
| **main.py** | Orchestrates the entire pipeline: loads documents, builds vectorstore, runs the chat loop |
| **loader.py** | Loads PDF and TXT files with proper error handling |
| **splitter.py** | Splits documents into chunks (1000 chars) with 200-char overlap |
| **vectorstore.py** | Creates FAISS vector store and retriever with OpenAI embeddings |
| **chain.py** | Builds the RAG chain with question condensing, document retrieval, and LLM response |

---

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pragyandhar/Langchain-P8-Conversational-RAG.git
   cd Langchain-P8-Conversational-RAG
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install langchain langchain-openai langchain-text-splitters langchain-community python-dotenv faiss-cpu
   ```
   
   For GPU acceleration with FAISS:
   ```bash
   pip install faiss-gpu
   ```

4. **Set up environment variables**
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

---

## Usage

### Running the Application

```bash
python main.py
```

### Interactive Chat Session

1. **Provide documents**: Enter file paths (PDF or TXT) one by one. Press Enter to finish.
   ```
   Enter file paths one by one. Press Enter with no input when done.
   File 1: /path/to/document1.pdf
   File 2: /path/to/document2.txt
   File 3: 
   ```

2. **Ask questions**: Once loading is complete, ask questions about your documents.
   ```
   You: What is machine learning?
   Assistant: [Answer based on documents]
   [Sources: document1.pdf (Page 1) | document2.txt]
   ```

### Special Commands

| Command | Function |
|---------|----------|
| `history` | View the complete chat history for the current session |
| `quit` | Exit the conversation |

### Example Conversation

```
You: What is artificial intelligence?
Assistant: Artificial intelligence (AI) is intelligence demonstrated by machines... [answer continues]
[Sources: sample.txt]

You: How is it different from machine learning?
Assistant: Machine learning is a subset of AI that... [answer continues]
[Sources: sample.txt]

You: Can you give me a concrete example?
Assistant: Some AI applications include advanced web search engines... [answer continues]
[Sources: sample.txt]
```

---

## How It Works

### Step 1: Document Ingestion
- Reads PDF and TXT files
- Preserves metadata (source, page numbers)

### Step 2: Text Chunking
```python
# Documents are split into 1000-character chunks with 200-char overlap
chunk_size=1000, chunk_overlap=200
```
This ensures context is preserved across chunk boundaries.

### Step 3: Vector Embeddings
- Each chunk is converted to a vector using OpenAI's `text-embedding-3-small` model
- Vectors capture semantic meaning for similarity search

### Step 4: Question Condensing
The system automatically condenses multi-turn questions by:
- Resolving pronouns ("it", "they", "this")
- Expanding demonstratives ("the above", "the previous")
- Creating fully self-contained questions without losing intent

Example:
```
Chat History: "User: What is AI?"
Latest Question: "Can you explain how it works?"
→ Condensed: "Can you explain how Artificial Intelligence works?"
```

### Step 5: Retrieval & Answering
1. **Retrieve**: FAISS finds top-3 most similar chunks to the question
2. **Combine**: Retrieved documents are formatted as context
3. **Generate**: GPT-4o-mini generates response based on context and history
4. **Ground**: Response is strictly based on provided documents

### Step 6: Session Management
- Each conversation gets a unique session ID
- Full message history is maintained in memory
- History is used for context-aware question condensing

---

## Configuration

### Tuning Parameters

#### In `splitter.py`:
```python
chunk_size=1000       # Size of each chunk (characters)
chunk_overlap=200     # Overlap between chunks
```

#### In `vectorstore.py`:
```python
k=3                   # Number of chunks to retrieve per query
```

#### In `chain.py`:
```python
model="gpt-4o-mini"   # LLM model (can change to gpt-4 for more accuracy)
temperature=0         # Deterministic responses (0-1 scale)
```

### Environment Variables

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=sk-...
```

---

## Performance Notes

- **Embedding Generation**: ~0.1-0.5 seconds per 1000 tokens
- **Retrieval Time**: <100ms for FAISS search
- **Response Generation**: 1-5 seconds depending on model

---

## Privacy & Security

- All documents are processed locally during chunking
- Embeddings and LLM calls go through OpenAI API
- Session history is stored in-memory (not persisted)
- No documents are stored after the session ends

---

## Future Enhancements

- [ ] Persistent session storage (database)
- [ ] Support for more file formats (DOCX, HTML, Markdown)
- [ ] Streaming responses for real-time feedback
- [ ] Batch PDF processing with progress bars
- [ ] Custom embeddings model support
- [ ] Query result caching for performance
- [ ] Web UI interface (Streamlit/Gradio)

---

## Sample Data

The project includes `sample.txt` with content about AI, ML, and Deep Learning to test the system.

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://openai.com/docs/api)
- [FAISS Vector Database](https://github.com/facebookresearch/faiss)
- [RAG Explained](https://python.langchain.com/docs/use_cases/question_answering/)

---

## Tips for Best Results

1. **Use quality documents**: Clear, well-structured documents work best
2. **Chunk size tuning**: Larger chunks preserve more context, smaller chunks are more precise
3. **Number of retrievals**: Increase `k` in `vectorstore.py` if missing relevant context
4. **Temperature setting**: Keep at 0 for factual answers, increase for creative responses
5. **Follow-up questions**: Leverage conversation memory by asking follow-ups

---

## License

This project is created for educational purposes.

---

## Author

Created as Project 8 combining RAG and Conversational Memory concepts.

---

Happy document querying!
