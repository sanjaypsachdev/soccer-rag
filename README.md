# Soccer RAG

RAG pipeline for querying soccer data using LangChain, OpenAI embeddings, and Pinecone vector store.

## Prerequisites

- Python 3.13+
- `uv` package manager
- OpenAI API key
- Pinecone API key

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=soccer-rag
   
   # Optional: Hybrid search configuration
   ENABLE_HYBRID_SEARCH=false  # Set to 'true' to enable hybrid search
   SEMANTIC_WEIGHT=0.7         # Weight for semantic search (0.0-1.0)
   BM25_WEIGHT=0.3              # Weight for BM25 keyword search (0.0-1.0)
   ```

## Usage

### Ingestion

Load PDFs from `datasets/`, chunk, embed, and ingest into Pinecone:

```bash
uv run python main.py --ingest
```

Processes:
- Recursively loads PDF files from `datasets/`
- Chunks documents using `RecursiveCharacterTextSplitter` (1000 chars, 100 overlap)
- Generates embeddings via `text-embedding-3-small`
- Syncs with Pinecone index (creates if missing)

### Chatbot

Interactive Q&A session with chat history:

```bash
uv run python main.py --chat
```

Uses:
- `gpt-4o-mini` for answer generation
- `text-embedding-3-small` for query embeddings
- Retrieves top 4 relevant chunks (configurable via `TOP_K` environment variable)

### Hybrid Search

The pipeline supports **hybrid search** that combines semantic similarity search (via Pinecone embeddings) with keyword-based search (BM25). This improves retrieval for queries requiring exact term matching, numerical data, or specific keywords.

**To enable hybrid search:**
1. Set `ENABLE_HYBRID_SEARCH=true` in your `.env` file
2. Optionally adjust weights:
   - `SEMANTIC_WEIGHT`: Weight for semantic search (default: 0.7)
   - `BM25_WEIGHT`: Weight for BM25 keyword search (default: 0.3)
   - Weights should typically sum to 1.0 for best results

**Benefits:**
- **Semantic search** excels at understanding context and meaning
- **BM25 search** excels at exact keyword matching and numerical queries
- **Combined** they provide better coverage for diverse query types

**Example queries that benefit from hybrid search:**
- "Premier League revenue 2024" (keyword-heavy)
- "What were the main financial highlights?" (semantic-heavy)
- "How much did broadcasting rights generate?" (mixed)

## Chunking Strategy

The pipeline currently uses **RecursiveCharacterTextSplitter** from LangChain to split documents into manageable chunks:

### Configuration
- **Chunk Size**: 1000 characters (default, configurable via `Config.CHUNK_SIZE`)
- **Chunk Overlap**: 100 characters (default, configurable via `Config.CHUNK_OVERLAP`)

### Process
1. **PDF Processing**: Documents are processed page-by-page using PyMuPDF
2. **Content Extraction**: Each page's content (text, tables, and images) is extracted
3. **Chunking**: The `RecursiveCharacterTextSplitter` intelligently splits text at natural boundaries (paragraphs, sentences, etc.) while respecting the chunk size and overlap constraints
4. **Metadata**: Each chunk includes:
   - `source_file`: Full path to the source PDF
   - `file_name`: Name of the PDF file
   - `file_hash`: SHA256 hash for change detection
   - `modified_time`: File modification timestamp
   - `page_number`: Page number from the source document

## Architecture

```
app/
├── config.py          # Environment configuration & hybrid search settings
├── document_loader.py # PDF loading & chunking (PyMuPDF)
├── vectorstore.py     # Pinecone operations & hybrid search (semantic + BM25)
├── ingestion.py       # Ingestion workflow
└── chatbot.py         # Interactive Q&A with conversation history
```

## Dependencies

- `langchain`, `langchain-openai`, `langchain-pinecone`, `langchain-community`, `langchain-classic`
- `pinecone-client`
- `pymupdf` (PDF processing)
- `rank-bm25` (BM25 keyword search for hybrid retrieval)
- `python-dotenv`

## Features

- ✅ **Hybrid Search**: Combines semantic (embedding-based) and keyword (BM25) retrieval for improved accuracy
- ✅ **Intelligent Chunking**: Recursive text splitting with overlap for context preservation
- ✅ **Conversation History**: Maintains context across multiple questions in a session
- ✅ **Incremental Sync**: Only updates changed documents during ingestion
- ✅ **Metadata Preservation**: Tracks source files, page numbers, and file hashes

## Possible Improvements

- **Evaluation Framework**: Implement a comprehensive evaluation system to measure retrieval quality (precision, recall) and answer accuracy using benchmark question-answer pairs from the Premier League reports
- **Web Interface**: Develop a web-based UI (using Streamlit or FastAPI + React) to make the chatbot more accessible and provide features like conversation history persistence, document visualization, and export capabilities
- **Advanced Reranking**: Implement cross-encoder models for better result reranking
- **Query Expansion**: Add query expansion techniques to improve retrieval recall
