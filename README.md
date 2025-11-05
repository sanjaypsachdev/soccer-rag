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
- Retrieves top 4 relevant chunks (configurable via `Config.K`)

## Chunking Strategy

The pipeline uses **RecursiveCharacterTextSplitter** from LangChain to split documents into manageable chunks:

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
├── config.py          # Environment configuration
├── document_loader.py # PDF loading & chunking (PyMuPDF)
├── embedder.py        # OpenAI embeddings
├── vectorstore.py     # Pinecone operations
├── ingestion.py       # Ingestion workflow
└── chatbot.py         # Interactive Q&A
```

## Dependencies

- `langchain`, `langchain-openai`, `langchain-pinecone`
- `pinecone-client`
- `pymupdf` (PDF processing)
- `python-dotenv`
