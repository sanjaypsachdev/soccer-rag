"""Ingestion workflow for syncing vector store with datasets directory."""
from langchain_openai import OpenAIEmbeddings

from app.config import Config
from app.document_loader import DocumentLoader
from app.vectorstore import Vectorstore


def run_ingestion():
    """Run the ingestion workflow to sync vector store with datasets directory."""
    print("Starting RAG pipeline for soccer data...")

    # Get configuration
    datasets_dir = Config.get_datasets_dir()
    index_name = Config.get_pinecone_index_name()
    chunk_size = Config.get_chunk_size()
    chunk_overlap = Config.get_chunk_overlap()
    pinecone_api_key = Config.get_pinecone_api_key()
    openai_api_key = Config.get_openai_api_key()

    # Step 1: Initialize document loader
    print("\n[1/3] Initializing document loader...")
    loader = DocumentLoader(
        file_path=str(datasets_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_overlap_strategy="overlap",
    )
    
    # Step 2: Initialize vector store
    print("\n[2/3] Initializing vector store...")
    print("✓ Pinecone API key found")
    print(f"✓ Index name: {index_name}")
    
    vectorstore = Vectorstore(
        index_name=index_name,
        api_key=pinecone_api_key,
    )

    # Initialize embeddings model
    print("✓ OpenAI API key found")
    embeddings_model = OpenAIEmbeddings(
        model=Config.get_embeddings_model_name(),
        openai_api_key=openai_api_key,
    )
    vectorstore.initialize_vectorstore(embeddings_model, show_progress=True)

    # Step 3: Sync vector store with datasets directory
    print("\n[3/3] Syncing vector store with datasets directory...")
    print("This will:")
    print("  - Add documents from new files")
    print("  - Update documents from changed files")
    print("  - Delete documents from removed files")
    print("  - Skip unchanged files")
    print()
    
    stats = vectorstore.sync_with_datasets(loader, datasets_dir, show_progress=True)
    
    print(f"\n✅ Sync completed!")
    print(f"  - Added: {stats['added']} documents from new files")
    print(f"  - Updated: {stats['updated']} documents from changed files")
    print(f"  - Deleted: {stats['deleted']} documents from removed files")
    print(f"  - Unchanged: {stats['unchanged']} documents from unchanged files")

    print("\n✅ RAG pipeline completed successfully!")
    print(f"Vector store ready at index: {index_name}")

