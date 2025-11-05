"""Vectorstore class for managing Pinecone vector database operations."""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Constants
DEFAULT_EMBEDDING_DIMENSION = 1536
PROGRESS_UPDATE_INTERVAL = 5
DEFAULT_BATCH_SIZE = 100
MAX_QUERY_RESULTS = 10000


class Vectorstore:
    """Manages Pinecone vector database operations."""

    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Vectorstore.

        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
        """
        self.index_name = index_name
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter."
            )

        try:
            self.pinecone_client = Pinecone(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Pinecone client: {e}")

        self.vectorstore = None
        self._index = None
        self._embedding_dimension = None

    def _ensure_initialized(self):
        """Ensure vectorstore is initialized."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call initialize_vectorstore first.")

    def _log(self, message: str, show_progress: bool = True):
        """Log a message if progress is enabled."""
        if show_progress:
            print(message)

    def _get_embedding_dimension(self, embeddings) -> int:
        """Get the embedding dimension from the embeddings model."""
        try:
            test_embedding = embeddings.embed_query("test")
            return len(test_embedding)
        except Exception:
            return DEFAULT_EMBEDDING_DIMENSION

    def _get_index_names(self) -> List[str]:
        """Get list of all index names from Pinecone."""
        try:
            indexes = self.pinecone_client.list_indexes()
            if hasattr(indexes, '__iter__'):
                return [idx.name if hasattr(idx, 'name') else str(idx) for idx in indexes]
            return []
        except Exception as e:
            raise ConnectionError(
                f"Failed to list Pinecone indexes. Please check your API key and internet connection. "
                f"Error: {str(e)}"
            )

    def _is_index_ready(self, index_info) -> bool:
        """Check if index is ready using multiple fallback methods."""
        # Check status attribute
        if hasattr(index_info, 'status'):
            status = index_info.status
            if isinstance(status, dict):
                if status.get('ready', False):
                    return True
            elif isinstance(status, str) and status.lower() in ['ready', 'readyforquery']:
                return True
        
        # Check ready attribute directly
        if hasattr(index_info, 'ready'):
            return bool(index_info.ready)
        
        return False

    def _wait_for_index_ready(self, show_progress: bool = True):
        """Wait for index to become ready."""
        start_time = time.time()
        last_progress_time = 0

        while True:
            try:
                index_info = self.pinecone_client.describe_index(self.index_name)
                if self._is_index_ready(index_info):
                    return True
            except Exception:
                pass  # Continue waiting

            # Show progress every 5 seconds
            elapsed = time.time() - start_time
            if show_progress and int(elapsed) - last_progress_time >= PROGRESS_UPDATE_INTERVAL:
                print(f"   Still waiting... ({int(elapsed)}s elapsed)")
                last_progress_time = int(elapsed)

            time.sleep(2)  # Wait 2 seconds between checks

    def _get_index_dimension(self, index_info) -> Optional[int]:
        """Extract dimension from index info using multiple fallback methods."""
        if hasattr(index_info, 'dimension'):
            return index_info.dimension
        if hasattr(index_info, 'spec') and hasattr(index_info.spec, 'dimension'):
            return index_info.spec.dimension
        if isinstance(index_info, dict):
            return index_info.get('dimension')
        return None

    def _verify_index_dimension(self, show_progress: bool = True):
        """Verify that existing index has the correct dimension."""
        try:
            index_info = self.pinecone_client.describe_index(self.index_name)
            existing_dimension = self._get_index_dimension(index_info)

            if existing_dimension and existing_dimension != self._embedding_dimension:
                raise ValueError(
                    f"Index dimension mismatch! "
                    f"Existing index '{self.index_name}' has dimension {existing_dimension}, "
                    f"but embeddings require dimension {self._embedding_dimension}. "
                    f"Please delete the existing index manually or create a new index with the correct dimension."
                )
            if existing_dimension:
                self._log(f"   ✓ Index dimension matches: {existing_dimension}", show_progress)
        except ValueError:
            raise
        except Exception as e:
            self._log(f"   ⚠️  Could not verify index dimension: {e}", show_progress)

    def _create_index(self, show_progress: bool = True):
        """Create a new Pinecone index."""
        self._log(f"   Index '{self.index_name}' not found. Creating new index...", show_progress)

        self.pinecone_client.create_index(
            name=self.index_name,
            dimension=self._embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        self._log("   Index created. Waiting for it to be ready...", show_progress)
        self._wait_for_index_ready(show_progress)
        self._log("   ✓ Index is ready", show_progress)

    def _initialize_pinecone_index(self, show_progress: bool = True):
        """Initialize connection to Pinecone index, creating if needed."""
        self._log(f"   Checking if index '{self.index_name}' exists...", show_progress)

        index_names = self._get_index_names()

        if self.index_name not in index_names:
            self._create_index(show_progress)
        else:
            self._log(f"   ✓ Index '{self.index_name}' exists", show_progress)
            self._verify_index_dimension(show_progress)

    def _setup_vectorstore(self, embeddings):
        """Set up the LangChain vectorstore and underlying Pinecone index."""
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings,
        )

        # Store reference to underlying Pinecone index
        try:
            self._index = self.pinecone_client.Index(self.index_name)
        except Exception:
            try:
                self._index = self.vectorstore._index
            except Exception:
                pass

    def initialize_vectorstore(self, embeddings, show_progress: bool = True):
        """
        Initialize the LangChain Pinecone vector store.
        Creates the index if it doesn't exist.
        If index exists with wrong dimension, raises a ValueError.

        Args:
            embeddings: Embeddings instance (e.g., OpenAIEmbeddings)
            show_progress: Whether to show progress messages (default: True)
            
        Raises:
            ValueError: If the existing index has a dimension mismatch
            ConnectionError: If connection to Pinecone fails
        """
        self._log("   Connecting to Pinecone...", show_progress)
        
        self._embedding_dimension = self._get_embedding_dimension(embeddings)
        self._log(f"   Embedding dimension: {self._embedding_dimension}", show_progress)
        
        try:
            self._initialize_pinecone_index(show_progress)
            self._log("   Initializing vector store connection...", show_progress)
            self._setup_vectorstore(embeddings)
            self._log("   ✓ Vector store initialized successfully", show_progress)
        except ConnectionError:
            raise
        except ValueError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                raise ConnectionError(
                    f"Failed to connect to Pinecone. Please check your internet connection and API key. "
                    f"Error: {error_msg}"
                )
            raise ValueError(f"Failed to initialize vector store: {error_msg}")

    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Add documents to the vector store.
        Embeddings are generated automatically by the vectorstore.

        Args:
            documents: List of Document objects (can include metadata)
            batch_size: Number of documents to process in each batch
        """
        self._ensure_initialized()

        if not documents:
            return

        for batch_idx in range(0, len(documents), batch_size):
            batch = documents[batch_idx:batch_idx + batch_size]
            self.vectorstore.add_documents(batch)

    def add_texts(self, texts: List[str]):
        """Add texts directly to the vector store."""
        self._ensure_initialized()
        self.vectorstore.add_texts(texts)

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """Perform similarity search in the vector store."""
        self._ensure_initialized()
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Perform similarity search with scores."""
        self._ensure_initialized()
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, k: int = 4, search_kwargs: Optional[Dict] = None):
        """Get a retriever from the vectorstore."""
        self._ensure_initialized()
        if search_kwargs is None:
            search_kwargs = {"k": k}
        else:
            search_kwargs.setdefault("k", k)
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def _create_dummy_vector(self) -> List[float]:
        """Create a dummy zero vector for queries that require a vector."""
        dimension = self._embedding_dimension or DEFAULT_EMBEDDING_DIMENSION
        return [0.0] * dimension

    def _query_with_filter(self, filter_dict: Optional[Dict] = None) -> List:
        """Query index with optional metadata filter."""
        if not self._index:
            return []
        
        try:
            dummy_vector = self._create_dummy_vector()
            results = self._index.query(
                vector=dummy_vector,
                top_k=MAX_QUERY_RESULTS,
                include_metadata=True,
                filter=filter_dict
            )
            return results.matches
        except Exception as e:
            print(f"Error querying index: {e}")
            return []

    def delete_by_source_file(self, source_file: str) -> int:
        """Delete all documents from the vector store that came from a specific source file."""
        self._ensure_initialized()
        
        if not self._index:
            raise ValueError("Pinecone index not accessible. Cannot delete documents.")
        
        try:
            documents = self.get_documents_by_source_file(source_file)
            count = len(documents)
            
            if count > 0:
                self._index.delete(filter={"source_file": source_file})
            
            return count
        except Exception as e:
            print(f"Error deleting documents from {source_file}: {e}")
            return 0
    
    def get_documents_by_source_file(self, source_file: str) -> List[Document]:
        """Get all documents from the vector store that came from a specific source file."""
        self._ensure_initialized()
        
        matches = self._query_with_filter(filter_dict={"source_file": source_file})
        
        documents = []
        for match in matches:
            metadata = match.metadata or {}
            text = metadata.get('text', '') or metadata.get('page_content', '')
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    def get_all_stored_source_files(self) -> Set[str]:
        """Get all unique source file paths currently stored in the vector store."""
        if not self._index:
            return set()
        
        matches = self._query_with_filter()
        source_files = set()
        
        for match in matches:
            if match.metadata and 'source_file' in match.metadata:
                source_files.add(match.metadata['source_file'])
        
        return source_files
    
    def _group_documents_by_source(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by their source file."""
        file_documents: Dict[str, List[Document]] = {}
        for doc in documents:
            source_file = doc.metadata.get("source_file")
            if source_file:
                file_documents.setdefault(source_file, []).append(doc)
        return file_documents

    def _get_file_hash(self, documents: List[Document]) -> str:
        """Extract file hash from document metadata."""
        if documents and documents[0].metadata:
            return documents[0].metadata.get("file_hash", "")
        return ""

    def _process_file_sync(
        self, 
        source_file: str, 
        file_path: Path, 
        current_docs: List[Document],
        stats: Dict[str, int],
        show_progress: bool = True,
        iterator=None
    ):
        """Process a single file for sync operation."""
        if not current_docs:
            return
        
        file_name = Path(source_file).name
        current_hash = self._get_file_hash(current_docs)
        stored_docs = self.get_documents_by_source_file(source_file)
        
        if show_progress and iterator:
            iterator.set_postfix(file=file_name)
        
        if stored_docs:
            stored_hash = self._get_file_hash(stored_docs)
            if current_hash != stored_hash:
                # File changed - update
                self.delete_by_source_file(source_file)
                if show_progress and iterator:
                    iterator.set_postfix(file=file_name, status="updating...")
                self.add_documents(current_docs, batch_size=DEFAULT_BATCH_SIZE)
                stats["updated"] += len(current_docs)
                if show_progress and iterator:
                    iterator.set_postfix(file=file_name, status="updated")
            else:
                # File unchanged
                stats["unchanged"] += len(current_docs)
                if show_progress and iterator:
                    iterator.set_postfix(file=file_name, status="unchanged")
        else:
            # New file - add
            if show_progress and iterator:
                iterator.set_postfix(file=file_name, status="adding...")
            self.add_documents(current_docs, batch_size=DEFAULT_BATCH_SIZE)
            stats["added"] += len(current_docs)
            if show_progress and iterator:
                iterator.set_postfix(file=file_name, status="added")

    def _handle_failed_file(
        self, 
        source_file: str, 
        file_path: Path, 
        stats: Dict[str, int],
        show_progress: bool = True,
        iterator=None
    ):
        """Handle files that exist but failed to load."""
        if show_progress and iterator:
            iterator.set_postfix(file=Path(source_file).name, status="failed")
        
        print(f"\n⚠️  Warning: File exists but failed to load: {file_path}")
        
        stored_docs = self.get_documents_by_source_file(source_file)
        if stored_docs:
            deleted_count = self.delete_by_source_file(source_file)
            stats["deleted"] += deleted_count
            print(f"   Deleted {deleted_count} orphaned documents from failed file")

    def sync_with_datasets(self, document_loader, datasets_dir: Path, show_progress: bool = True) -> Dict[str, int]:
        """
        Sync the vector store with the datasets directory.
        - Deletes documents from files that no longer exist
        - Updates documents from files that have changed
        - Adds documents from new files
        
        Args:
            document_loader: DocumentLoader instance
            datasets_dir: Path to the datasets directory
            show_progress: Whether to show progress indicators (default: True)
            
        Returns:
            Dictionary with counts of deleted, updated, and added documents
        """
        from tqdm import tqdm
        
        self._ensure_initialized()
        
        self._log("\n📊 Analyzing datasets directory...", show_progress)
        
        current_files = {str(f.absolute()): f for f in document_loader.get_file_paths()}
        current_file_paths = set(current_files.keys())
        self._log(f"   Found {len(current_files)} PDF files", show_progress)
        
        self._log("🔍 Checking existing documents in vector store...", show_progress)
        stored_files = self.get_all_stored_source_files()
        self._log(f"   Found {len(stored_files)} source files in vector store", show_progress)
        
        stats = {"deleted": 0, "updated": 0, "added": 0, "unchanged": 0}
        
        self._log("\n📚 Loading and chunking documents...", show_progress)
        all_documents = document_loader.load_and_chunk_with_metadata(show_progress=show_progress)
        file_documents = self._group_documents_by_source(all_documents)
        
        self._log(f"\n🔄 Syncing vector store ({len(current_files)} files to process)...", show_progress)
        
        iterator = (tqdm(current_files.items(), desc="Processing files", unit="file", disable=not show_progress) 
                   if show_progress else current_files.items())
        
        for source_file, file_path in iterator:
            if source_file in file_documents:
                self._process_file_sync(source_file, file_path, file_documents[source_file], 
                                       stats, show_progress, iterator if show_progress else None)
            else:
                self._handle_failed_file(source_file, file_path, stats, show_progress, 
                                        iterator if show_progress else None)
        
        # Delete documents from files that no longer exist
        files_to_delete = stored_files - current_file_paths
        if files_to_delete:
            self._log(f"\n🗑️  Deleting documents from {len(files_to_delete)} removed files...", show_progress)
            delete_iterator = (tqdm(files_to_delete, desc="Deleting files", unit="file", disable=not show_progress) 
                             if show_progress else files_to_delete)
            for source_file in delete_iterator:
                deleted_count = self.delete_by_source_file(source_file)
                stats["deleted"] += deleted_count
        
        return stats

