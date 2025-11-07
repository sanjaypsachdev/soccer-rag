"""Document synchronization manager for vectorstore."""
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from tqdm import tqdm

from app.vectorstore.utils import DEFAULT_BATCH_SIZE, log_message


class SyncManager:
    """Manages synchronization of documents with datasets directory."""

    def __init__(self, vectorstore):
        """
        Initialize SyncManager.

        Args:
            vectorstore: Vectorstore instance to sync
        """
        self.vectorstore = vectorstore

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
        stored_docs = self.vectorstore.get_documents_by_source_file(source_file)

        if show_progress and iterator:
            iterator.set_postfix(file=file_name)

        if stored_docs:
            stored_hash = self._get_file_hash(stored_docs)
            if current_hash != stored_hash:
                # File changed - update
                self.vectorstore.delete_by_source_file(source_file)
                if show_progress and iterator:
                    iterator.set_postfix(file=file_name, status="updating...")
                self.vectorstore.add_documents(current_docs, batch_size=DEFAULT_BATCH_SIZE)
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
            self.vectorstore.add_documents(current_docs, batch_size=DEFAULT_BATCH_SIZE)
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

        stored_docs = self.vectorstore.get_documents_by_source_file(source_file)
        if stored_docs:
            deleted_count = self.vectorstore.delete_by_source_file(source_file)
            stats["deleted"] += deleted_count
            print(f"   Deleted {deleted_count} orphaned documents from failed file")

    def sync_with_datasets(self, document_loader, datasets_dir: Path, show_progress: bool = True) -> Dict[str, int]:
        """
        Sync the vector store with the datasets directory.

        Args:
            document_loader: DocumentLoader instance
            datasets_dir: Path to the datasets directory
            show_progress: Whether to show progress indicators

        Returns:
            Dictionary with counts of deleted, updated, and added documents
        """
        log_message("\n📊 Analyzing datasets directory...", show_progress)

        current_files = {str(f.absolute()): f for f in document_loader.get_file_paths()}
        current_file_paths = set(current_files.keys())
        log_message(f"   Found {len(current_files)} PDF files", show_progress)

        log_message("🔍 Checking existing documents in vector store...", show_progress)
        stored_files = self.vectorstore.get_all_stored_source_files()
        log_message(f"   Found {len(stored_files)} source files in vector store", show_progress)

        stats = {"deleted": 0, "updated": 0, "added": 0, "unchanged": 0}

        log_message("\n📚 Loading and chunking documents...", show_progress)
        all_documents = document_loader.load_and_chunk_with_metadata(show_progress=show_progress)
        file_documents = self._group_documents_by_source(all_documents)

        log_message(f"\n🔄 Syncing vector store ({len(current_files)} files to process)...", show_progress)

        iterator = (tqdm(current_files.items(), desc="Processing files", unit="file", disable=not show_progress)
                   if show_progress else current_files.items())

        for source_file, file_path in iterator:
            if source_file in file_documents:
                self._process_file_sync(
                    source_file, file_path, file_documents[source_file],
                    stats, show_progress, iterator if show_progress else None
                )
            else:
                self._handle_failed_file(
                    source_file, file_path, stats, show_progress,
                    iterator if show_progress else None
                )

        # Delete documents from files that no longer exist
        files_to_delete = stored_files - current_file_paths
        if files_to_delete:
            log_message(f"\n🗑️  Deleting documents from {len(files_to_delete)} removed files...", show_progress)
            delete_iterator = (tqdm(files_to_delete, desc="Deleting files", unit="file", disable=not show_progress)
                             if show_progress else files_to_delete)
            for source_file in delete_iterator:
                deleted_count = self.vectorstore.delete_by_source_file(source_file)
                stats["deleted"] += deleted_count

        return stats

