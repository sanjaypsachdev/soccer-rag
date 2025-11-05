"""DocumentLoader class for loading and chunking PDF documents."""
import hashlib
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Recursively loads and chunks PDF documents with support for text, tables, and images."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunk_overlap_strategy: str = "overlap",
    ):
        """
        Initialize the DocumentLoader.

        Args:
            file_path: File path or directory containing PDF files
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            chunk_overlap_strategy: Strategy for handling chunk overlap
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_overlap_strategy = chunk_overlap_strategy

        # Initialize text splitter
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def _get_file_hash(self, pdf_path: Path) -> str:
        """
        Get a hash of the file content for change detection.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            SHA256 hash of the file content
        """
        try:
            with open(pdf_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _extract_table_as_text(self, table) -> str:
        """
        Extract table data and convert to readable text format.
        
        Args:
            table: PyMuPDF table object
            
        Returns:
            Text representation of the table
        """
        try:
            # Try to extract table using the extract() method
            rows = []
            table_data = table.extract()
            for row in table_data:
                # Convert each row to a readable format
                row_text = " | ".join([str(cell).strip() if cell else "" for cell in row])
                if row_text.strip():  # Only add non-empty rows
                    rows.append(row_text)
            return "\n".join(rows) if rows else ""
        except AttributeError:
            # If extract() doesn't work, try to get table as list of lists
            try:
                rows = []
                for row in table:
                    if isinstance(row, (list, tuple)):
                        row_text = " | ".join([str(cell).strip() if cell else "" for cell in row])
                        if row_text.strip():
                            rows.append(row_text)
                return "\n".join(rows) if rows else ""
            except Exception:
                return ""
        except Exception:
            return ""
    
    def _load_pdf_as_documents(self, pdf_path: Path) -> List[Document]:
        """
        Load a PDF file and extract text, tables, and image metadata.
        Handles text, tables, and infographics appropriately.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects with extracted content
        """
        try:
            doc = fitz.open(str(pdf_path))
            documents = []
            
            for page_num, page in enumerate(doc):
                page_text_parts = []
                
                # Extract regular text
                text = page.get_text("text")
                if text.strip():
                    page_text_parts.append(f"=== Page {page_num + 1} Text ===\n{text}")
                
                # Extract tables
                try:
                    # Try to find tables (available in PyMuPDF 1.23.0+)
                    if hasattr(page, 'find_tables'):
                        tables = page.find_tables()
                        if tables:
                            for table_idx, table in enumerate(tables):
                                table_text = self._extract_table_as_text(table)
                                if table_text.strip():
                                    page_text_parts.append(
                                        f"\n=== Page {page_num + 1} Table {table_idx + 1} ===\n{table_text}"
                                    )
                    else:
                        # Fallback: If find_tables() is not available in this PyMuPDF version,
                        # tables will be included in the regular text extraction above
                        pass
                except Exception:
                    # Table extraction might fail, continue with text
                    pass
                
                # Check for images/infographics
                image_list = page.get_images()
                if image_list:
                    image_count = len(image_list)
                    page_text_parts.append(
                        f"\n=== Page {page_num + 1} Note ===\n"
                        f"This page contains {image_count} image(s) or infographic(s). "
                        f"Visual content may not be fully captured in text form."
                    )
                
                # Combine all content for this page
                if page_text_parts:
                    combined_content = "\n\n".join(page_text_parts)
                    documents.append(Document(
                        page_content=combined_content,
                        metadata={"page": page_num + 1}
                    ))
            
            doc.close()
            return documents
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
            return []

    def _find_pdf_files(self) -> List[Path]:
        """
        Recursively find all PDF files in the specified directory.

        Returns:
            List of paths to PDF files
        """
        pdf_files = []
        if self.file_path.is_file() and self.file_path.suffix.lower() == ".pdf":
            pdf_files.append(self.file_path)
        elif self.file_path.is_dir():
            pdf_files.extend(self.file_path.rglob("*.pdf"))
        return pdf_files

    def load_and_chunk(self) -> List[str]:
        """
        Load all PDF files and return chunked text.
        
        For backward compatibility, returns only text chunks.

        Returns:
            List of text chunks
        """
        documents = self.load_and_chunk_with_metadata()
        return [doc.page_content for doc in documents]
    
    def load_and_chunk_with_metadata(self, show_progress: bool = True) -> List[Document]:
        """
        Load all PDF files and return chunked documents with metadata.

        Args:
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of Document objects with metadata (source_file, file_hash, modified_time)
        """
        from tqdm import tqdm
        
        pdf_files = self._find_pdf_files()
        all_documents = []

        iterator = tqdm(pdf_files, desc="Loading files", unit="file", disable=not show_progress) if show_progress else pdf_files
        
        for pdf_file in iterator:
            if show_progress:
                iterator.set_postfix(file=pdf_file.name)
            
            # Load PDF documents (text, tables, and images)
            pdf_documents = self._load_pdf_as_documents(pdf_file)
            if pdf_documents:
                # Get file metadata
                file_hash = self._get_file_hash(pdf_file)
                file_path_str = str(pdf_file.absolute())
                modified_time = pdf_file.stat().st_mtime
                
                # Process each page's content
                for pdf_doc in pdf_documents:
                    # Chunk the page content (which may include text, tables, and image notes)
                    chunks = self.chunker.create_documents([pdf_doc.page_content])
                    
                    # Add metadata to each chunk
                    for chunk in chunks:
                        chunk.metadata = {
                            "source_file": file_path_str,
                            "file_name": pdf_file.name,
                            "file_hash": file_hash,
                            "modified_time": modified_time,
                        }
                        # Preserve page number from PDF document metadata if available
                        if "page" in pdf_doc.metadata:
                            chunk.metadata["page_number"] = pdf_doc.metadata["page"]
                        all_documents.append(chunk)
        
        if show_progress:
            print(f"✓ Loaded {len(all_documents)} document chunks from {len(pdf_files)} files")

        return all_documents

    def load(self) -> List[str]:
        """
        Alias for load_and_chunk for backward compatibility.

        Returns:
            List of text chunks
        """
        return self.load_and_chunk()
    
    def get_file_paths(self) -> List[Path]:
        """
        Get list of all PDF file paths in the datasets directory.
        
        Returns:
            List of Path objects for PDF files
        """
        return self._find_pdf_files()

