"""Configuration management for the RAG pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for managing environment variables and settings."""
    
    # Default values
    CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1000) 
    CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 100)
    TOP_K = os.getenv("TOP_K", 4)
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    TEMPERATURE = os.getenv("TEMPERATURE", 0.0)
    
    @staticmethod
    def get_pinecone_api_key() -> str:
        """Get Pinecone API key from environment variables."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment."
            )
        return api_key
    
    @staticmethod
    def get_pinecone_index_name() -> str:
        """Get Pinecone index name from environment variables."""
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME environment variable is not set. "
                "Please set it in your .env file or environment."
            )
        return index_name
    
    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenAI API key from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment."
            )
        return api_key
    
    @staticmethod
    def get_datasets_dir() -> Path:
        """Get the datasets directory path."""
        return Path("datasets")
    
    @staticmethod
    def get_chunk_size() -> int:
        """Get the chunk size for document splitting."""
        return Config.CHUNK_SIZE
    
    @staticmethod
    def get_chunk_overlap() -> int:
        """Get the chunk overlap for document splitting."""
        return Config.CHUNK_OVERLAP
    
    @staticmethod
    def get_default_k() -> int:
        """Get the default number of results to retrieve."""
        return Config.TOP_K
    
    @staticmethod
    def get_embeddings_model_name() -> str:
        """Get the embeddings model name."""
        return Config.EMBEDDINGS_MODEL
    
    @staticmethod
    def get_llm_model_name() -> str:
        """Get the LLM model name for answer generation."""
        return Config.LLM_MODEL
    
    @staticmethod
    def get_llm_temperature() -> float:
        """Get the temperature for LLM generation."""
        return Config.TEMPERATURE

