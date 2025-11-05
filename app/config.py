"""Configuration management for the RAG pipeline."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for managing environment variables and settings."""
    
    # Default values
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 100
    DEFAULT_K = 4
    DEFAULT_EMBEDDINGS_MODEL = "text-embedding-3-small"
    DEFAULT_LLM_MODEL = "gpt-5-nano"
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_TEMPERATURE = 0.0
    
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
        return Config.DEFAULT_CHUNK_SIZE
    
    @staticmethod
    def get_chunk_overlap() -> int:
        """Get the chunk overlap for document splitting."""
        return Config.DEFAULT_CHUNK_OVERLAP
    
    @staticmethod
    def get_default_k() -> int:
        """Get the default number of results to retrieve."""
        return Config.DEFAULT_K
    
    @staticmethod
    def get_embeddings_model_name() -> str:
        """Get the embeddings model name."""
        return Config.DEFAULT_EMBEDDINGS_MODEL
    
    @staticmethod
    def get_embeddings_timeout() -> float:
        """Get the timeout for embeddings API calls."""
        return Config.DEFAULT_TIMEOUT
    
    @staticmethod
    def get_embeddings_max_retries() -> int:
        """Get the maximum number of retries for embeddings API calls."""
        return Config.DEFAULT_MAX_RETRIES
    
    @staticmethod
    def get_llm_model_name() -> str:
        """Get the LLM model name for answer generation."""
        return Config.DEFAULT_LLM_MODEL
    
    @staticmethod
    def get_llm_temperature() -> float:
        """Get the temperature for LLM generation."""
        return Config.DEFAULT_TEMPERATURE

