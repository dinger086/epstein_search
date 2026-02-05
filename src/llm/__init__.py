"""LLM module for Ollama integration and RAG."""

from .ollama_client import chat, list_models, check_model_available
from .rag import ask, ask_with_sources

__all__ = [
    "chat",
    "list_models",
    "check_model_available",
    "ask",
    "ask_with_sources",
]
