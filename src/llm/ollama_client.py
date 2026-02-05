"""Ollama API client wrapper."""

from typing import Generator

from ..config import LLM_MODEL

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def _check_ollama():
    """Verify Ollama is available."""
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama is required. Install with: pip install ollama")


def chat(
    prompt: str,
    model: str = None,
    system: str = None,
    context: str = None,
    stream: bool = False
) -> str | Generator[str, None, None]:
    """
    Send a chat message to Ollama.

    Args:
        prompt: User prompt
        model: Model to use
        system: System prompt
        context: Additional context to include
        stream: Whether to stream the response

    Returns:
        Response text or generator if streaming
    """
    _check_ollama()
    model = model or LLM_MODEL

    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    user_content = prompt
    if context:
        user_content = f"Context:\n{context}\n\n{prompt}"

    messages.append({"role": "user", "content": user_content})

    if stream:
        def generate():
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
        return generate()
    else:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]


def chat_with_history(
    messages: list[dict],
    model: str = None,
    stream: bool = False
) -> str | Generator[str, None, None]:
    """
    Continue a conversation with message history.

    Args:
        messages: List of message dicts with role and content
        model: Model to use
        stream: Whether to stream response

    Returns:
        Response text or generator if streaming
    """
    _check_ollama()
    model = model or LLM_MODEL

    if stream:
        def generate():
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
        return generate()
    else:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]


def list_models() -> list[dict]:
    """
    List available Ollama models.

    Returns:
        List of model info dicts
    """
    _check_ollama()

    response = ollama.list()
    return response.get("models", [])


def check_model_available(model: str) -> bool:
    """
    Check if a model is available locally.

    Args:
        model: Model name to check

    Returns:
        True if model is available
    """
    _check_ollama()

    models = list_models()
    model_names = [m.get("name", "").split(":")[0] for m in models]
    model_names.extend([m.get("name", "") for m in models])

    return model in model_names or model.split(":")[0] in model_names


def pull_model(model: str, progress_callback=None) -> bool:
    """
    Pull a model from Ollama.

    Args:
        model: Model name to pull
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful
    """
    _check_ollama()

    try:
        for progress in ollama.pull(model, stream=True):
            if progress_callback:
                status = progress.get("status", "")
                completed = progress.get("completed", 0)
                total = progress.get("total", 0)
                progress_callback(status, completed, total)
        return True
    except Exception:
        return False
