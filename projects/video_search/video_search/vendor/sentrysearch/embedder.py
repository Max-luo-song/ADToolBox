"""Small embedder facade for the vendored Qwen API backend."""

from .qwen_api_embedder import QwenAPIEmbedder

_current_embedder = None


def get_embedder(backend: str = "qwen_api", **kwargs):
    global _current_embedder
    if backend != "qwen_api":
        raise ValueError(f"Unsupported embedding backend: {backend}")
    if _current_embedder is None:
        _current_embedder = QwenAPIEmbedder(dimensions=kwargs.get("dimensions", 768))
    return _current_embedder


def reset_embedder() -> None:
    global _current_embedder
    _current_embedder = None
