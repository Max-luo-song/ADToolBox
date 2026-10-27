"""Natural-language driving-video retrieval capability."""

from .config import SearchConfig
from .service import (
    VideoSearchResult,
    index_local,
    retrieve_video,
    search_internet,
    search_local,
)

__all__ = [
    "SearchConfig",
    "VideoSearchResult",
    "index_local",
    "search_local",
    "search_internet",
    "retrieve_video",
]
