# Semantic video search project / 视频语义检索项目

This is the migrated working video pipeline, including preprocessing, video chunking, multimodal embedding calls, persistent vector indexing, query recall, video reranking, clip export and explicit internet search. The `vendor/sentrysearch` runtime is retained with its existing internal naming; the public component is named by function.

```bash
python -m venv environments/video_search
environments/video_search/bin/python -m pip install -e projects/video_search
export DASHSCOPE_API_KEY=your-key
export TOOLBOX_VIDEO_SEARCH_HOME=/absolute/path/to/video-library
environments/video_search/bin/video-search index /path/to/videos
environments/video_search/bin/video-search search-local "a car cuts in on a highway"
```

Set `video_search.python` in the toolkit runtime configuration. `video_retriever` then uses this project by default. Use `action=index` with `video_path` for indexing, `action=search` with `requirement` for local semantic search. Internet search is only invoked with `action=search_internet`.

Reranking requires explicitly configured `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_BUCKET` and `OSS_ENDPOINT`. Temporary candidate videos are uploaded for model access and cleanup is attempted afterwards. No private storage bucket is built in. Indexes and videos remain outside Git. `catalog_path` selects the optional lightweight caption-search mode; it is not the default semantic engine.

中文：完整实现位于 `video_search/service.py`、`pipeline/`、`reranker.py` 和 `vendor/`。默认工具调用真实语义检索；字幕检索只作为显式选项。
