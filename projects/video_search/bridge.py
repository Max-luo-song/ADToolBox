"""JSON process bridge to the actual indexing/search pipeline."""
import json
import sys
from contextlib import redirect_stdout

def main():
    try:
        from video_search.config import SearchConfig
        from video_search.service import index_local, search_local, search_internet
        data = json.load(sys.stdin)
        config = SearchConfig.for_library(data.get("library", "default"))
        with redirect_stdout(sys.stderr):
            action = data.get("action", "search")
            if action == "index":
                result = index_local(data["video_path"], config)
            else:
                result = (search_internet if action == "search_internet" else search_local)(data["requirement"], config).to_dict()
        print(json.dumps({"status":"success","result":result},ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status":"error","error":{"message":str(exc)}}))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
