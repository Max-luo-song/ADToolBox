"""Reference caption-retrieval bridge: one input JSON, one output JSON."""
import json
import sys
from driving_toolbox.tools.video_retriever.caption_search import search

try:
    result = search(json.load(sys.stdin))
    print(json.dumps({"status":"success","result":result}))
except Exception as exc:
    print(json.dumps({"status":"error","error":{"message":str(exc)}}))
    raise SystemExit(1)
