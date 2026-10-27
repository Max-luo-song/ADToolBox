"""JSON bridge to evidence-based token forensics."""
import json
import sys
from contextlib import redirect_stdout

try:
    from service import analyze_scene
    data = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        value = analyze_scene(data["token"], config_path=data.get("config_path"), run_name=data.get("run_name", "toolbox"),
                              resume=data.get("resume", True), timeout=data.get("timeout", 1800))
    analysis = {"failure_type":[value.reason_category],"failure_reason":value.failure_reason,
                "key_scene_factors":value.observations,"risk_factors":[],"forensics":value.raw,
                "uncertainty":value.uncertainty}
    print(json.dumps({"status":"success","result":{"failure_analysis":analysis,"report_path":value.report_path,
                     "result_path":value.result_path}},ensure_ascii=False))
except Exception as exc:
    print(json.dumps({"status":"error","error":{"message":str(exc)}}))
    raise SystemExit(1)
