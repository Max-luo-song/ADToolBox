import json
import math
from pathlib import Path

from ..core.errors import InputError
from ..core.io import file_path, positive, read_json


def _box(state):
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("position"), list)
        or len(state["position"]) < 2
    ):
        raise InputError("Every box requires position [x,y], length and width")
    x, y = state["position"][:2]
    heading = state.get("heading", 0.0)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y, heading)):
        raise InputError("Positions and heading must be finite numbers")
    length = positive(state.get("length"), "length")
    width = positive(state.get("width"), "width")
    c, s = math.cos(heading), math.sin(heading)
    return [
        (x + c * a - s * b, y + s * a + c * b)
        for a, b in [
            (-length / 2, -width / 2),
            (length / 2, -width / 2),
            (length / 2, width / 2),
            (-length / 2, width / 2),
        ]
    ]


def overlaps(a, b):
    a, b = _box(a), _box(b)
    for polygon in (a, b):
        for i in range(2):
            p, q = polygon[i], polygon[(i + 1) % 4]
            axis = (-(q[1] - p[1]), q[0] - p[0])
            pa = [x * axis[0] + y * axis[1] for x, y in a]
            pb = [x * axis[0] + y * axis[1] for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def filter_case(data):
    scenario_path = file_path(data["scenario_path"], "scenario_path")
    scenario = read_json(scenario_path)
    if scenario.get("coordinate_system") != "world_xy_meters":
        raise InputError("Scenario coordinate_system must be world_xy_meters; headings are radians")
    frames = scenario.get("frames")
    if not isinstance(frames, list) or not frames:
        raise InputError("Scenario needs a non-empty frames array")
    events = []
    indices = set()
    for i, frame in enumerate(frames):
        index = frame.get("frame_index", i)
        if type(index) is not int or index < 0 or index in indices:
            raise InputError("frame_index must be unique non-negative integers")
        indices.add(index)
        ego = frame.get("ego")
        _box(ego)
        for actor in frame.get("actors", []):
            if actor.get("valid", True) is False:
                continue
            if overlaps(ego, actor):
                events.append({"frame_index": index, "object_id": actor.get("id", "unknown")})
    case = {
        "case_id": data.get("case_id", scenario_path.stem),
        "scenario_path": str(scenario_path),
        "reasons": ["dynamic_box_overlap"] if events else [],
        "collision_events": events,
    }
    result = {
        "is_failure_case": bool(events),
        "failure_cases": [case] if events else [],
        "collision_events": events,
        "frames_evaluated": len(frames),
        "evaluation_scope": "dynamic_2d_oriented_boxes",
        "_warnings": [
            "Box overlap is a screening signal, not collision responsibility or road-compliance scoring."
        ],
    }
    if events and data.get("output_library"):
        path = Path(data["output_library"]).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
        result["_artifacts"] = [{"kind": "case_library", "path": str(path)}]
    return result
