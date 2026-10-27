"""Tests for the staged scene-conversion helpers bundled with the simulation engine."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_SCENE = ROOT / "projects" / "simulation" / "scripts" / "stage_scene.py"


def _load_stage_scene():
    spec = importlib.util.spec_from_file_location("stage_scene", STAGE_SCENE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageSceneTests(unittest.TestCase):
    def test_rendered_ego_pose_is_normalized_for_simulation(self):
        stage_scene = _load_stage_scene()
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ply = {
                "coordinate_convention": "OpenCV world-to-camera for normalized PLY coordinates",
                "ply_world_transform": {"center": [10, 20, 30], "scale": 10},
                "frames": [{"frame_index": 0, "intrinsics": [[1]], "world_to_camera_opencv": identity}],
            }
            rendered = {
                "coordinate_convention": "OpenCV world-to-camera (x right, y down, z forward)",
                "frames": [{"frame_index": 0, "intrinsics": [[2]], "world_to_camera_opencv": identity}],
            }
            (scene / "ply_camera_params.json").write_text(json.dumps(ply), encoding="utf-8")
            (scene / "camera_params.json").write_text(json.dumps(rendered), encoding="utf-8")
            result, source = stage_scene._normalized_camera(scene, 1)
            self.assertEqual(source, "camera_params.json")
            self.assertEqual(result["frames"][0]["world_to_camera_opencv"][0][3], 1.0)
            self.assertTrue(result["frames"][0]["world_to_camera_opencv"][1][3] == 2.0)
            self.assertTrue(result["frames"][0]["world_to_camera_opencv"][2][3] == 3.0)


if __name__ == "__main__":
    unittest.main()
