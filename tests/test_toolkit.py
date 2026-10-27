import asyncio
import importlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from driving_toolbox import list_tools, run_tool
from driving_toolbox.backends.coordinates import local_to_scene_world
from driving_toolbox.tools.edit_config_generator.compiler import to_rigid
from driving_toolbox.tools.failure_case_filter.evaluator import overlaps

ROOT = Path(__file__).resolve().parents[1]


class ToolkitTests(unittest.TestCase):
    def test_all_fourteen_have_independent_entries_and_valid_schemas(self):
        import jsonschema

        tools = list_tools()
        self.assertEqual(len(tools), 14)
        self.assertEqual(len({tool["name"] for tool in tools}), 14)
        for tool in tools:
            jsonschema.Draft202012Validator.check_schema(tool["input_schema"])
            self.assertTrue(callable(importlib.import_module(f"driving_toolbox.tools.{tool['name']}").run))
            result = run_tool(tool["name"], [])
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error"]["code"], "invalid_input")

    def test_examples_match_input_contracts(self):
        import jsonschema

        for tool in list_tools():
            data = json.loads((ROOT / "examples" / tool["name"] / "input.json").read_text())
            jsonschema.validate(data, tool["input_schema"])

    def test_import_and_listing_do_not_load_gpu_packages(self):
        script = "import sys; from driving_toolbox import list_tools; assert len(list_tools()) == 14; assert not {'torch','cv2','ultralytics'} & set(sys.modules)"
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_compile_real_edit_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edit.json"
            result = run_tool(
                "edit_config_generator",
                {
                    "operations": [{"operation": "remove", "instance_id": 3}],
                    "instances": [{"instance_id": 3}],
                    "output_path": str(path),
                },
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(json.loads(path.read_text())["operations"][0]["instance_id"], 3)
            self.assertEqual(result["artifacts"][0]["path"], str(path.resolve()))

    def test_compiler_rejects_missing_identity_unknown_identity_and_nonfinite_scale(self):
        for data in [
            {"operations": [{"operation": "remove"}]},
            {"operations": [{"operation": "remove", "instance_id": 7}], "instances": [{"instance_id": 0}]},
            {"operations": [{"operation": "scale", "instance_id": 0, "scale": float("nan")}]},
            {"operations": [{"operation": "trajectory_edit", "instance_id": 0, "waypoints": [[1, 2]]}]},
        ]:
            self.assertEqual(run_tool("edit_config_generator", data)["status"], "error")

    def test_all_six_operations_compile_to_renderer_wire_format(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset.ply"
            asset.write_text("ply\n")
            operations = [
                {
                    "operation": "trajectory_edit",
                    "instance_id": 0,
                    "waypoints": [[0, 0, 0]],
                    "method": "catmull_rom",
                },
                {"operation": "rotate", "instance_id": 0, "rotation": [0, 90, 0]},
                {"operation": "scale", "instance_id": 0, "scale": 1.1},
                {"operation": "remove", "instance_id": 1},
                {"operation": "add", "asset_path": str(asset), "scale": 0.5, "anchor_pixel": [20, 30]},
                {"operation": "replace", "instance_id": 2, "asset_path": str(asset), "scale": 0.8},
            ]
            result = run_tool("edit_config_generator", {"operations": operations})
            self.assertEqual(result["status"], "success")
            rigid = to_rigid(result["result"]["edit_config"])["Nodes"]["RigidNodes"]
            self.assertEqual(set(rigid), {item["operation"] for item in operations})
            self.assertEqual(rigid["trajectory_edit"]["method"], ["cubic"])

    def test_metric_camera_offsets_convert_per_frame_and_lock_vertical_axis(self):
        pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        for actual, expected in zip(
            local_to_scene_world([2, 3, 0], pose, 0.1), [-0.3, 0.0, 0.2], strict=True
        ):
            self.assertAlmostEqual(actual, expected)

    def test_oriented_boxes_avoid_axis_aligned_false_positive(self):
        ego = {"position": [0, 0], "length": 6, "width": 1, "heading": math.pi / 4}
        actor = {"position": [0, 3], "length": 6, "width": 1, "heading": math.pi / 4}
        self.assertFalse(overlaps(ego, actor))
        self.assertTrue(overlaps(ego, {**ego, "position": [0.1, 0.1]}))

    def test_curation_writes_only_failures_and_requires_explicit_units(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "cases.jsonl"
            result = run_tool(
                "failure_case_filter",
                {
                    "scenario_path": str(ROOT / "examples/failure_case_filter/scenario.json"),
                    "output_library": str(library),
                },
            )
            self.assertTrue(result["result"]["is_failure_case"])
            self.assertEqual(len(library.read_text().splitlines()), 1)
            source = Path(directory) / "bad.json"
            source.write_text(json.dumps({"frames": []}))
            self.assertEqual(
                run_tool("failure_case_filter", {"scenario_path": str(source)})["status"], "error"
            )

    def test_caption_search_ranks_and_exports_actual_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "day.mp4").write_bytes(b"fixture-only-not-a-real-video")
            (root / "night.mp4").write_bytes(b"fixture-only-not-a-real-video")
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    [
                        {"video_path": "night.mp4", "caption": "urban night pedestrians"},
                        {"video_path": "day.mp4", "caption": "highway daylight cut in"},
                    ]
                )
            )
            output = root / "selected.mp4"
            result = run_tool(
                "video_retriever",
                {"requirement": "highway daylight", "catalog_path": str(catalog), "export_path": str(output)},
            )
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["result"]["candidates"][0]["video_path"].endswith("day.mp4"))
            self.assertEqual(output.read_bytes(), (root / "day.mp4").read_bytes())

    def test_file_only_validation_never_claims_requirement_or_physics_pass(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            video.write(b"test fixture")
            video.flush()
            result = run_tool(
                "scene_validator", {"rendered_video": video.name, "scene_requirement": {"target": "cut in"}}
            )
            self.assertEqual(result["status"], "success")
            self.assertFalse(result["result"]["passed"])
            self.assertFalse(result["result"]["physics_verified"])

    def test_diagnostics_are_advisory_and_fail_closed(self):
        self.assertEqual(run_tool("recovery_advisor", {"evaluation": {}})["result"]["action"], "inspect")
        result = run_tool("recovery_advisor", {"evaluation": {"passed": True}})
        self.assertEqual(result["result"]["action"], "continue")
        self.assertFalse(result["result"]["executes_actions"])

    def test_missing_gpu_backend_is_explicit(self):
        result = run_tool("reconstruction", {"video_path": "/path/to/video.mp4"})
        self.assertEqual(result["error"]["code"], "missing_dependency")

    def test_model_transport_validates_json_and_hides_credentials(self):
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "scene_requirement": "cut in",
                                "failure_type": "cut_in",
                                "required_conditions": [],
                                "required_actors": ["ego", "car"],
                                "risk_factors": [],
                                "target_effect": "test braking",
                            }
                        )
                    }
                }
            ]
        }
        with patch.dict(os.environ, {"TOOLBOX_API_KEY": "test-secret"}, clear=True):
            with patch(
                "driving_toolbox.core.llm.urlopen", return_value=io.BytesIO(json.dumps(body).encode())
            ) as request:
                result = run_tool(
                    "scene_requirement_generator",
                    {"failure_analysis": {"reason": "cut in"}},
                    config={"llm": {"base_url": "https://example.invalid/v1", "model": "example"}},
                )
                self.assertEqual(result["status"], "success")
                sent = request.call_args.args[0]
                self.assertEqual(sent.get_header("Authorization"), "Bearer test-secret")
                self.assertNotIn("test-secret", json.dumps(result))
            with patch(
                "driving_toolbox.core.llm.urlopen",
                return_value=io.BytesIO(b'{"choices":[{"message":{"content":"{}"}}]}'),
            ):
                result = run_tool(
                    "scene_requirement_generator",
                    {"failure_analysis": {}},
                    config={"llm": {"base_url": "https://example.invalid/v1", "model": "example"}},
                )
                self.assertEqual(result["status"], "error")

    def test_external_json_process_success_failure_and_timeout(self):
        data = {"video_path": "/path/to/video.mp4"}
        command = [
            sys.executable,
            "-c",
            "import json,sys; value=json.load(sys.stdin); print(json.dumps({'status':'success','result':{'received':value}}))",
        ]
        config = {"backends": {"reconstruction": {"command": command}}}
        result = run_tool("reconstruction", data, config=config)
        self.assertEqual(result["result"]["received"], data)
        for command, timeout in [
            ([sys.executable, "-c", "print('not-json')"], 5),
            ([sys.executable, "-c", "import time; time.sleep(5)"], 0.02),
        ]:
            result = run_tool(
                "reconstruction",
                data,
                config={"backends": {"reconstruction": {"command": command, "timeout": timeout}}},
            )
            self.assertEqual(result["error"]["code"], "backend_failed")

    def test_cli_errors_and_real_example(self):
        result = subprocess.run(
            [sys.executable, "-m", "driving_toolbox", "run", "recovery_advisor", "--input", "-"],
            input='{"evaluation":{"passed":true}}',
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["status"], "success")
        result = subprocess.run(
            [sys.executable, "-m", "driving_toolbox", "run", "recovery_advisor", "--input", "-"],
            input="{}",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)

    def test_mcp_stdio_discovery_call_and_error(self):
        async def check():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "driving_toolbox.mcp_server"],
                env={"PYTHONPATH": str(ROOT / "src")},
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertEqual(len(tools.tools), 14)
                    result = await session.call_tool("recovery_advisor", {"evaluation": {"passed": True}})
                    self.assertFalse(result.isError)
                    failed = await session.call_tool("reconstruction", {"video_path": "/path/to/video.mp4"})
                    self.assertTrue(failed.isError)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
