"""Exercise launcher contracts with local fixture executables, without pretending to run models."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from adtoolbox import run_tool


class BackendContractTests(unittest.TestCase):
    def test_separate_simulation_and_policy_processes_produce_fresh_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim = root / "engine/simengine/runner/run_simulation.py"
            sim.parent.mkdir(parents=True)
            sim.write_text(
                "import json,sys\nfrom pathlib import Path\n"
                "args=dict(arg.split('=',1) for arg in sys.argv[1:])\n"
                "path=Path(args['output_dir'])/'runner_report_fixture.json'\n"
                "path.write_text(json.dumps([{'succeeded':True,'parameters':args}]))\n"
            )
            policy = root / "policy/closed_loop/sim_test.py"
            policy.parent.mkdir(parents=True)
            policy.write_text("print('fixture policy client')\n")
            config_path, checkpoint = root / "policy.py", root / "policy.pth"
            config_path.touch()
            checkpoint.touch()
            scenes, assets = root / "scenes", root / "assets"
            scenes.mkdir()
            assets.mkdir()
            result = run_tool(
                "policy_runner",
                {
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint),
                    "model_name": "fixture",
                    "data_type": "fixture",
                    "with_metric_manager": False,
                    "scenario_data_folder_path": str(scenes),
                    "asset_folder_path": str(assets),
                },
                config={
                    "simulation": {
                        "root": str(root),
                        "simulation_python": sys.executable,
                        "policy_python": sys.executable,
                        "timeout": 10,
                    }
                },
            )
            self.assertEqual(result["status"], "success", result)
            self.assertEqual(result["result"]["reports"][0]["parameters"]["with_metric_manager"], "false")
            self.assertTrue(Path(result["artifacts"][0]["path"]).is_file())

    def test_custom_launcher_cannot_reuse_stale_runner_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, checkpoint, script = root / "policy.py", root / "policy.pth", root / "noop.sh"
            config_path.touch()
            checkpoint.touch()
            script.write_text("#!/bin/sh\nexit 0\n")
            report = root / "experiments/closed_loop_exps/fixture/fixture_NR/runner_report_old.json"
            report.parent.mkdir(parents=True)
            report.write_text('[{"succeeded":true}]')
            result = run_tool(
                "policy_runner",
                {
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint),
                    "model_name": "fixture",
                    "data_type": "fixture",
                },
                config={"simulation": {"root": str(root), "script": str(script)}},
            )
            self.assertEqual(result["status"], "error")
            self.assertIn("fresh", result["error"]["message"])

    def test_training_requires_new_checkpoint_and_empty_work_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "policy/scripts/train.py"
            script.parent.mkdir(parents=True)
            script.write_text(
                "import sys\nfrom pathlib import Path\n"
                "work=Path(sys.argv[sys.argv.index('--work-dir')+1])\n"
                "(work/'epoch_1.pth').write_bytes(b'fixture-not-real-weights')\n"
            )
            data = {"work_dir": str(root / "training"), "epochs": 1}
            for key in ("config_path", "base_checkpoint", "ann_file", "token_file"):
                path = root / key
                path.touch()
                data[key] = str(path)
            config = {"training": {"root": str(root), "python": sys.executable, "timeout": 10}}
            result = run_tool("policy_trainer", data, config=config)
            self.assertEqual(result["status"], "success", json.dumps(result))
            self.assertTrue(Path(result["result"]["checkpoint"]).is_file())
            self.assertEqual(run_tool("policy_trainer", data, config=config)["status"], "error")


if __name__ == "__main__":
    unittest.main()
