"""Convert edited Gaussian target frames into a simulation scene using the bundled engine."""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "projects" / "simulation" / "scripts"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, default=0)
    parser.add_argument("--scene-id", default="adtoolbox-scene-000")
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--translation-scale", type=float, required=True)
    parser.add_argument("--python", required=True, help="Python with the scene builder's numpy/torch/sklearn dependencies")
    args = parser.parse_args()
    if args.translation_scale <= 0:
        parser.error("translation-scale must be positive and externally calibrated")
    stage = SCRIPTS / "stage_scene.py"
    builder = SCRIPTS / "build_dynamic_scene.py"
    for path in (stage, builder):
        if not path.is_file():
            parser.error(f"Bundled simulation engine is incomplete; missing {path}")
    spec = importlib.util.spec_from_file_location("stage_scene", stage)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(args.source, args.staging, args.instance_id, args.source_fps)
    subprocess.run([args.python,str(builder),"--source",str(args.staging.resolve()),"--output-root",str(args.output.resolve()),
                    "--scene-id",args.scene_id,"--translation-scale",str(args.translation_scale)],check=True,cwd=builder.parent)

if __name__ == "__main__":
    main()
