"""Initialize the included compute source and install its version-pinned extensions."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=["reconstruction", "simulation"])
    args = parser.parse_args()
    relative = "projects/" + args.project
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--", relative], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/"extensions"/args.project/"apply_extensions.py"),
                    "--backend", str(ROOT/relative), "--apply"], check=True)
    print("Source ready. Install that project's documented GPU dependencies and configure weights/data before running.")

if __name__ == "__main__":
    main()
