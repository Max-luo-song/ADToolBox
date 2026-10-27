"""Apply the version-pinned simulation patch to a separate installed backend."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply after checks; default is read-only")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "manifest.json").read_text())
    root = args.backend.expanduser().resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != manifest["revision"]:
        raise SystemExit("Backend revision does not match the patch manifest")
    if all((root / entry["path"]).is_file() and hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() == entry["patched_sha256"] for entry in manifest["files"]):
        print("Extensions already installed")
        return
    for entry in manifest["files"]:
        path = root / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if digest != entry["base_sha256"]:
            raise SystemExit("Target differs from the expected clean base: " + entry["path"])
    patch = here / "scene_extensions.patch"
    subprocess.run(["git", "-C", str(root), "apply", "--check", str(patch)], check=True)
    if args.apply:
        subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
    print("Patch applied" if args.apply else "Patch is applicable; pass --apply to install")

if __name__ == "__main__":
    main()
