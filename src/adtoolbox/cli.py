import argparse
import json
import sys

from .catalog import get_tool, list_tools
from .core.io import read_json, write_json
from .registry import run_tool


def main():
    parser = argparse.ArgumentParser(
        prog="tool-box", description="Independent tools for driving scenario engineering"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List all tools without importing GPU dependencies")
    listing.add_argument("--json", action="store_true")
    schema = commands.add_parser("schema", help="Show a tool's input JSON Schema")
    schema.add_argument("tool")
    run = commands.add_parser("run", help="Run exactly one tool")
    run.add_argument("tool")
    run.add_argument("--input", required=True, help="Input JSON path, or - for stdin")
    run.add_argument("--config", help="Runtime configuration JSON")
    run.add_argument("--output", help="Also save the complete result envelope")
    doctor = commands.add_parser("doctor", help="Check optional dependencies and configured paths")
    doctor.add_argument("--config")
    args = parser.parse_args()
    try:
        if args.command == "list":
            value = (
                list_tools()
                if args.json
                else "\n".join(f"{x['name']:30} {x['title']} [{x['runtime']}]" for x in list_tools())
            )
        elif args.command == "schema":
            value = get_tool(args.tool)["input_schema"]
        elif args.command == "doctor":
            from .doctor import diagnose

            value = diagnose(read_json(args.config) if args.config else {})
        else:
            data = json.load(sys.stdin) if args.input == "-" else read_json(args.input)
            value = run_tool(args.tool, data, config=read_json(args.config) if args.config else {})
            if args.output:
                write_json(args.output, value)
        print(
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        )
        if isinstance(value, dict) and value.get("status") == "error":
            raise SystemExit(1)
    except (ValueError, OSError) as exc:
        print(
            json.dumps({"status": "error", "error": {"code": "invalid_request", "message": str(exc)}}),
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
