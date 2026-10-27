"""Natural-language MCP client for the external Scene Editing server."""
from __future__ import annotations
import asyncio, json, os, sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

def text_result(result: Any) -> str:
    return "\n".join(item.text for item in result.content if getattr(item, "text", None) is not None)

def _chat_kwargs(messages, openai_tools):
    """Build chat completion kwargs, omitting temperature when unset or unsupported."""
    kwargs = {
        "model": os.environ.get("OPENAI_MODEL", "qwen3.5-plus"),
        "messages": messages,
        "tools": openai_tools,
        "tool_choice": "auto",
        "max_tokens": int(os.environ.get("OPENAI_MAX_TOKENS", "2048")),
    }
    temp = os.environ.get("OPENAI_TEMPERATURE")
    if temp:
        kwargs["temperature"] = float(temp)
    return kwargs

async def run(payload: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from openai import AsyncOpenAI
    # Default to the vendored scene_editing core; TOOLBOX_EDITING_ROOT overrides.
    _default = Path(__file__).resolve().parent
    root = Path(os.environ.get("TOOLBOX_EDITING_ROOT", str(_default))).resolve()
    load_dotenv(root / ".env.local", override=False)
    output_path = str(Path(payload["output_path"]).resolve())
    env = os.environ.copy()
    env.update({"ASSETS_DIR": str(root / "assets"), "SMPL_ASSETS_DIR": str(root / "assets" / "pedestrian"), "OUTPUT_DIR": str(Path(output_path).parent)})
    server = StdioServerParameters(command=sys.executable, args=["-m", "scene_editing"], cwd=str(root), env=env)
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=os.environ.get("OPENAI_BASE_URL"))
    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        tools = (await session.list_tools()).tools
        openai_tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema or {"type": "object"}}} for t in tools]
        required_asset = payload.get("asset_path")
        asset_instruction = f" The user-selected asset is exactly {required_asset}; use this path for the replacement and do not substitute another asset." if required_asset else ""
        output_kind = payload.get("output_kind", "rigid_config")
        if output_kind == "ego_trajectory_offsets":
            task_instruction = "Convert the natural-language request into camera-local ego render-trajectory offsets. Use set_ego_trajectory_offsets with [forward, left, up] values in scene backend native units; do not invent meters conversion. Supply either one constant vector or exactly frame_count vectors when frame_count is provided. This changes rendered camera poses, not the closed-loop policy's braking decision. Then call save_ego_trajectory_offsets with the exact output_path. Do not call RigidNodes tools."
            save_tool = "save_ego_trajectory_offsets"
        else:
            task_instruction = "Convert the user's natural-language edit request into scene backend RigidNodes tool calls. If instance_info_path is provided, pass it to start_config. For an existing instance, trajectory waypoints are offsets from that instance's original position on each frame, not absolute scene positions. For a 40-frame front-car cut-in, aim to enter the ego lane around frame 12 and compensate the original vehicle motion through frame 39; a constant offset after frame 12 does not hold lane position. The instance_info contains 2D boxes, not metric 3D positions: never turn a requested final gap into a forward offset or claim lane membership is measured from boxes alone. Keep the vertical offset at zero unless the request explicitly requires height change; do not claim road contact is verified. Use list_assets before choosing an asset. For add or replace, use exactly the supplied asset_path and asset_scale; if either is absent, report the missing value and do not invent one. Preview the draft, then call save_config with the exact supplied output_path."
            save_tool = "save_config"
        messages = [{"role": "system", "content": "You operate the scene backend Scene Editing MCP server. " + task_instruction + " Do not claim success without the required save tool." + asset_instruction}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        calls_seen = []
        saved_path = None
        for _ in range(12):
            response = await client.chat.completions.create(**_chat_kwargs(messages, openai_tools))
            message = response.choices[0].message
            calls = message.tool_calls or []
            if not calls:
                if saved_path: break
                raise RuntimeError(message.content or "Scene Editing finished without save_config")
            messages.append(message.model_dump(exclude_none=True))
            for call in calls:
                try: arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc: raise RuntimeError(f"invalid Scene Editing tool arguments: {exc}") from exc
                result = await session.call_tool(call.function.name, arguments=arguments)
                result_text = text_result(result)
                calls_seen.append(call.function.name)
                if call.function.name == save_tool:
                    try:
                        saved = json.loads(result_text)
                        saved_path = saved.get("saved_path") or (saved.get("saved_paths") or [None])[0]
                    except json.JSONDecodeError: pass
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})
        if not saved_path: raise RuntimeError("Scene Editing did not call save_config")
        return {"saved_path": str(Path(saved_path).resolve()), "tool_calls": calls_seen}

def main() -> None:
    try: print(json.dumps({"status": "success", "result": asyncio.run(run(json.load(sys.stdin)))}, ensure_ascii=False))
    except Exception as exc:
        subs = getattr(exc, "exceptions", None)
        if subs:
            import traceback, sys as _sys
            for sub in subs:
                traceback.print_exception(type(sub), sub, sub.__traceback__, file=_sys.stderr)
            exc = subs[0]
        print(json.dumps({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False)); raise SystemExit(1)

if __name__ == "__main__": main()
