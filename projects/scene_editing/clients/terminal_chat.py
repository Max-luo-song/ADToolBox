"""Terminal chat client for Scene Editing MCP.

This file intentionally lives outside the scene_editing package. The package stays
as a pure MCP server; this script connects an OpenAI-compatible chat API to the
local MCP tools.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_ROOT / ".env.local", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:30000/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key. SGLang local servers commonly accept any non-empty value.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "qwen3"),
        help="Model name exposed by the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("OPENAI_MAX_TOKENS", "2048")),
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=int(os.environ.get("MAX_TOOL_ROUNDS", "8")),
        help="Maximum consecutive tool-call rounds for one user message.",
    )
    parser.add_argument(
        "--show-tool-process",
        action="store_true",
        default=os.environ.get("SHOW_TOOL_PROCESS", "").lower()
        in {"1", "true", "yes", "on"},
        help="Print each model/tool round while answering.",
    )
    parser.add_argument("--assets-dir", default=os.environ.get("ASSETS_DIR", "assets"))
    parser.add_argument(
        "--smpl-assets-dir",
        default=os.environ.get("SMPL_ASSETS_DIR", "assets/pedestrian"),
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs"))
    return parser.parse_args()


class SceneEditingTerminalClient:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - import guard for optional client
            raise SystemExit(
                "Missing dependency: openai. Install with `pip install -e .`."
            ) from exc

        self.args = args
        self.openai = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
        self.session: Any | None = None
        self.tools: list[Any] = []
        self.messages: list[dict[str, Any]] = []
        self.show_tool_process = bool(args.show_tool_process)

    async def __aenter__(self) -> "SceneEditingTerminalClient":
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:  # pragma: no cover - import guard for optional client
            raise SystemExit(
                "Missing dependency: mcp. Install with `pip install -e .`."
            ) from exc

        env = os.environ.copy()
        env.update(
            {
                "ASSETS_DIR": str((PROJECT_ROOT / self.args.assets_dir).resolve())
                if not Path(self.args.assets_dir).is_absolute()
                else self.args.assets_dir,
                "SMPL_ASSETS_DIR": str(
                    (PROJECT_ROOT / self.args.smpl_assets_dir).resolve()
                )
                if not Path(self.args.smpl_assets_dir).is_absolute()
                else self.args.smpl_assets_dir,
                "OUTPUT_DIR": str((PROJECT_ROOT / self.args.output_dir).resolve())
                if not Path(self.args.output_dir).is_absolute()
                else self.args.output_dir,
            }
        )
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "scene_editing"],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        self._exit_stack = AsyncExitStack()
        try:
            await self._exit_stack.enter_async_context(self.openai)
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server)
            )
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()
            self.tools = (await self.session.list_tools()).tools
            self.reset()
        except BaseException:
            await self._exit_stack.aclose()
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        return await self._exit_stack.__aexit__(exc_type, exc, tb)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self._system_prompt()}]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object"},
                },
            }
            for tool in self.tools
        ]

    def tool_names(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description or ''}" for tool in self.tools)

    async def answer(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        trace: list[str] = []
        turn_saved_paths: list[str] = []
        failed_tool_calls: dict[tuple[str, str, str], int] = {}

        for round_index in range(1, self.args.max_tool_rounds + 1):
            self._trace(f"[round {round_index}] 请求模型", trace)
            try:
                response = await self.openai.chat.completions.create(
                    model=self.args.model,
                    messages=self.messages,
                    tools=self.openai_tools(),
                    tool_choice="auto",
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                )
            except Exception as exc:
                summary = "\n".join(trace[-12:]) if trace else "(无工具调用记录)"
                return (
                    f"模型请求失败: {type(exc).__name__}: {exc}\n"
                    f"最近过程:\n{summary}"
                )

            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                assistant_message = message.model_dump(exclude_none=True)
                self.messages.append(assistant_message)
                if message.content:
                    self._trace(
                        f"[round {round_index}] 模型最终回复: {self._shorten(message.content)}",
                        trace,
                    )
                    guarded = self._guard_save_claim(message.content, turn_saved_paths)
                    if guarded != message.content:
                        self.messages[-1]["content"] = guarded
                        self._trace(
                            "[guard] 已拦截未保存却声称已保存的回复",
                            trace,
                        )
                    return guarded
                return self._format_empty_model_response(
                    message=message,
                    finish_reason=getattr(choice, "finish_reason", None),
                    trace=trace,
                )

            assistant_message, normalized_tool_calls = self._normalize_assistant_tool_calls(
                message.content,
                tool_calls,
                trace,
            )
            self.messages.append(assistant_message)

            self._trace(
                f"[round {round_index}] 模型请求 {len(tool_calls)} 个工具调用",
                trace,
            )
            for tool_call in normalized_tool_calls:
                self._trace(
                    (
                        f"[tool] {tool_call['name']} "
                        f"args={self._shorten(tool_call['arguments'])}"
                    ),
                    trace,
                )
                try:
                    tool_result = await self._call_mcp_tool(
                        tool_call["name"],
                        tool_call["arguments"],
                    )
                except Exception as exc:
                    tool_result = f"工具调用失败: {type(exc).__name__}: {exc}"
                self._trace(f"[tool result] {self._shorten(tool_result)}", trace)
                if tool_call["name"] == "save_config":
                    turn_saved_paths.extend(self._extract_saved_paths(tool_result))
                if self._is_tool_error(tool_result):
                    failure_key = (
                        tool_call["name"],
                        tool_call["arguments"],
                        tool_result,
                    )
                    failed_tool_calls[failure_key] = (
                        failed_tool_calls.get(failure_key, 0) + 1
                    )
                    if failed_tool_calls[failure_key] >= 2:
                        summary = "\n".join(trace[-12:])
                        return (
                            "检测到模型重复提交同一个失败工具调用,已停止。\n"
                            f"错误: {tool_result}\n"
                            f"建议修正: {self._format_tool_hint(tool_call['name'])}\n"
                            f"最近过程:\n{summary}"
                        )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result,
                    }
                )

        summary = "\n".join(trace[-12:]) if trace else "(无工具调用记录)"
        return f"工具调用轮次过多,已停止。最近过程:\n{summary}"

    async def list_assets(self) -> str:
        return await self._call_mcp_tool("list_assets", "{}")

    async def _call_mcp_tool(self, name: str, arguments_json: Any) -> str:
        if self.session is None:
            raise RuntimeError("MCP session is not initialized.")
        arguments, _ = self._parse_tool_arguments(arguments_json)
        result = await self.session.call_tool(name, arguments=arguments)
        return "\n".join(
            item.text for item in result.content if getattr(item, "text", None) is not None
        )

    def _normalize_assistant_tool_calls(
        self,
        content: str | None,
        tool_calls: list[Any],
        trace: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        assistant_tool_calls: list[dict[str, Any]] = []
        normalized: list[dict[str, str]] = []

        for tool_call in tool_calls:
            arguments, valid_json = self._parse_tool_arguments(
                tool_call.function.arguments
            )
            arguments_json = json.dumps(arguments, ensure_ascii=False)
            if not valid_json:
                self._trace(
                    (
                        f"[tool args normalized] {tool_call.function.name} "
                        f"原始参数不是 JSON object,已改为 {arguments_json}"
                    ),
                    trace,
                )

            assistant_tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": arguments_json,
                    },
                }
            )
            normalized.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": arguments_json,
                }
            )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": assistant_tool_calls,
        }
        if content:
            assistant_message["content"] = content
        return assistant_message, normalized

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> tuple[dict[str, Any], bool]:
        if isinstance(arguments, dict):
            return arguments, True
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {}, False
            if isinstance(parsed, dict):
                return parsed, True
            return {}, False
        return {}, False

    def _format_empty_model_response(
        self,
        message: Any,
        finish_reason: str | None,
        trace: list[str],
    ) -> str:
        dumped = message.model_dump(exclude_none=True)
        keys = ", ".join(sorted(dumped.keys())) or "(无字段)"
        reasoning = dumped.get("reasoning_content") or dumped.get("reasoning")
        if reasoning:
            self._trace(
                f"[empty response] reasoning={self._shorten(str(reasoning))}",
                trace,
            )
        self._trace(
            (
                "[empty response] 模型没有返回 content 或 tool_calls; "
                f"finish_reason={finish_reason}, message字段={keys}"
            ),
            trace,
        )
        summary = "\n".join(trace[-8:]) if trace else "(无过程记录)"
        hint = (
            "模型返回了空回复:没有 content,也没有 OpenAI 标准 tool_calls。\n"
            "这通常表示 SGLang 的 tool-call parser 没有成功产出结构化工具调用,"
            "或者 Qwen3 只生成了 reasoning/thinking 而没有最终回答。\n"
            f"finish_reason: {finish_reason}\n"
            f"message字段: {keys}\n"
            f"最近过程:\n{summary}"
        )
        return hint

    def _trace(self, message: str, trace: list[str]) -> None:
        trace.append(message)
        if self.show_tool_process:
            print(f"\n过程> {message}")

    @staticmethod
    def _shorten(text: str, limit: int = 500) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    @staticmethod
    def _is_tool_error(tool_result: str) -> bool:
        return tool_result.startswith(
            (
                "Input validation error:",
                "参数错误:",
                "保存失败:",
                "工具调用失败:",
            )
        )

    @staticmethod
    def _format_tool_hint(tool_name: str) -> str:
        return "检查工具参数是否符合 schema;可先调用 preview_config 查看当前草稿。"

    @staticmethod
    def _extract_saved_paths(tool_result: str) -> list[str]:
        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            return []
        paths: list[str] = []
        saved_path = payload.get("saved_path")
        if isinstance(saved_path, str):
            paths.append(saved_path)
        saved_paths = payload.get("saved_paths")
        if isinstance(saved_paths, list):
            paths.extend(path for path in saved_paths if isinstance(path, str))
        return paths

    @staticmethod
    def _guard_save_claim(content: str, saved_paths: list[str]) -> str:
        if saved_paths:
            return content
        save_claim_markers = (
            "已保存",
            "保存到",
            "保存路径",
            "默认路径",
            "saved",
            "Saved",
        )
        if not any(marker in content for marker in save_claim_markers):
            return content
        return (
            "配置草稿已生成,但本轮没有调用 save_config,因此尚未写入文件。\n"
            "需要落盘时请继续输入“保存到默认路径”或指定 output_path。"
        )

    def _system_prompt(self) -> str:
        return (
            "你是 Scene Editing 终端助手。用户要生成、检查或保存 3DGS 编辑配置时,"
            "必须优先使用可用 MCP 工具。不要凭空编造资产路径;需要资产时先调用"
            " list_assets 或 list_smpl_assets。工具结果已经是可信的配置或保存结果。"
            "位置和路径点的坐标统一为 [x, y, z] = [前, 左, 上],单位为米。"
            "前方 x 为正、后方 x 为负;左侧 y 为正、右侧 y 为负;上方 z 为正、下方 z 为负。"
            "例如前方8米、右侧1.5米、高度0米必须写为 [8, -1.5, 0]。"
            "只有调用 save_config 且工具结果包含 saved_path 或 saved_paths 时,"
            "才可以说配置已保存或给出保存路径。仅生成/追加草稿时必须明确说明"
            "尚未保存,不要把默认输出路径说成保存路径。最终用中文简洁总结,"
            "可包含关键对象、位置和时间窗口;保存路径只在确实保存后给出。"
        )


async def main_async() -> None:
    args = parse_args()
    async with SceneEditingTerminalClient(args) as client:
        print(
            "Scene Editing terminal chat. Type /exit to quit, /tools to list tools, "
            "/assets to list assets, /trace to toggle process output, "
            "/reset to clear history."
        )
        while True:
            try:
                user_text = await asyncio.to_thread(input, "\n你> ")
            except EOFError:
                print()
                break

            user_text = user_text.strip()
            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                break
            if user_text == "/reset":
                client.reset()
                print("\n助手> 已清空对话历史。")
                continue
            if user_text == "/tools":
                print(f"\n助手>\n{client.tool_names()}")
                continue
            if user_text == "/assets":
                print(f"\n助手>\n{await client.list_assets()}")
                continue
            if user_text == "/trace":
                client.show_tool_process = not client.show_tool_process
                state = "开启" if client.show_tool_process else "关闭"
                print(f"\n助手> 工具调用过程显示已{state}。")
                continue

            answer = await client.answer(user_text)
            print(f"\n助手> {answer}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
