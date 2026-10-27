# Driving video retrieval / 驾驶视频检索

`video_retriever` · Runtime: **native + optional backend**

Rank captioned local video records; use a configured backend for semantic retrieval.

## Run independently

```python
from driving_toolbox.tools.video_retriever import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema video_retriever
tool-box run video_retriever --input examples/video_retriever/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `requirement` | `string` | yes |
| `catalog_path` | `string` | no |
| `top_k` | `integer` | no |
| `export_path` | `string` | no |
| `min_score` | `number` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

`matched`, ranked `candidates`, selected `video_path` and `method`.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Native retrieval uses BM25 over supplied captions and tags. It does not inspect pixels. Visual-semantic recall, reranking and clip export can be supplied by a JSON backend. No automatic internet fallback.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema video_retriever` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
