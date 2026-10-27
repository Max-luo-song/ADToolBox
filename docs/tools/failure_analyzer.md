# Failure evidence analysis / 失败证据分析

`failure_analyzer` · Runtime: **llm**

Analyze observations and uncertainty without inventing missing evidence.

## Run independently

```python
from driving_toolbox.tools.failure_analyzer import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema failure_analyzer
tool-box run failure_analyzer --input examples/failure_analyzer/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `failure_case` | `object` | no |
| `token` | `string` | no |
| `evidence` | `array` | no |
| `image_paths` | `array` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

`failure_analysis`: failure types, hypotheses, observations and uncertainty.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Uses only supplied evidence. A token is metadata; a token-only forensic workflow requires a configured JSON backend. Local image attachments can be provided with `image_paths`; they are sent to the selected model endpoint.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema failure_analyzer` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
