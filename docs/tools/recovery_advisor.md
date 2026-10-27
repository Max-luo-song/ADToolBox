# Execution diagnostics / 执行诊断建议

`recovery_advisor` · Runtime: **native**

Return diagnostic advice without running retries or dispatching other tools.

## Run independently

```python
from driving_toolbox.tools.recovery_advisor import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema recovery_advisor
tool-box run recovery_advisor --input examples/recovery_advisor/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `evaluation` | `object` | yes |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

`action`, `reason`, suggestions and `executes_actions=false`.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Returns advice only. Does not retry, roll back, maintain history or dispatch any tool.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema recovery_advisor` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
