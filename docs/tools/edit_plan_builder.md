# Scene edit planning / 场景编辑计划

`edit_plan_builder` · Runtime: **llm**

Ground an edit plan in supplied instances and coordinate information.

## Run independently

```python
from driving_toolbox.tools.edit_plan_builder import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema edit_plan_builder
tool-box run edit_plan_builder --input examples/edit_plan_builder/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `edit_request` | `string` | yes |
| `scene_info` | `object` | no |
| `instances` | `array` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Natural-language plan, selected instances, proposed operations and constraints.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Requires an explicit model endpoint. Missing instance identities and scale must remain constraints.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema edit_plan_builder` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
