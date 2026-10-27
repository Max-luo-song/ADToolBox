# Edit configuration compiler / 编辑配置编译

`edit_config_generator` · Runtime: **native + llm**

Compile typed edits into a portable, validated scene-edit document.

## Run independently

```python
from adtoolbox.tools.edit_config_generator import run
result = run(input_data, config=runtime_config)
```

```bash
adtoolbox schema edit_config_generator
adtoolbox run edit_config_generator --input examples/edit_config_generator/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `operations` | `array` | no |
| `edit_plan` | `string` | no |
| `instances` | `array` | no |
| `coordinate_system` | `string` | no |
| `output_path` | `string` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Validated portable `edit_config`; optional saved JSON artifact.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Supports trajectory editing, rotation in XYZ degrees, scale, remove, add and replace. Typed operations work without a model. Natural-language edit plans need a model. Asset paths and scale are explicit. Trajectory values are offsets, not absolute positions.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`adtoolbox schema edit_config_generator` 提供完整协议。标为 LLM 的能力需要模型配置；标为 bundled engine 的能力驱动 projects/ 下内嵌引擎（需自备环境与权重），具体见[引擎指南](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
