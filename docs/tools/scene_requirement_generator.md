# Scenario requirements / 场景需求生成

`scene_requirement_generator` · Runtime: **llm**

Translate a failure analysis into grounded scenario requirements.

## Run independently

```python
from adtoolbox.tools.scene_requirement_generator import run
result = run(input_data, config=runtime_config)
```

```bash
adtoolbox schema scene_requirement_generator
adtoolbox run scene_requirement_generator --input examples/scene_requirement_generator/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `failure_analysis` | `object` | yes |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Scenario description, required conditions, actors, risks and target effect.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Uses an explicitly configured model. Generated requirements are proposals, not verified facts.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`adtoolbox schema scene_requirement_generator` 提供完整协议。标为 LLM 的能力需要模型配置；标为 bundled engine 的能力驱动 projects/ 下内嵌引擎（需自备环境与权重），具体见[引擎指南](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
