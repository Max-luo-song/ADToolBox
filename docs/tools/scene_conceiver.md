# Scenario conception / 场景方案设计

`scene_conceiver` · Runtime: **llm**

Design a base-video query and minimal, plausible scenario edits.

## Run independently

```python
from adtoolbox.tools.scene_conceiver import run
result = run(input_data, config=runtime_config)
```

```bash
adtoolbox schema scene_conceiver
adtoolbox run scene_conceiver --input examples/scene_conceiver/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `scene_requirement` | `object` | yes |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Base-video requirements, retrieval query, edit ideas and expected effect.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

An expected ego response is a test objective; this tool does not control the evaluated policy.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`adtoolbox schema scene_conceiver` 提供完整协议。标为 LLM 的能力需要模型配置；标为 bundled engine 的能力驱动 projects/ 下内嵌引擎（需自备环境与权重），具体见[引擎指南](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
