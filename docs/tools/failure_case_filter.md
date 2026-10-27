# Failure case curation / 失败案例筛选

`failure_case_filter` · Runtime: **native**

Screen heading-aware dynamic vehicle box overlaps and optionally append cases to JSONL.

## Run independently

```python
from adtoolbox.tools.failure_case_filter import run
result = run(input_data, config=runtime_config)
```

```bash
adtoolbox schema failure_case_filter
adtoolbox run failure_case_filter --input examples/failure_case_filter/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `scenario_path` | `string` | yes |
| `case_id` | `string` | no |
| `output_library` | `string` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Failure flag, dynamic-box overlap events and optional JSONL library entry.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Accepts portable JSON rather than untrusted pickle. Requires world XY meters, heading in radians and explicit box dimensions. This is 2D oriented-box screening, not official collision responsibility, road compliance, static geometry or continuous-time checking. Concurrent writes to one JSONL file require external coordination.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`adtoolbox schema failure_case_filter` 提供完整协议。标为 LLM 的能力需要模型配置；标为 bundled engine 的能力驱动 projects/ 下内嵌引擎（需自备环境与权重），具体见[引擎指南](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
