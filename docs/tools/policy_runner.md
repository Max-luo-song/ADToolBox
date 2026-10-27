# Closed-loop policy evaluation / 闭环策略运行

`policy_runner` · Runtime: **bundled engine**

Run an explicitly configured closed-loop policy and collect runner reports.

## Run independently

```python
from adtoolbox.tools.policy_runner import run
result = run(input_data, config=runtime_config)
```

```bash
adtoolbox schema policy_runner
adtoolbox run policy_runner --input examples/policy_runner/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `config_path` | `string` | yes |
| `checkpoint_path` | `string` | yes |
| `model_name` | `string` | yes |
| `data_type` | `string` | yes |
| `react_type` | `string` | no |
| `scenario_data_folder_path` | `string` | no |
| `asset_folder_path` | `string` | no |
| `scenario_pkl_file_name` | `string` | no |
| `num_scenarios` | `integer` | no |
| `max_frame_per_scene` | `integer` | no |
| `with_metric_manager` | `boolean` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Fresh runner reports and result directory.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Requires existing compatible scene assets, map/metric annotations and a configured policy. Custom reconstruction outputs must be converted explicitly before simulation; this tool does not invent map annotations or assume metric scale. See the bundled converter chain in `projects/simulation/scripts` (staged by `scripts/convert_scene.py`).

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`adtoolbox schema policy_runner` 提供完整协议。标为 LLM 的能力需要模型配置；标为 bundled engine 的能力驱动 projects/ 下内嵌引擎（需自备环境与权重），具体见[引擎指南](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
