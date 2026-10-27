# Supervised policy fine-tuning / 策略监督微调

`policy_trainer` · Runtime: **optional backend**

Launch supervised ILFT with explicit data, configuration and initial weights.

## Run independently

```python
from driving_toolbox.tools.policy_trainer import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema policy_trainer
tool-box run policy_trainer --input examples/policy_trainer/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `config_path` | `string` | yes |
| `base_checkpoint` | `string` | yes |
| `ann_file` | `string` | yes |
| `token_file` | `string` | yes |
| `work_dir` | `string` | yes |
| `epochs` | `integer` | no |
| `training_mode` | `string` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

New checkpoint, training mode, epochs and work directory.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Only supervised ILFT. Requires explicit annotations, token file, compatible config and base weights. Filtered cases are not automatically valid training labels. The work directory must be empty.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema policy_trainer` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
