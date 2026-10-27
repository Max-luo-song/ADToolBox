# Scene editing and rendering / 场景编辑与渲染

`scene_editor` · Runtime: **optional backend**

Apply instance edits, coordinate conversion and optional camera offsets.

## Run independently

```python
from driving_toolbox.tools.scene_editor import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema scene_editor
tool-box run scene_editor --input examples/scene_editor/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `video_path` | `string` | yes |
| `edit_config_path` | `string` | no |
| `camera_params_path` | `string` | no |
| `meters_to_scene_units` | `number` | no |
| `num_frames` | `integer` | no |
| `ego_trajectory_offsets_path` | `string` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Edited scene assets, rendered video and source edit configuration.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Uses the same video and frame range as instance discovery. Camera-local metric trajectory edits require an explicit scale and matching camera poses. Existing-instance edits also require scene_editing.instance_info_path. Scene editing reruns the reconstruction backend.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema scene_editor` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
