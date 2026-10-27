# Scene consistency checks / 场景一致性检查

`scene_validator` · Runtime: **native + optional llm**

Check artifacts and optionally inspect sampled video frames; report unverified physical properties.

## Run independently

```python
from driving_toolbox.tools.scene_validator import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema scene_validator
tool-box run scene_validator --input examples/scene_validator/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `rendered_video` | `string` | yes |
| `scene_requirement` | `object` | yes |
| `scene_ply` | `string` | no |
| `scene_config` | `string` | no |
| `visual_check` | `boolean` | no |
| `frame_count` | `integer` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Artifact validity, optional visual assessment, issues and pass status.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

File checks alone never pass requirement validation. `visual_check=true` needs ffmpeg, ffprobe and a vision-capable endpoint; sampled frames are sent to that endpoint. Physical and numerical geometry validity remain unverified.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema scene_validator` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
