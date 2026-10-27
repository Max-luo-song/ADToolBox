# Dynamic scene reconstruction / 动态场景重建

`reconstruction` · Runtime: **optional backend**

Reconstruct video into dynamic Gaussian assets and camera parameter artifacts.

## Run independently

```python
from driving_toolbox.tools.reconstruction import run
result = run(input_data, config=runtime_config)
```

```bash
tool-box schema reconstruction
tool-box run reconstruction --input examples/reconstruction/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `video_path` | `string` | yes |
| `num_frames` | `integer` | no |
| `export_render_state` | `boolean` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Scene PLY, camera parameters, rendered video, input frames and frame metadata.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Requires an installed backend and the extensions described in the backend guide. Frames default to 40 but are configurable. Metric scale remains unknown until externally calibrated. GPU inference has not been reproduced by this release’s portable checks.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`tool-box schema reconstruction` 提供完整协议。标为 LLM 的能力需要模型配置；标为 optional backend 的能力需要自行安装后端，具体见[后端说明](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
