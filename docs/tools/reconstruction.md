# Dynamic scene reconstruction / 动态场景重建

`reconstruction` · Runtime: **bundled engine**

Reconstruct video into dynamic Gaussian assets and camera parameter artifacts.

## Run independently

```python
from adtoolbox.tools.reconstruction import run
result = run(input_data, config=runtime_config)
```

```bash
adtoolbox schema reconstruction
adtoolbox run reconstruction --input examples/reconstruction/input.json --config configs/runtime.local.json
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

Requires the bundled reconstruction engine with its environment and weights prepared as described in the backend guide. Frames default to 40 but are configurable. Metric scale remains unknown until externally calibrated. GPU inference has not been reproduced by this release’s portable checks.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`adtoolbox schema reconstruction` 提供完整协议。标为 LLM 的能力需要模型配置；标为 bundled engine 的能力驱动 projects/ 下内嵌引擎（需自备环境与权重），具体见[引擎指南](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
