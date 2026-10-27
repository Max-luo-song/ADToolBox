# Traffic instance tracking / 交通实例跟踪

`instance_discovery` · Runtime: **optional vision / engine**

Detect and track visible traffic participants with an optional tracking backend.

## Run independently

```python
from adtoolbox.tools.instance_discovery import run
result = run(input_data, config=runtime_config)
```

```bash
adtoolbox schema instance_discovery
adtoolbox run instance_discovery --input examples/instance_discovery/input.json --config configs/runtime.local.json
```

## Inputs

| Field | JSON type | Required |
|---|---|---|
| `video_path` | `string` | no |
| `image_dir` | `string` | no |
| `weights` | `string` | no |
| `num_frames` | `integer` | no |
| `confidence` | `number` | no |
| `classes` | `array` | no |
| `output_path` | `string` | no |
| `pixel_space` | `string` | no |

The catalog contains the full machine-readable schema. Runtime configuration is passed separately from tool input. See [configuration](../configuration.md).

## Outputs

Tracked `instances` with IDs, classes and per-frame pixel boxes.

All results use the [common envelope](../contracts.md). Errors are explicit; a missing backend is never presented as a successful mock result.

## Scope and requirements

Install the vision extra and provide weights explicitly, or configure a separate process. No masks or metric 3D geometry are claimed. Use `pixel_space=reconstruction_input` for the bundled renderer profile; source-image boxes cannot be mixed with resized renderer images.

## 中文说明

本工具可通过 Python、CLI 或可选 MCP 独立调用。上表说明输入字段，`adtoolbox schema instance_discovery` 提供完整协议。标为 LLM 的能力需要模型配置；标为 bundled engine 的能力驱动 projects/ 下内嵌引擎（需自备环境与权重），具体见[引擎指南](../backends/README.md)。输出包含状态、结果、产物和警告，缺少依赖时不会返回模拟成功。此版本的 GPU 重建、仿真和训练尚未完成独立复现验证。
