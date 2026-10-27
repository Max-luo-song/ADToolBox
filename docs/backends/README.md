# Bundled compute engines

ADToolbox bundles the heavy compute engines it drives, so the tools below run against real,
included source instead of an interface you have to fill in yourself. Model weights and datasets
are still downloaded separately; each engine keeps its own Python environment. Legal attribution
for the bundled engines lives in [NOTICE](../../NOTICE) and
[third-party notices](../../THIRD_PARTY_NOTICES.md).

| Capability | Bundled engine | Setup guide |
|---|---|---|
| Dynamic reconstruction and scene editing | `projects/reconstruction` — single-view dynamic Gaussian reconstruction with editing extensions | [Reconstruction guide](reconstruction.md) |
| Closed-loop policy evaluation | `projects/simulation/engine` — Gaussian closed-loop simulation | [Policy guide](policy.md) |
| Supervised policy fine-tuning (ILFT) | `projects/simulation/policy` — training engine | [Policy guide](policy.md) |
| Instance discovery | Optional Ultralytics detector + ByteTrack (`tools/instance_discovery/tracker.py`) | Explicit local weights; review upstream code/model terms |
| Semantic video search / token forensics | `projects/video_search` / `projects/failure_analysis` | Included; need your model endpoints and data |

The engines are verified for source completeness and integration wiring. GPU reconstruction,
closed-loop metrics and training outcomes depend on your hardware, checkpoints and data; see
[validation status](../validation.md) before interpreting a supported interface as a measured result.

中文：重型引擎已随仓库提供（源码级内嵌），权重与数据仍需按各自指南准备；每个引擎使用独立
Python 环境，通过配置文件指定解释器与路径。法定归属信息集中在 NOTICE 与第三方声明中。
