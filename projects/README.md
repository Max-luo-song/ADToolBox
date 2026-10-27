# Real capability projects / 实际能力项目

This directory contains the compute implementations used by the tool library. The private Agent scheduler is not included.

| Project | Source | Tool support |
|---|---|---|
| `reconstruction/` | Bundled engine source with ADToolbox editing/export extensions applied | Model inference, Gaussian export, instance editing, camera offsets and rendering |
| `scene_editing/` | Full editing service and model-to-MCP client | Asset discovery, edit builders, preview, save, natural-language configuration |
| `video_search/` | Full indexing/retrieval pipeline | Video chunking, multimodal embeddings, vector store, reranking, export, explicit internet search |
| `failure_analysis/` | Full evidence investigation project | Official results, image inspection, numeric evidence and token reports |
| `simulation/` | Bundled engine source (`engine/` + `policy/`) plus the scene-conversion and training-export scripts in `scripts/` | Closed-loop simulation, policy execution, scene conversion and supervised training |

Everything is tracked directly in this repository; no submodule step is required. Engine code
retains its original licenses and notices (see `NOTICE` and `THIRD_PARTY_NOTICES.md`); it is not
represented as original ADToolbox model implementations. Weights, private data, indexes,
environments and credentials are not committed. The Python wheel does not contain these source
projects: use this repository or set `ADTOOLBOX_PROJECTS_DIR` to its `projects` directory.

Run `python scripts/setup_project.py <reconstruction|simulation>` to verify a bundled engine tree
and print its remaining setup steps.

中文：这里是真实实现，不是后端接口占位。重建、仿真/训练引擎源码已内嵌并应用了自有扩展；
编辑、检索和失败分析保留完整工具组件。框架的全局编排不在这里。
