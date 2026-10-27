# Real capability projects / 实际能力项目

This directory contains the compute implementations used by the tool library. The private Agent scheduler is not included.

| Project | Source | Tool support |
|---|---|---|
| `reconstruction/` | Fixed-revision Git submodule + `extensions/reconstruction` | Model inference, Gaussian export, instance editing, camera offsets and rendering |
| `scene_editing/` | Full migrated editing service and model-to-MCP client | Asset discovery, edit builders, preview, save, natural-language configuration |
| `video_search/` | Full migrated indexing/retrieval pipeline | Video chunking, multimodal embeddings, vector store, reranking, export, explicit internet search |
| `failure_analysis/` | Full migrated evidence investigation project | Official results, image inspection, numeric evidence and token reports |
| `simulation/` | Fixed-revision Git submodule + `extensions/simulation` | Closed-loop simulation, policy execution, scene conversion and supervised training |

Clone with `git clone --recurse-submodules`. GitHub displays the two upstream projects as linked directories pinned to specific commits. Their code and dependencies retain their original licenses; they are not represented as original TOOLBOX model implementations.

For an existing clone: `git submodule update --init --recursive`. Then run the setup script for the compute project you need. Weights, private data, indexes, environments and credentials are not committed. The Python wheel alone does not contain these source projects: use this repository or set `TOOLBOX_PROJECTS_DIR` to its `projects` directory.

中文：这里是真实实现，不是后端接口占位。重建和仿真/训练通过子模块固定公开项目源码，再应用我们原有的扩展；编辑、检索和失败分析保留完整工具组件。框架的全局编排不在这里。
