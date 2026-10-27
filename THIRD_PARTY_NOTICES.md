# Third-party notices

ADToolbox's standalone integration code is distributed under Apache-2.0. That license does not relicense bundled engines, libraries, weights, assets or datasets.

## Bundled engine source

| Component | Relationship | Attribution / terms |
|---|---|---|
| Reconstruction engine (`projects/reconstruction`) | Vendored with ADToolbox editing/export extensions applied | Derived from an Apache-2.0 project by Xiaomi Corporation and Tsinghua University (2025). Original LICENSE and NOTICE retained in `licenses/reconstruction/` and in the project directory. The upstream NOTICE discusses derived VGGT (facebookresearch) material and separate checkpoint terms; review them before use. |
| Simulation and training stack (`projects/simulation`) | Vendored simulation engine (`engine/`) and policy training engine (`policy/`) with ADToolbox scene-conversion additions | Derived from an Apache-2.0 framework by OpenDriveLab contributors. License retained in `licenses/simulation/` and in the project directory. Review all subcomponents (including mmdetection3d-derived code under `policy/`) separately. |
| TAPIP3D (`projects/reconstruction/third_party/TAPIP3D`) | Vendored 4D tracking utility used by reconstruction inference | Upstream project by Zhejiang University / Ant Group contributors; no license file ships with the vendored subset. Review upstream terms before redistribution. |
| cotracker (`projects/reconstruction/third_party/cotracker`) | Vendored point-tracking helpers used through TAPIP3D | Derived from co-tracker (facebookresearch). Upstream code is CC-BY-NC (non-commercial) with separate model terms; review before any commercial use. |

## Tool-level integrations

| Component | Relationship | Attribution / terms |
|---|---|---|
| the ADToolbox authors (2026) | Adapted standalone coordinate and video preprocessing helpers | Apache-2.0; modifications noted in source and NOTICE. No agent orchestration code included. |
| MCP Python SDK | Optional protocol runtime | [Upstream](https://github.com/modelcontextprotocol/python-sdk), MIT. Installed as a dependency. |
| Ultralytics / ByteTrack integration | Optional detector/tracker, not vendored | [Ultralytics licensing](https://www.ultralytics.com/license). Review applicable AGPL-3.0 / enterprise terms and model terms before enabling. |
| SegFormer / Transformers / PyTorch | Optional preprocessing environment, not vendored | Obtain code and model licenses from the exact packages and model providers you select. |

No checkpoints, dataset videos, downloaded assets or private project history are distributed. A bundled engine is not an assertion that all of its transitive dependencies use Apache-2.0. Preserve upstream notices when redistributing modified engine files.
