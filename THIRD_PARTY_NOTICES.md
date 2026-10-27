# Third-party notices

TOOLBOX's standalone integration code is distributed under Apache-2.0. That license does not relicense optional engines, libraries, weights, assets or datasets.

| Component | Relationship | Attribution / terms |
|---|---|---|
| adagent contributors (2026) | Adapted standalone coordinate and video preprocessing helpers | Apache-2.0; modifications noted in source and NOTICE. No agent orchestration code included. |
| DGGT, Xiaomi Corporation and Tsinghua University (2025) | Optional reconstruction backend; adapted inference/helper changes supplied as a patch | Original top-level LICENSE and NOTICE retained in `licenses/reconstruction/`; [upstream](https://github.com/xiaomi-research/dggt). The upstream NOTICE discusses derived VGGT material and separate checkpoint terms; review them before use. |
| WorldEngine, OpenDriveLab | Optional simulation / training engine, not vendored | [Upstream](https://github.com/OpenDriveLab/WorldEngine), Apache-2.0 at the inspected top level. Review all subcomponents separately. |
| MCP Python SDK | Optional protocol runtime | [Upstream](https://github.com/modelcontextprotocol/python-sdk), MIT. Installed as a dependency. |
| Ultralytics / ByteTrack integration | Optional detector/tracker, not vendored | [Ultralytics licensing](https://www.ultralytics.com/license). Review applicable AGPL-3.0 / enterprise terms and model terms before enabling. |
| SegFormer / Transformers / PyTorch | Optional preprocessing environment, not vendored | Obtain code and model licenses from the exact packages and model providers you select. |

No checkpoints, dataset videos, downloaded assets, video-search vendor implementation or private project history are distributed. A backend profile is not an assertion that all of its transitive dependencies use Apache-2.0. Preserve upstream notices when applying or redistributing modified engine files.
