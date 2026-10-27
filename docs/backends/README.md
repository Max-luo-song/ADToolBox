# Optional compute backends

TOOLBOX owns the public functional interfaces, portable contracts and integration layer. Third-party compute engines, checkpoints and datasets are installed separately. The default install does not download models or copy backend source.

| Capability | Integration profile | Installation / scope |
|---|---|---|
| Dynamic reconstruction and scene editing | DGGT single-view Mode 2 + extension patch | [Reconstruction profile](reconstruction.md) |
| Closed-loop inference | WorldEngine / SimEngine + policy client | [Policy profile](policy.md) |
| Supervised ILFT | WorldEngine / AlgEngine | [Policy profile](policy.md) |
| Instance discovery | Optional Ultralytics detector + ByteTrack | Explicit local weights; review upstream code/model terms |
| Semantic video search / token forensics | User-provided JSON process | [Bridge protocol](../configuration.md); no upstream model implementation is bundled |

These profiles describe compatibility contracts, not completed GPU reproduction. See [validation status](../validation.md) before interpreting a supported interface as a measured result.

中文：后端按需安装，公开目录与 API 按功能命名。这里明确列出实际后端和支持条件，保留应有归属。接口验证与 GPU 复现分开记录。
