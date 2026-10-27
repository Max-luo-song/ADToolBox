# Contracts

All 14 Python entries accept `run(input_data: dict, *, config: dict | None = None)` and return:

```json
{
  "schema_version": "1.0",
  "tool": "edit_config_generator",
  "status": "success",
  "result": {},
  "artifacts": [],
  "warnings": [],
  "error": null
}
```

Expected input, configuration and backend failures use `status=error` with `error.code` and `error.message`. CLI execution returns 1 for tool errors and 2 for malformed requests. Unexpected programming defects can still raise exceptions; they are not silently converted into successes.

Artifacts contain `kind` and a local absolute `path`. Local paths are resolved on the process host. No global workflow state, implicit previous result or automatic next-tool selection is involved.

## Coordinate conventions

Portable edits use `camera_local_meters` (forward, left, up) or explicitly backend-specific `backend_world` units. `meters_to_scene_units` must come from external calibration. The renderer adapter maps local axes to OpenCV camera axes (right, down, forward), uses each frame's camera pose, and locks road-plane trajectory displacement to the backend's horizontal plane. This is a renderer-specific assumption, not a general coordinate transform for arbitrary elevated roads. Rotations are XYZ degrees; scale is a positive multiplicative factor.

Instance boxes must use the same pixel geometry and frame sequence as reconstruction. Tracking with `pixel_space=reconstruction_input` applies the renderer profile's 518-pixel width / multiple-of-14 height / center-crop preprocessing. Source frame timing remains distinct from model-normalized Gaussian time.

Failure screening JSON uses `world_xy_meters`, `position=[x,y]`, `heading` in radians, `length` and `width` in meters, and per-frame `ego` / `actors`. See the synthetic example. Box touching counts as overlap. Static objects, responsibility and road compliance are outside this screen.

## 中文

所有工具使用相同结果信封，不依赖上一步状态。坐标、尺度、帧序列必须显式匹配；未校准场景不能假定米制。视觉验证不会替代物理或官方评测。失败筛选仅针对动态车辆二维有向框的逐帧重叠。
