# Reconstruction engine

Dynamic Gaussian scene reconstruction from a single forward-facing driving video, with
ADToolbox editing and export extensions applied. Used by the `reconstruction` and
`scene_editor` tools.

## What it does

- Feed-forward dynamic 3D Gaussian reconstruction over an initial frame window
  (the ADToolbox profile uses 40 frames at the source video frame rate).
- Exports static, dynamic and sky point clouds (PLY), camera parameters
  (`camera_params.json`, `ply_camera_params.json`) and rendered videos.
- Instance editing: trajectory offsets, rotation, scaling, removal, asset addition and
  replacement over rigid Gaussian groups, then re-rendering of the edited scene.
- Optional per-frame dynamic PLY export for the simulation scene converter.

## Layout

- `inference.py` — reconstruction / editing / export entry point.
- `gsrecon/` — model package (aggregator, camera/point/track/Gaussian heads, decoder).
- `datasets/`, `utils/`, `train.py` — data pipelines and training entry.
- `third_party/TAPIP3D`, `third_party/cotracker` — vendored tracking utilities.

## Setup

```bash
python scripts/setup_project.py reconstruction   # from the repository root
```

Create a dedicated Python environment from `requirements.txt` (PyTorch/CUDA stack), then
download the model checkpoints you need. Point `reconstruction.root`, `reconstruction.python`,
`reconstruction.checkpoint` and `reconstruction.segmentation_model` at them in your runtime
configuration — see [the reconstruction guide](../../docs/backends/reconstruction.md).

## Attribution

This tree is derived from an upstream Apache-2.0 research project and retains its original
`LICENSE` and `NOTICE` (also under `licenses/reconstruction/` at the repository root). The
NOTICE covers material derived from VGGT (facebookresearch) and separate checkpoint terms.
Vendored tracking utilities keep their own upstream terms; cotracker derives from co-tracker,
whose license is non-commercial. See `THIRD_PARTY_NOTICES.md`. ADToolbox modifications are
marked in the affected source files.
