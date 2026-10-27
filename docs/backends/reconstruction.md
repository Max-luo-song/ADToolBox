# Dynamic reconstruction profile

Install the external [DGGT project](https://github.com/xiaomi-research/dggt) and its documented PyTorch/CUDA dependencies. Use the exact base revision recorded in `extensions/reconstruction/manifest.json` before applying the provided patch. Follow `extensions/README.md`; do not apply blindly to another revision or a dirty checkout.

The patch provides the adapted single-view inference/export/editor helpers used by this profile, including camera/PLY export, dynamic frame export, rigid edits and optional camera offsets. It is distributed as a patch, not a vendored engine. Upstream checkpoints and subdependencies keep their own terms; see [third-party notices](../../THIRD_PARTY_NOTICES.md).

Configure `reconstruction.root`, `python`, `preprocess_python`, `checkpoint`, `segmentation_model` and optional `device`, `timeout`, `output_dir`. The preprocessing environment needs OpenCV, NumPy, PyTorch and Transformers; segmentation_model is a local SegFormer model/processor directory. The bundled preprocessing helper extracts the requested initial frame range and records its actual FPS.

```bash
tool-box run reconstruction --input examples/reconstruction/input.json --config configs/runtime.local.json
```

To edit, first discover instances over the **same reconstructed input frames**, with `pixel_space=reconstruction_input`. Save that result as `scene_editing.instance_info_path`. Compile a portable edit document, then call `scene_editor` using the source video, matching frame count, poses and a calibrated `meters_to_scene_units`. Existing-instance identities and boxes are validated before launching rendering. An add/replace asset must have an explicit scale; compatible asset geometry is your responsibility.

Supported operations: trajectory offsets, XYZ rotation, scaling, removal, addition and replacement. Native render-state export is restricted to unedited reconstruction. The interface rejects missing extensions before starting GPU inference. Parameterizing frame count does not guarantee every sequence length fits GPU memory or the model's supported window.

This release checks patch applicability and Python syntax. GPU reconstruction and edit fidelity have not been reproduced in a clean backend environment.
