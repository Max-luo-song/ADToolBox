# Dynamic reconstruction guide

The reconstruction engine is bundled under `projects/reconstruction` (single-view dynamic Gaussian
reconstruction, Mode 2 editing extensions included). Verify the tree and print the setup checklist:

```bash
python scripts/setup_project.py reconstruction
```

Create a dedicated Python environment from `projects/reconstruction/requirements.txt` (PyTorch,
CUDA and rendering dependencies are listed there), then download the model checkpoints described
in `projects/reconstruction/README.md`. Checkpoints and subdependencies keep their own terms; see
[third-party notices](../../THIRD_PARTY_NOTICES.md).

Configure `reconstruction.root`, `python`, `preprocess_python`, `checkpoint`,
`segmentation_model` and optional `device`, `timeout`, `output_dir`. The preprocessing
environment needs OpenCV, NumPy, PyTorch and Transformers; `segmentation_model` is a local
SegFormer model/processor directory. The bundled preprocessing helper extracts the requested
initial frame range and records its actual FPS.

```bash
adtoolbox run reconstruction --input examples/reconstruction/input.json --config configs/runtime.local.json
```

To edit, first discover instances over the **same reconstructed input frames**, with
`pixel_space=reconstruction_input`. Save that result as `scene_editing.instance_info_path`.
Compile a portable edit document, then call `scene_editor` using the source video, matching frame
count, poses and a calibrated `meters_to_scene_units`. Existing-instance identities and boxes are
validated before launching rendering. An add/replace asset must have an explicit scale;
compatible asset geometry is your responsibility.

Supported operations: trajectory offsets, XYZ rotation, scaling, removal, addition and
replacement. Native render-state export is restricted to unedited reconstruction. The interface
checks engine completeness before starting GPU inference. Parameterizing frame count does not
guarantee every sequence length fits GPU memory or the model's supported window.

GPU reconstruction and edit fidelity depend on your hardware, weights and data; see
[validation status](../validation.md).
