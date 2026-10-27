# Where the implementation lives / 实现在哪里

| Tool | Tool-side implementation | Actual compute project |
|---|---|---|
| Failure analysis | `tools/failure_analyzer/tool.py` | `projects/failure_analysis` |
| Requirements / conception / edit planning | Corresponding `tools/<name>/tool.py` contains model prompts and result contracts | Shared model transport only; no model weights needed locally |
| Video retrieval | `tools/video_retriever/tool.py` and `caption_search.py` | `projects/video_search`, complete semantic pipeline |
| Instance tracking | `tools/instance_discovery/tracker.py` | Optional installed detector; real model inference code is present |
| Reconstruction | `tools/reconstruction/service.py` and `preprocess.py` | `projects/reconstruction`, bundled engine source with editing extensions |
| Edit configuration | `tools/edit_config_generator/compiler.py` and `mcp_client.py` | `projects/scene_editing`, complete editing service and model client |
| Scene rendering | `tools/reconstruction/service.py::edit` | Bundled reconstruction renderer |
| Scene validation | `tools/scene_validator/validator.py` | ffmpeg + configured vision endpoint |
| Diagnostics | `tools/recovery_advisor/tool.py` | Standalone rules; no automatic scheduling |
| Policy simulation | `tools/policy_runner/launcher.py` and `closed_loop.py` | `projects/simulation/engine` and `projects/simulation/policy` |
| Failure curation | `tools/failure_case_filter/evaluator.py` | Standalone geometry and JSONL persistence |
| Fine-tuning | `tools/policy_trainer/trainer.py` | `projects/simulation/policy` |
| Scene conversion | `scripts/convert_scene.py` | `projects/simulation/scripts/stage_scene.py` + `build_dynamic_scene.py` |
| Training data export | — | `projects/simulation/scripts/export_policy_training_data.py` |

Paths in the middle column are relative to `src/adtoolbox/`. Every tool has its own
implementation directory. Heavy models keep their own environments. Engine source is bundled;
checkpoints, datasets and private Agent scheduling are not.

The scene conversion chain handles edited target tracks with explicit timing and is a constrained
converter, not a general road/map or training-label generator.
