# Where the implementation lives / 实现在哪里

| Tool | Tool-side implementation | Actual compute project |
|---|---|---|
| Failure analysis | `tools/failure_analyzer/tool.py` | `projects/failure_analysis` |
| Requirements / conception / edit planning | Corresponding `tools/<name>/tool.py` contains model prompts and result contracts | Shared model transport only; no model weights needed locally |
| Video retrieval | `tools/video_retriever/tool.py` and `caption_search.py` | `projects/video_search`, complete semantic pipeline |
| Instance tracking | `tools/instance_discovery/tracker.py` | Optional installed detector; real model inference code is present |
| Reconstruction | `tools/reconstruction/service.py` and `preprocess.py` | `projects/reconstruction`, pinned model source plus extension patch |
| Edit configuration | `tools/edit_config_generator/compiler.py` and `mcp_client.py` | `projects/scene_editing`, complete editing service and model client |
| Scene rendering | `tools/reconstruction/service.py::edit` | Patched reconstruction renderer |
| Scene validation | `tools/scene_validator/validator.py` | ffmpeg + configured vision endpoint |
| Diagnostics | `tools/recovery_advisor/tool.py` | Standalone rules; no automatic scheduling |
| Policy simulation | `tools/policy_runner/launcher.py` and `closed_loop.py` | `projects/simulation/projects/SimEngine` and `AlgEngine` |
| Failure curation | `tools/failure_case_filter/evaluator.py` | Standalone geometry and JSONL persistence |
| Fine-tuning | `tools/policy_trainer/trainer.py` | `projects/simulation/projects/AlgEngine` |

Paths in the middle column are relative to `src/driving_toolbox/`. Every tool has its own implementation directory; the old central implementation facade has been removed. Heavy models keep their own environments. Public source projects and necessary extensions are provided; checkpoints, datasets and private Agent scheduling are not.

This change ports actual component implementations. It does not claim a new full GPU reproduction. Existing scene conversion handles one edited target track and is a constrained converter, not a general road/map or training-label generator.
