# edit_plan_builder

From the repository root:

```bash
tool-box run edit_plan_builder --input examples/edit_plan_builder/input.json
```

Replace `/path/to/…` entries with real local inputs. LLM and optional-backend tools also need `--config configs/runtime.local.json` and the relevant environment variables. Examples containing external paths are configuration templates, not completed runs.

See the [tool guide](../../docs/tools/edit_plan_builder.md) for inputs, outputs and limits.
