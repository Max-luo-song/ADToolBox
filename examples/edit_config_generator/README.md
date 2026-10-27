# edit_config_generator

From the repository root:

```bash
adtoolbox run edit_config_generator --input examples/edit_config_generator/input.json
```

Replace `/path/to/…` entries with real local inputs. LLM and optional-backend tools also need `--config configs/runtime.local.json` and the relevant environment variables. Examples containing external paths are configuration templates, not completed runs.

See the [tool guide](../../docs/tools/edit_config_generator.md) for inputs, outputs and limits.
