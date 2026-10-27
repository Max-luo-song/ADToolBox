# failure_analyzer

From the repository root:

```bash
tool-box run failure_analyzer --input examples/failure_analyzer/input.json
```

Replace `/path/to/…` entries with real local inputs. LLM and optional-backend tools also need `--config configs/runtime.local.json` and the relevant environment variables. Examples containing external paths are configuration templates, not completed runs.

See the [tool guide](../../docs/tools/failure_analyzer.md) for inputs, outputs and limits.
