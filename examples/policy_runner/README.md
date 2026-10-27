# policy_runner

From the repository root:

```bash
adtoolbox run policy_runner --input examples/policy_runner/input.json
```

Replace `/path/to/…` entries with real local inputs. LLM and optional-backend tools also need `--config configs/runtime.local.json` and the relevant environment variables. Examples containing external paths are configuration templates, not completed runs.

See the [tool guide](../../docs/tools/policy_runner.md) for inputs, outputs and limits.
