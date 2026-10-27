# Examples

Run these from the repository root after installation:

```bash
adtoolbox run edit_config_generator --input examples/edit_config_generator/input.json
adtoolbox run failure_case_filter --input examples/failure_case_filter/input.json
adtoolbox run recovery_advisor --input examples/recovery_advisor/input.json
```

These three examples execute without API keys or optional backends. The curation example uses explicitly synthetic geometry; it is not a simulation benchmark. Each of the 14 tools has its own input template and guide. Optional-backend examples need real local data and configuration.
