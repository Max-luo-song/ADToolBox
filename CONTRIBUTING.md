# Contributing

Keep every tool independently callable. Add schemas, examples and a meaningful test when changing behavior. Do not add framework scheduling, global agent prompts, credentials, private data, backend environments or checkpoints.

```bash
python -m pip install -e '.[dev,mcp]'
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
python scripts/check_distribution.py
python -m build
```

Describe the concrete behavior change and what was verified. Record GPU results with backend revisions, environment, input provenance and artifacts; do not label protocol tests as GPU validation.
