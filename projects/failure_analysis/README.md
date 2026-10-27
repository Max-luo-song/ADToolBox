# Failure evidence investigation / 失败证据分析

The original token-based evidence pipeline is included: official outcome extraction, image investigation, numeric evidence and final reports.

```bash
python -m venv environments/failure_analysis
environments/failure_analysis/bin/python -m pip install -r projects/failure_analysis/requirements.txt
```

Configure `failure_analysis.python`. Copy `configs/complete.json.example` to a local configuration and supply your actual simulation/result roots. For model credentials, create `projects/failure_analysis/.secrets/vlm.json` from `configs/vlm.json.example` and restrict it with `chmod 600`. Credentials and local configs must not be committed.

Call `failure_analyzer` with `token` and the absolute `config_path`. It invokes the included `run.py` pipeline and returns evidence reports. Calls using a `failure_case` instead of a token still support direct evidence-to-model analysis.

中文：提供 token 会调用此项目的完整证据流程；不是让用户自己实现一个 token 后端。需准备实际的仿真记录、配置和视觉模型密钥。
