<div align="center">

![ADToolbox — Driving Scenario Engineering](assets/brand/hero.svg)

**Independent tools for driving scenario engineering.**

[English](README.md) · [简体中文](README.zh-CN.md) · [Tool reference](docs/tools/README.md) · [Examples](examples/README.md)

[![CI](https://github.com/Max-luo-song/tool_box/actions/workflows/ci.yml/badge.svg)](https://github.com/Max-luo-song/tool_box/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3b726b)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-334a50)](LICENSE)

</div>

### Team

ADToolbox is developed by a joint team from **East China Normal University (ECNU)** and the **Shanghai Innovation Institute (SII)**.

<p align="center">
  <img src="assets/affiliations/east-china-normal-university.png" alt="East China Normal University" width="43%" />
  &nbsp;&nbsp;&nbsp;
  <img src="assets/affiliations/shanghai-innovation-institute.png" alt="Shanghai Innovation Institute" width="45%" />
</p>

<p align="center">
  Luosong Guo · Haochun Gao · Xiang Yang · Dinghao LIn · Yi Luo<br />
  Zhenyu Hou · Ao Gao · Chaoyang Li · Jingyu Gong<sup>†</sup> · Yuan Xie<sup>†</sup>
</p>

<p align="center"><sup>†</sup> Project leaders.</p>

Our work focuses on reusable tools for driving video retrieval, dynamic 3D reconstruction, scene editing, and policy evaluation, making these capabilities easier to use independently and integrate into research workflows.

ADToolbox brings failure evidence, scenario design, video retrieval, dynamic reconstruction, editing and policy evaluation behind a consistent interface. Use one tool from Python, call it from a terminal, or expose it to an MCP client. Each call has explicit inputs and inspectable outputs.

The library includes **14 independent tools with real implementations**, and the heavy compute engines ship with the repository: a dynamic Gaussian reconstruction engine with editing extensions, a closed-loop simulation engine, and a policy training stack live under [`projects/`](projects/README.md). Orchestration and next-step selection belong to your own application — ADToolbox deliberately does not include an agent framework.

### Start in one minute

```bash
git clone https://github.com/Max-luo-song/tool_box.git
cd tool_box
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

adtoolbox list
adtoolbox run edit_config_generator --input examples/edit_config_generator/input.json
adtoolbox run failure_case_filter --input examples/failure_case_filter/input.json
```

These examples compile a real edit document and screen synthetic box geometry. They need **no API key, GPU or model download**. The curation example also writes a local JSONL case entry. External-data examples use explicit path templates.

```python
from adtoolbox.tools.edit_config_generator import run

result = run({
    "coordinate_system": "camera_local_meters",
    "operations": [{"operation": "remove", "instance_id": 0}],
})
assert result["status"] == "success"
print(result["result"]["edit_config"])
```

### The tool collection

| Tool | Capability | Runtime |
|---|---|---|
| [`failure_analyzer`](docs/tools/failure_analyzer.md) | Failure evidence analysis | llm |
| [`scene_requirement_generator`](docs/tools/scene_requirement_generator.md) | Scenario requirements | llm |
| [`scene_conceiver`](docs/tools/scene_conceiver.md) | Scenario conception | llm |
| [`video_retriever`](docs/tools/video_retriever.md) | Driving video retrieval | native + engine |
| [`instance_discovery`](docs/tools/instance_discovery.md) | Traffic instance tracking | optional vision / engine |
| [`reconstruction`](docs/tools/reconstruction.md) | Dynamic scene reconstruction | bundled engine |
| [`edit_plan_builder`](docs/tools/edit_plan_builder.md) | Scene edit planning | llm |
| [`edit_config_generator`](docs/tools/edit_config_generator.md) | Edit configuration compiler | native + llm |
| [`scene_editor`](docs/tools/scene_editor.md) | Scene editing and rendering | bundled engine |
| [`scene_validator`](docs/tools/scene_validator.md) | Scene consistency checks | native + optional llm |
| [`recovery_advisor`](docs/tools/recovery_advisor.md) | Execution diagnostics | native |
| [`policy_runner`](docs/tools/policy_runner.md) | Closed-loop policy evaluation | bundled engine |
| [`failure_case_filter`](docs/tools/failure_case_filter.md) | Failure case curation | native |
| [`policy_trainer`](docs/tools/policy_trainer.md) | Supervised policy fine-tuning | bundled engine |

**Native** tools run with the base package. **LLM** tools require your configured model endpoint. **Bundled engine** tools drive real engine source included under `projects/`; you still prepare the engine's Python environment, model weights and data. Native video retrieval ranks supplied captions; the bundled `projects/video_search` pipeline provides full semantic indexing and retrieval.

### Designed to be composed

- **One contract.** Versioned JSON results, typed inputs, explicit artifacts and useful errors.
- **Independent environments.** Reconstruction, tracking, simulation and training can use separate runtimes.
- **Portable scene edits.** Trajectory, rotation, scale, removal, addition and replacement share one functional interface.
- **Evidence with limits.** Visual plausibility, geometric screening and official policy metrics remain distinguishable.

### Prepare the engines

Configure only what you use. Start with [runtime configuration](docs/configuration.md), then follow the [engine guides](docs/backends/README.md). Verify a bundled engine tree and print its setup checklist:

```bash
python scripts/setup_project.py reconstruction
python scripts/setup_project.py simulation
```

An edited reconstruction can be converted into a simulation scene with `scripts/convert_scene.py`, and a simulated rollout can be exported as supervised fine-tuning data with `projects/simulation/scripts/export_policy_training_data.py`. Existing integrations can also use the [JSON subprocess bridge](docs/configuration.md#external-json-backend).

For model tools, export `ADTOOLBOX_API_KEY`, `ADTOOLBOX_BASE_URL` and `ADTOOLBOX_MODEL`. Text evidence and explicitly attached frames are sent to that endpoint. The library does not automatically load `.env` files.

```bash
python -m pip install -e '.[mcp]'
adtoolbox doctor --config configs/runtime.local.json
adtoolbox-mcp --config configs/runtime.local.json
```

See [MCP setup](docs/mcp.md), [coordinate and data contracts](docs/contracts.md), and [per-tool examples](examples/README.md).

### Release scope

This is an **alpha toolkit release**. Portable implementations, model transport and process/protocol contracts are covered by tests. The bundled engines are verified for source completeness and integration wiring; clean GPU reconstruction, edited-scene fidelity, closed-loop metrics and policy training outcomes depend on your hardware, checkpoints and data. [Validation status](docs/validation.md) records the distinction.

Box-overlap screening does not establish collision responsibility or road compliance. Sampled-frame review does not certify physics. Fine-tuning requires valid labels; a mined failure is not automatically a training example.

### Contribute

See [contributing](CONTRIBUTING.md) and [security](SECURITY.md). Useful next steps are reproducible engine profiles, measured editing examples and additional portable scene adapters.

### License and acknowledgments

Standalone toolkit code is licensed under [Apache-2.0](LICENSE). The bundled engines under `projects/` are derived from Apache-2.0 upstream projects and retain their original licenses and attribution; see [NOTICE](NOTICE), [third-party notices](THIRD_PARTY_NOTICES.md) and the engine guides. Model weights and datasets are not distributed.
