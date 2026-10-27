# Validation status / 验证状态

This is an alpha standalone distribution. Interface support and reproduced compute results are recorded separately.

| Area | Evidence | Status |
|---|---|---|
| Portable package | Python 3.12/macOS and Python 3.10/Linux; 20 unittest cases on each covering 14 schemas and independent entries | Passed |
| Native tools | Actual CLI edit compilation, synthetic overlap filtering with JSONL persistence, advisory diagnostics | Passed |
| MCP | Actual stdio initialize, list all 14 tools, successful invocation and explicit tool error | Passed |
| Model transport | Request construction, credential separation and output-schema validation using controlled response fixtures | Passed; no live model-quality measurement |
| Optional backend process contracts | Separate fixture simulation/policy executables, fresh-report checks, fixture training checkpoint, failure/timeout paths | Passed; fixtures do not execute real models |
| Reconstruction extension | Exact upstream revision and per-file hashes in manifest; patch application and Python 3.12 syntax | Passed |
| Default imports | Tool listing does not load PyTorch, OpenCV or detector packages | Passed |
| GPU reconstruction / editing | Requires clean engine installation, weights, input provenance and measured artifacts | Not reproduced |
| Closed-loop metrics | Requires compatible scenes, policy weights and metric/map annotations | Not reproduced |
| Policy fine-tuning quality | Requires valid annotations and pre/post-training evaluation | Not reproduced |

The backend tests generate temporary **fixture reports and fixture checkpoint bytes** solely to verify launcher behavior. They do not validate model inference, checkpoint quality or performance. Example scenario geometry is synthetic. No benchmark improvements or end-to-end autonomous workflow claims are made.

CI runs the portable suite on Python 3.10 and 3.12. A workflow badge links to the actual workflow state; local success does not imply a remote CI run has finished.

Release checks also scan source and built archives for private paths, framework imports, credentials and runtime/model assets, then install the wheel into a clean base environment. These checks reduce accidental inclusion; they are not a substitute for reviewing newly added files.

中文：基础工具与协议的通过记录，不代表 GPU 重建、编辑效果、仿真指标或训练收益已验证。后端测试使用明确的临时替身产物；示例案例使用合成几何。后续真实复现应补充后端版本、环境、输入来源、运行命令和结果产物。
