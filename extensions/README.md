# Standalone backend extensions

The public library does not vendor complete engines. The reconstruction profile ships a **version-pinned patch** against DGGT revision `a3276d2bbe4cbb03bcc117830b1836110a27adeb` and retains its original LICENSE and NOTICE under `licenses/reconstruction/`.

```bash
git clone https://github.com/xiaomi-research/dggt.git /path/to/reconstruction-backend
git -C /path/to/reconstruction-backend checkout a3276d2bbe4cbb03bcc117830b1836110a27adeb
python extensions/reconstruction/apply_extensions.py --backend /path/to/reconstruction-backend
python extensions/reconstruction/apply_extensions.py --backend /path/to/reconstruction-backend --apply
```

Install engine dependencies and weights according to the upstream instructions before running TOOLBOX. The installer verifies both revision and source hashes, then checks patch applicability. It refuses a different or already modified target. The patch adds the existing reconstruction/export/edit extensions and adapted dataset/renderer helpers. It does not include pretrained weights or the full backend.

Patch application and Python syntax are verified separately from GPU behavior. A successful dry run does not validate numerical correctness or editing fidelity. The current patch is an initial port of the source project's backend extensions; clean GPU reproduction is still pending.

`scene_conversion/stage_scene.py` stages edited target Gaussians for an external scene builder. It requires either source timestamps or an explicit fallback source FPS. It does not generate a road map, collision-responsibility annotations or training labels.

中文：扩展以固定版本补丁和独立脚本形式分发，不复制整个第三方项目。先检查再手动应用，不会改动原有项目。补丁安装成功不代表 GPU 效果已验证。
