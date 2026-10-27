# Real-project migration in progress / 真实项目迁移中

This commit saves work in progress. It moves the tool implementations into individual tool packages and includes real editing, semantic video-search and failure-analysis source projects. Reconstruction and simulation/training are now pinned upstream Git submodules, with local integration extensions.

The simulation extension includes the previously missing common scene-conversion implementation. `scripts/setup_project.py` and `scripts/convert_scene.py` are included with the migration.

Remaining work: finish wiring and documenting all migrated project paths; update packaging and release-boundary checks for submodules; reconcile older README/tool descriptions with the new layout; check the migrated integrations. Earlier validation results apply to the previous release, not this unfinished migration. No new GPU run is claimed.

To obtain the real model projects from Git:

```bash
git submodule update --init --recursive
```

中文：本提交先保存当前进度。真实的编辑、视频语义检索和失败分析组件已经迁入；重建、仿真与训练项目通过固定版本子模块提供。各工具的实际实现已移到自己的目录。仍需补齐接线、文档、打包与子模块检查规则；旧版本的测试通过记录不代表此次迁移已验证。此次未重新执行 GPU 任务。
