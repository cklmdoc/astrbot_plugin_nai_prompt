# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 与语义化版本规范。

## [v1.1.0] - 2026-08-13

### 新增
- 最终提示词新增 LLM 语义去重：合并同义词并保持原有顺序与权重语法，失败自动降级不去重。

### 变更
- 移除自动前置的质量词（`masterpiece` / `best quality`）。
- 移除默认负面提示词，负面 Prompt 不再输出。
- 输出排版由框线表格改为极简纯文本：正面提示词 + 一行元信息注释（来源 / NSFW / 角色 / 多角色建议）。
- `max_prompt_length` 默认值由 `1800` 改为 `0`（`0` 表示不截断，可按需配置上限）。
