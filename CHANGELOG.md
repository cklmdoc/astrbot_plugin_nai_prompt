# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 与语义化版本规范。

## [v1.4.0] - 2026-08-14

### 新增
- 切换到 NAI 新版格式：角色用 `人物名(作品名)` 引用（DanbooruSearch 组合 Character canonical + Copyright 作品名）。
- 多角色 Character Prompting：用 `|` 分隔 base prompt 与各角色 prompt，最多 6 人，人数标签在 base、各角色用 `girl`/`boy`/`other`，位置以自然语言强化。
- 互动标签：`source#动作` / `target#动作` / `mutual#动作`。
- 渲染文字（`Text:` / `no text`）与情绪词支持。

### 变更
- 权重语法改为新版 `权重::标签::`（weak→`0.8::tag::`、strong→`1.2::tag::`、very_strong→`1.5::tag::`）。
- 单角色输出改为平铺 `人物名(作品名), 1girl, solo, 特征...`。
- 移除旧版 `(tag:1.2)` / `[tag]` / `{tag}` 权重语法与多角色"请自行拆分"提示。

## [v1.3.0] - 2026-08-14

### 新增

- 支持生成示例图：通过调用NAI生图插件的 OpenAI Images API 兼容本地生图服务，在返回提示词后按配置生成并回传示例图。
- 新增配置 `enable_image_generation`（默认关）、`image_count`（默认 1）、`image_api_url`（默认 `http://127.0.0.1:8765`）。

### 变更

- 生图失败静默降级为仅返回文本提示词，不影响提示词输出。

## \[v1.2.0] - 2026-08-14

### 新增

- 支持自然语言权重/强调控制：识别"突出/强调/重点/弱化/淡化"等意图词，输出 NAI 原生权重语法（strong → `(tag:1.2)`、very\_strong → `(tag:1.4)`、weak → `[tag]`）。
- 图片反推新增主动优化：自动加权核心特征并补全风格/构图/光照/景别缺失标签，可通过 `image_auto_optimize` 开关关闭。

### 变更

- 权重语法由自定义 `1.2::tag::` 改为 NAI 原生括号 `(tag:1.2)` / `[tag]`。
- 标签组装在语义去重前先去除完全重复并套用权重语法。

## \[v1.1.0] - 2026-08-13

### 新增

- 最终提示词新增 LLM 语义去重：合并同义词并保持原有顺序与权重语法，失败自动降级不去重。

### 变更

- 移除自动前置的质量词（`masterpiece` / `best quality`）。
- 移除默认负面提示词，负面 Prompt 不再输出。
- 输出排版由框线表格改为极简纯文本：正面提示词 + 一行元信息注释（来源 / NSFW / 角色 / 多角色建议）。
- `max_prompt_length` 默认值由 `1800` 改为 `0`（`0` 表示不截断，可按需配置上限）。

