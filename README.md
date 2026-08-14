# NAI 提示词助手

AstrBot 插件：将 `/提示词 <自然语言描述>` 转成可复制的 NAI 正面提示词。

## 安装

将整个 `astrbot_plugin_nai_prompt` 目录复制到 AstrBot 的 `data/plugins/`，在 WebUI 插件页加载或重载。AstrBot 会按 `requirements.txt` 安装 `aiohttp`。

默认所有用户均可使用。如需限制，将 `enable_whitelist` 设为 `true` 并在 `allowed_user_ids` 填写允许的用户 ID；开启后仅管理员及白名单内用户可用。

## 使用

```text
/提示词 绿色双马尾猫耳少女，穿校服，在樱花树下微笑
/提示词 流萤和橘望在花园里拥抱，流萤在左边   # 多角色互动 + 位置
/提示词 突出红色长发，弱化背景，穿校服   # 强调/弱化标签权重
/提示词 白裙子长头发 + 图片   # 反推图片并可按描述微调
/提示词 + 图片                # 仅反推图片
```

图片输入时：命令后附带图片，或在文字中附图片链接即可。插件先经 WD14 Tagger 反推标签，再由 LLM 整理后进入同一流程。

流程：当前会话 LLM 按 JSON 规则转译（图片为 Tagger 标签整理）→ DanbooruSearch 语义搜索角色与作品名 → 关联 General 特征增强 → 套用权重语法 → 多角色 Character Prompting 组装 → LLM 语义去重 → 最终 Prompt。

DanbooruSearch 默认 API：`https://sakizuki-danboorusearch.hf.space/api`。若不可用或角色匹配分数不足，插件仍输出 LLM 转译结果。

图反推默认 API：`https://smilingwolf-wd-tagger.hf.space`（Gradio 新协议/3.x 协议均可），可在配置中替换为自建镜像。

## 权重与强调

用自然语言表达强调/弱化，插件会输出 NAI 新版 `权重::标签::` 语法：

- 突出/强调/重点 → `1.2::tag::`
- 非常/极其/强烈 → `1.5::tag::`
- 弱化/淡化/忽略 → `0.8::tag::`

图片反推时默认开启主动优化（`image_auto_optimize`，默认 `true`）：自动给核心特征标签加权，并补全反推不出的风格/构图/光照/景别缺失标签；关闭后仅忠实整理反推标签。

## 新版格式

- 角色：`人物名(作品名)` 引用（如 `texas the omertosa (arknights)`），由 DanbooruSearch 组合 Character canonical 与 Copyright 作品名。
- 多角色：`{人物 [tags], {位置} 人物}` 包裹 + `{人物 与 人物}` 连接，最多 6 人，位置由 LLM 判断。
- 互动：`source#动作` / `target#动作` / `mutual#动作`。
- 文字与情绪：`Text: 内容` / `no text`，可加入情绪词增强表现力。

## 示例图生成

开启 `enable_image_generation`（默认 `false`）后，插件会在返回提示词后调用本地生图服务生成示例图，并按 `image_count`（默认 1）回传指定张数。

生图服务为[NAI生图插件](https://github.com/woakato/astrbot_plugin_nai_image)的 OpenAI Images API 兼容接口（默认 `http://127.0.0.1:8765`，即 AstrBot 本地生图插件），API Key 与模型名由插件内置占位符填充、服务不校验；生图失败时静默降级为仅返回文本提示词。

输出为极简纯文本：正面提示词 + 一行元信息注释（来源 / NSFW / 角色）。默认不截断（`max_prompt_length` 设为 0），可按需配置上限。

成人向仅在 `allow_adult_prompts=true` 且解析出 `explicit` 等级时启用；其它情况均生成全年龄版本。
