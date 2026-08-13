# NAI 提示词助手

AstrBot 插件：将 `/提示词 <自然语言描述>` 转成可复制的 NAI 正面提示词。

## 安装

将整个 `astrbot_plugin_nai_prompt` 目录复制到 AstrBot 的 `data/plugins/`，在 WebUI 插件页加载或重载。AstrBot 会按 `requirements.txt` 安装 `aiohttp`。

首次使用前，在插件配置填写 `allowed_user_ids`；为空时仅 AstrBot 管理员可用。

## 使用

```text
/提示词 绿色双马尾猫耳少女，穿校服，在樱花树下微笑
/提示词 流萤和橘望在花园里拥抱
/提示词 白裙子长头发 + 图片   # 反推图片并可按描述微调
/提示词 + 图片                # 仅反推图片
```

图片输入时：命令后附带图片，或在文字中附图片链接即可。插件先经 WD14 Tagger 反推标签，再由 LLM 整理后进入同一流程。

流程：当前会话 LLM 按 JSON 规则转译（图片为 Tagger 标签整理）→ DanbooruSearch 语义搜索角色 → 关联 General 特征增强 → LLM 语义去重 → 最终 Prompt 组装。

DanbooruSearch 默认 API：`https://sakizuki-danboorusearch.hf.space/api`。若不可用或角色匹配分数不足，插件仍输出 LLM 转译结果。

图反推默认 API：`https://smilingwolf-wd-tagger.hf.space`（Gradio 新协议/3.x 协议均可），可在配置中替换为自建镜像。

输出为极简纯文本：正面提示词 + 一行元信息注释（来源 / NSFW / 角色 / 多角色建议）。默认不截断（`max_prompt_length` 设为 0），可按需配置上限。

成人向仅在 `allow_adult_prompts=true` 且解析出 `explicit` 等级时启用；其它情况均生成全年龄版本。
