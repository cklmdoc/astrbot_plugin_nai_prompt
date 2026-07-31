# NAI 提示词助手

AstrBot 插件：将 `/提示词 <自然语言描述>` 转成可复制的 NAI 正面 / 负面提示词。

## 安装

将整个 `astrbot_plugin_nai_prompt` 目录复制到 AstrBot 的 `data/plugins/`，在 WebUI 插件页加载或重载。AstrBot 会按 `requirements.txt` 安装 `aiohttp`。

首次使用前，在插件配置填写 `allowed_user_ids`；为空时仅 AstrBot 管理员可用。

## 使用

```text
/提示词 绿色双马尾猫耳少女，穿校服，在樱花树下微笑
/提示词 流萤和橘望在花园里拥抱
```

流程：当前会话 LLM 按 JSON 规则转译 → DanbooruSearch 语义搜索角色 → 关联 General 特征增强 → 最终 Prompt 组装。

DanbooruSearch 默认 API：`https://sakizuki-danboorusearch.hf.space/api`。若不可用或角色匹配分数不足，插件仍输出 LLM 转译结果。

成人向仅在 LLM 标记角色为 `adult` 且 `allow_adult_prompts=true` 时启用；其它情况均生成全年龄版本。
