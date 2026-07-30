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

流程：当前会话 LLM 解析严格 JSON → Danbooru 验证角色标签 → 可选 SeaArt 公开检索 → 本地规则兜底。

成人向仅在用户明确请求且 `allow_adult_prompts=true` 时生成；疑似未成年人描述始终降级为全年龄。
