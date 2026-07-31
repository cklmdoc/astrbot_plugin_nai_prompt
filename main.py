"""AstrBot NAI prompt helper plugin."""

from __future__ import annotations

import asyncio
import json
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig

from .prompt_engine import (
    DANBOORU_SEARCH_API_DEFAULT,
    CharacterTags,
    DanbooruSearchLookup,
    ParsedRequest,
    build_prompt,
    extract_json,
    format_result,
    parse_llm_response,
    resolve_nsfw_level,
)

HELP_TEXT = """用法：/提示词 <自然语言描述>

示例：
/提示词 流萤穿校服，在樱花树下微笑"""
MAX_DESCRIPTION_LENGTH = 500

LLM_SYSTEM_PROMPT = """你是 NAI 标签提示词解析器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。
将用户自然语言转换为紧凑、英文小写 NAI/Danbooru 标签数据。

JSON schema:
{
 "characters": [
   {
     "display_name": "角色中文或常用名；原创角色填空字符串",
     "danbooru_tag": "留空字符串；角色标签由插件查询服务确定",
     "tags": ["该角色本次明确指定的专属外观标签"]
   }
 ],
 "shared_tags": ["人数和共享主体标签，如 1girl"],
 "outfit_tags": ["用户明确指定的服装"],
 "action_tags": ["动作和互动"],
 "scene_tags": ["场景、道具、明确光照或时间"],
 "style_tags": ["少量风格标签，可用 1.2::key_tag:: 强调关键元素"],
 "nsfw_level": "safe|suggestive|explicit"
}

规则：
- 角色仅在明确提及既有角色时填写 display_name；原创 OC 的 characters 可为空数组。
- danbooru_tag 必须留空字符串，插件会使用 DanbooruSearch 查询 canonical 角色标签。
- 普通标签必须英文小写下划线；不要写完整句子。
- 不得添加用户未明确描述的服装、道具、天气、光照、时间或外观。
- 不得输出 masterpiece、best_quality、画师标签、负面词、尺寸或比例词；插件会统一处理。
- 不要堆叠同义词；每个概念只保留一个最准确标签。
- 未提及的数组必须返回空数组。"""

FORMAT_RETRY_PROMPT = """上一次输出不符合指定 JSON schema。
请只返回完整、合法的 JSON 对象，不要 Markdown、解释或其它文字。
必须包含 characters、shared_tags、outfit_tags、action_tags、scene_tags、style_tags、nsfw_level；
characters 的每一项必须包含 display_name、danbooru_tag、tags。"""

CONFLICT_FILTER_SYSTEM_PROMPT = """你是标签去冲突器。用户明确指定了服装和/或动作标签，你需要从角色的关联标签中移除与用户指定标签语义冲突的标签。

只返回一个合法 JSON 对象，不要 Markdown、解释或额外文字。

JSON schema:
{
  "filtered_tags": ["过滤后的标签列表"]
}

规则：
- 保留角色 canonical 标签（通常为第一个标签）
- 移除与用户指定服装标签语义冲突的标签（如用户指定 swimsuit，则移除 school_uniform、dress 等服装标签）
- 移除与用户指定动作标签语义冲突的标签（如用户指定 running，则移除 standing、sitting 等动作/姿态标签）
- 保留不冲突的特征标签（如发色、瞳色、体型等）"""


class NaiPromptPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.config = config or {}
        self.lookup: DanbooruSearchLookup | None = None
        self._last_request: dict[str, float] = {}

    async def initialize(self) -> None:
        self.lookup = DanbooruSearchLookup(
            api_url=self._cfg_str("danbooru_search_api_url", DANBOORU_SEARCH_API_DEFAULT),
            timeout_seconds=self._cfg_int("request_timeout_seconds", 5, 2, 15),
        )
        logger.info("[NAIPrompt] 插件已加载")

    async def terminate(self) -> None:
        self._last_request.clear()
        self.lookup = None
        logger.info("[NAIPrompt] 插件已停止")

    def _cfg_str(self, key: str, default: str = "") -> str:
        value = self.config.get(key, default) if hasattr(self.config, "get") else default
        return str(value).strip() if value is not None else default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default) if hasattr(self.config, "get") else default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _cfg_int(self, key: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(self.config.get(key, default))
        except (TypeError, ValueError, AttributeError):
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _sender_id(event: AstrMessageEvent) -> str:
        getter = getattr(event, "get_sender_id", None)
        return str(getter() if callable(getter) else "")

    @staticmethod
    def _is_admin(event: AstrMessageEvent) -> bool:
        for attr in ("is_admin", "is_admin_user"):
            value = getattr(event, attr, False)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = False
            if value is True:
                return True
        return False

    def _allowed(self, event: AstrMessageEvent) -> bool:
        if self._is_admin(event):
            return True
        raw_ids = self.config.get("allowed_user_ids", []) if hasattr(self.config, "get") else []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        allowed = {str(item).strip() for item in raw_ids if str(item).strip()} if isinstance(raw_ids, list) else set()
        return self._sender_id(event) in allowed

    def _cooldown_message(self, sender_id: str) -> str | None:
        cooldown = self._cfg_int("cooldown_seconds", 5, 0, 60)
        if cooldown == 0:
            return None
        now = time.monotonic()
        previous = self._last_request.get(sender_id)
        if previous is not None and now - previous < cooldown:
            remaining = max(1, int(cooldown - (now - previous)))
            return f"请求过于频繁，请 {remaining} 秒后再试。"
        self._last_request[sender_id] = now
        return None

    async def _parse_with_llm(self, event: AstrMessageEvent, description: str) -> ParsedRequest | None:
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
            if not provider:
                return None
            response = await provider.text_chat(prompt=description, contexts=[], system_prompt=LLM_SYSTEM_PROMPT)
            parsed = parse_llm_response(getattr(response, "completion_text", "") if response else "")
            if parsed is not None:
                return parsed
            logger.warning("[NAIPrompt] LLM 首次输出不符合 JSON schema，执行一次格式重试")
            retry = await provider.text_chat(
                prompt=f"{description}\n\n{FORMAT_RETRY_PROMPT}", contexts=[], system_prompt=LLM_SYSTEM_PROMPT,
            )
            return parse_llm_response(getattr(retry, "completion_text", "") if retry else "")
        except Exception as exc:
            logger.warning("[NAIPrompt] LLM 解析失败: %s", exc)
            return None

    async def _lookup_characters(self, parsed: ParsedRequest):
        if self.lookup is None:
            return []
        final_level = resolve_nsfw_level(parsed, self._cfg_bool("allow_adult_prompts", True))
        show_nsfw = final_level == "explicit"
        results = await asyncio.gather(*(self.lookup.lookup(item, show_nsfw) for item in parsed.characters[:20]))
        return [result for result in results if result is not None]

    async def _filter_conflicting_tags(self, event: AstrMessageEvent, parsed: ParsedRequest, lookup_results: list[CharacterTags]) -> list[CharacterTags]:
        """通过 LLM 过滤角色关联标签中与用户指定服装/动作冲突的默认标签。

        仅当用户明确指定了 outfit_tags 或 action_tags 时触发。
        按角色分别调用 LLM，并行执行；失败时降级为仅保留 canonical tag。

        Args:
            event: 消息事件
            parsed: LLM 解析结果，包含 outfit_tags 和 action_tags
            lookup_results: DanbooruSearch 查询结果列表

        Returns:
            过滤后的 CharacterTags 列表
        """
        if not parsed.outfit_tags and not parsed.action_tags:
            return lookup_results

        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if not provider:
            logger.warning("[NAIPrompt] 无可用 LLM provider，标签冲突过滤降级")
            return [CharacterTags(r.display_name, r.canonical_tag, [r.canonical_tag]) for r in lookup_results]

        async def filter_one(result: CharacterTags) -> CharacterTags:
            prompt = json.dumps({
                "outfit_tags": parsed.outfit_tags,
                "action_tags": parsed.action_tags,
                "character_tags": result.tags,
            }, ensure_ascii=False)
            try:
                response = await provider.text_chat(prompt=prompt, contexts=[], system_prompt=CONFLICT_FILTER_SYSTEM_PROMPT)
                text = getattr(response, "completion_text", "") if response else ""
                data = extract_json(text)
                if data and isinstance(data.get("filtered_tags"), list):
                    filtered = [str(t) for t in data["filtered_tags"] if isinstance(t, str)]
                    if filtered:
                        logger.info("[NAIPrompt] 角色 %s 标签冲突过滤完成: %d -> %d", result.display_name, len(result.tags), len(filtered))
                        return CharacterTags(result.display_name, result.canonical_tag, filtered)
            except Exception as exc:
                logger.warning("[NAIPrompt] 角色 %s 标签冲突过滤异常: %s", result.display_name, exc)
            # 降级：只保留 canonical tag
            return CharacterTags(result.display_name, result.canonical_tag, [result.canonical_tag])

        return list(await asyncio.gather(*(filter_one(r) for r in lookup_results)))

    @filter.command("提示词")
    async def prompt_command(self, event: AstrMessageEvent, description: str = ""):
        """/提示词 <自然语言描述>：生成可复制的 NAI 正负面提示词。"""
        description = str(description or "").strip()
        if not description:
            yield event.plain_result(HELP_TEXT)
            return
        if len(description) > MAX_DESCRIPTION_LENGTH:
            yield event.plain_result("描述过长，请精简至 500 字以内。")
            return
        if not self._allowed(event):
            yield event.plain_result("当前用户未获提示词功能授权，请联系管理员。")
            return
        throttled = self._cooldown_message(self._sender_id(event))
        if throttled:
            yield event.plain_result(throttled)
            return

        parsed = await self._parse_with_llm(event, description)
        if parsed is None:
            yield event.plain_result("提示词解析失败，请稍后重试。")
            return
        characters = await self._lookup_characters(parsed)
        characters = await self._filter_conflicting_tags(event, parsed, characters)
        result = build_prompt(
            parsed=parsed,
            lookup_results=characters,
            allow_adult=self._cfg_bool("allow_adult_prompts", True),
            max_length=self._cfg_int("max_prompt_length", 1800, 200, 5000),
        )
        yield event.plain_result(format_result(result))
