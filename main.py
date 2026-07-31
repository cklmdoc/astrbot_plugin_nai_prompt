"""AstrBot NAI prompt helper plugin."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig

from .prompt_engine import (
    ParsedRequest,
    TagLookup,
    build_prompt,
    format_result,
    parse_llm_response,
)

HELP_TEXT = """用法：/提示词 <自然语言描述>

示例：
/提示词 流萤穿校服，在樱花树下微笑"""
MAX_DESCRIPTION_LENGTH = 500

LLM_SYSTEM_PROMPT = """你是 NAI 标签提示词解析器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。
将用户的自然语言转换为紧凑、逗号分隔所需的英文 NAI/Danbooru 标签数据。

JSON schema:
{
 "characters": [
   {
     "display_name": "角色中文或常用名",
     "danbooru_tag": "canonical_character_tag_",
     "tags": ["该角色本次明确指定的专属外观标签"]
   }
 ],
 "shared_tags": ["人数和共享主体标签，如 1girl"],
 "outfit_tags": ["用户明确指定的服装"],
 "action_tags": ["动作和互动"],
 "scene_tags": ["场景、道具、明确光照或时间"],
 "style_tags": ["最多少量风格标签，可用 1.2::key_tag:: 强调关键元素"],
 "nsfw_level": "safe|suggestive|explicit"
}

规则：
- 角色仅在确定时才填写 danbooru_tag；不确定则留空，绝不编造。
- 普通标签必须英文小写下划线；不要写完整句子。
- 不得添加用户未明确描述的服装、道具、天气、光照、时间或外观。
- 不得输出 masterpiece、best_quality、画师标签、负面词、尺寸或比例词；插件会统一处理。
- 不要堆叠同义词；每个概念只保留一个最准确标签。
- 未提及的数组必须返回空数组。"""
FORMAT_RETRY_PROMPT = """上一次输出不符合指定 JSON schema。
请只返回完整、合法的 JSON 对象，不要 Markdown、解释或其它文字。
必须包含 characters、shared_tags、outfit_tags、action_tags、scene_tags、style_tags、nsfw_level；
characters 的每一项必须包含 display_name、danbooru_tag、tags。"""


class NaiPromptPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.config = config or {}
        self.lookup: TagLookup | None = None
        self._last_request: dict[str, float] = {}

    async def initialize(self) -> None:
        self.lookup = TagLookup(
            timeout_seconds=self._cfg_int("request_timeout_seconds", 5, 2, 15),
            proxy_url=self._cfg_str("proxy_url"),
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
        # AstrBot adapters expose this differently across versions; fail closed.
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
            response = await provider.text_chat(
                prompt=description,
                contexts=[],
                system_prompt=LLM_SYSTEM_PROMPT,
            )
            text = getattr(response, "completion_text", "") if response else ""
            parsed = parse_llm_response(text)
            if parsed is not None:
                return parsed
            logger.warning("[NAIPrompt] LLM 首次输出不符合 JSON schema，执行一次格式重试")
            retry_response = await provider.text_chat(
                prompt=f"{description}\n\n{FORMAT_RETRY_PROMPT}",
                contexts=[],
                system_prompt=LLM_SYSTEM_PROMPT,
            )
            retry_text = getattr(retry_response, "completion_text", "") if retry_response else ""
            return parse_llm_response(retry_text)
        except Exception as exc:
            logger.warning("[NAIPrompt] LLM 解析失败: %s", exc)
            return None

    async def _lookup_characters(self, parsed: ParsedRequest):
        if self.lookup is None:
            return []
        requests = parsed.characters[:20]
        results = await asyncio.gather(*(self.lookup.lookup(request) for request in requests))
        return [result for result in results if result is not None]

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
        sender_id = self._sender_id(event)
        throttled = self._cooldown_message(sender_id)
        if throttled:
            yield event.plain_result(throttled)
            return

        parsed = await self._parse_with_llm(event, description)
        if parsed is None:
            yield event.plain_result("提示词解析失败，请稍后重试。")
            return
        characters = await self._lookup_characters(parsed)
        result = build_prompt(
            parsed=parsed,
            description=description,
            lookup_results=characters,
            allow_adult=self._cfg_bool("allow_adult_prompts", True),
            max_length=self._cfg_int("max_prompt_length", 1800, 200, 5000),
            used_fallback=False,
        )
        yield event.plain_result(format_result(result))
