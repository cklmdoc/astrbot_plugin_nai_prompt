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
    fallback_parse,
    format_result,
    parse_llm_response,
)

HELP_TEXT = """用法：/提示词 <自然语言描述>

示例：
/提示词 绿色双马尾猫耳少女，穿校服，在樱花树下微笑
/提示词 流萤和橘望在花园里拥抱"""
MAX_DESCRIPTION_LENGTH = 500

LLM_SYSTEM_PROMPT = """你是 NAI/Danbooru 标签解析器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。
将中文描述转换为小写 Danbooru 风格英文标签。不要编造角色标签；只有你确知的角色才放入 character_candidates，格式如 firefly_。
JSON schema:
{
 "character_candidates": ["character_tag_"],
 "character_groups": [["character_specific_tag"]],
 "subject_tags": ["1girl"],
 "outfit_tags": ["school_uniform"],
 "action_tags": ["smile"],
 "scene_tags": ["cherry_blossoms"],
 "style_tags": ["anime_coloring"],
 "nsfw_level": "safe|suggestive|explicit"
}
所有 tag 必须是英文小写下划线形式。未提及的字段返回空数组。"""


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
            seaart_enabled=self._cfg_bool("seaart_enabled", True),
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
            return parse_llm_response(text)
        except Exception as exc:
            logger.warning("[NAIPrompt] LLM 解析失败: %s", exc)
            return None

    async def _lookup_characters(self, parsed: ParsedRequest):
        if self.lookup is None:
            return []
        candidates = parsed.character_candidates[:20]
        results = await asyncio.gather(*(self.lookup.lookup(candidate) for candidate in candidates))
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
        used_fallback = parsed is None
        if parsed is None:
            parsed = fallback_parse(description)
        characters = await self._lookup_characters(parsed)
        result = build_prompt(
            parsed=parsed,
            description=description,
            lookup_results=characters,
            allow_adult=self._cfg_bool("allow_adult_prompts", True),
            max_length=self._cfg_int("max_prompt_length", 1800, 200, 5000),
            used_fallback=used_fallback,
        )
        yield event.plain_result(format_result(result))
