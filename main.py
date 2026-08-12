"""AstrBot NAI prompt helper plugin."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig

from .prompt_engine import (
    DANBOORU_SEARCH_API_DEFAULT,
    TAGGER_API_DEFAULT,
    CharacterTags,
    DanbooruSearchLookup,
    ImageTaggerClient,
    ParsedRequest,
    build_prompt,
    extract_json,
    format_result,
    parse_llm_response,
    resolve_nsfw_level,
)

HELP_TEXT = """用法：/提示词 <自然语言描述>

支持图片反推：命令后附带图片，或在文字中附图片链接，即可反推并整理 NAI 提示词（可附加文字微调）。

示例：
/提示词 流萤穿校服，在樱花树下微笑
/提示词 + 图片 → 反推图片中的角色与特征"""
MAX_DESCRIPTION_LENGTH = 500
IMAGE_URL_PATTERN = re.compile(r"https?://[^\s]+?\.(?:png|jpe?g|webp|gif|bmp)(?:\?[^\s]*)?", re.IGNORECASE)

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

TAGGER_ORGANIZE_SYSTEM_PROMPT = """你是 NAI 标签提示词解析器，负责把图片反推标签整理为合法 JSON。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。

输入格式：
"标签: <WD14/Danbooru 反推标签，逗号分隔>
描述: <用户附加描述，可为空>"

JSON schema:
{
 "characters": [
   {
     "display_name": "识别出的角色名（如 hatsune_miku）；未识别到角色填空字符串",
     "danbooru_tag": "留空字符串；角色标签由插件查询服务确定",
     "tags": []
   }
 ],
 "shared_tags": ["人数和共享主体标签，如 1girl"],
 "outfit_tags": ["服装标签"],
 "action_tags": ["动作和互动标签"],
 "scene_tags": ["场景、道具、光照或时间标签"],
 "style_tags": ["少量风格标签，可用 1.2::key_tag:: 强调关键元素"],
 "nsfw_level": "safe|suggestive|explicit"
}

规则：
- 以反推标签为事实来源，将标签按语义归类到上述数组；识别出的角色名标签填入 characters 的 display_name。
- 用户描述与标签冲突时以用户描述为准（如“去掉校服”“换白色头发”）。
- 丢弃画师名、masterpiece、best_quality 等通用质量标签。
- nsfw_level 根据标签判定：含 naked/nipples/pussy 等为 explicit；含 underwear/ecchi 等为 suggestive；否则 safe。
- 普通标签必须英文小写下划线；不要写完整句子。
- 未提及的数组必须返回空数组。"""


class NaiPromptPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.config = config or {}
        self.lookup: DanbooruSearchLookup | None = None
        self.tagger: ImageTaggerClient | None = None
        self._last_request: dict[str, float] = {}

    async def initialize(self) -> None:
        self.lookup = DanbooruSearchLookup(
            api_url=self._cfg_str("danbooru_search_api_url", DANBOORU_SEARCH_API_DEFAULT),
            timeout_seconds=self._cfg_int("request_timeout_seconds", 5, 2, 15),
        )
        self.tagger = ImageTaggerClient(
            api_url=self._cfg_str("tagger_api_url", TAGGER_API_DEFAULT),
            timeout_seconds=self._cfg_int("tagger_timeout_seconds", 30, 5, 60),
            confidence_threshold=self._cfg_float("tagger_confidence_threshold", 0.35, 0.0, 1.0),
        )
        logger.info("[NAIPrompt] 插件已加载")

    async def terminate(self) -> None:
        self._last_request.clear()
        self.lookup = None
        self.tagger = None
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

    def _cfg_float(self, key: str, default: float, minimum: float, maximum: float) -> float:
        """读取浮点配置项并夹取到 [minimum, maximum] 区间。

        Args:
            key: 配置键名
            default: 缺省值
            minimum: 允许的最小值
            maximum: 允许的最大值

        Returns:
            夹取后的浮点值
        """
        try:
            value = float(self.config.get(key, default))
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

    async def _parse_with_llm(
        self,
        event: AstrMessageEvent,
        prompt: str,
        system_prompt: str = LLM_SYSTEM_PROMPT,
        retry_prompt: str = FORMAT_RETRY_PROMPT,
    ) -> ParsedRequest | None:
        """通过 LLM 将输入文本整理为 ParsedRequest，格式非法时按提示重试一次。

        Args:
            event: 消息事件
            prompt: 用户输入内容（自然语言描述或图片反推标签）
            system_prompt: LLM 系统提示词
            retry_prompt: 首次输出不符合 schema 时的重试提示词

        Returns:
            解析结果；失败返回 None
        """
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
            if not provider:
                return None
            response = await provider.text_chat(prompt=prompt, contexts=[], system_prompt=system_prompt)
            parsed = parse_llm_response(getattr(response, "completion_text", "") if response else "")
            if parsed is not None:
                return parsed
            logger.warning("[NAIPrompt] LLM 首次输出不符合 JSON schema，执行一次格式重试")
            retry = await provider.text_chat(
                prompt=f"{prompt}\n\n{retry_prompt}", contexts=[], system_prompt=system_prompt,
            )
            return parse_llm_response(getattr(retry, "completion_text", "") if retry else "")
        except Exception as exc:
            logger.warning("[NAIPrompt] LLM 解析失败: %s", exc)
            return None

    @staticmethod
    def _extract_image_url(description: str) -> str | None:
        """从命令文字中提取第一个图片 URL。

        Args:
            description: 命令文字内容

        Returns:
            匹配到的图片 URL；未找到返回 None
        """
        match = IMAGE_URL_PATTERN.search(description)
        return match.group(0) if match else None

    @staticmethod
    def _strip_image_url(description: str) -> str:
        """从描述中移除图片 URL，避免图片链接混入 LLM 输入。

        Args:
            description: 命令文字内容

        Returns:
            移除图片 URL 后的文字
        """
        return IMAGE_URL_PATTERN.sub("", description).strip()

    async def _first_image_source(self, event: AstrMessageEvent) -> str | None:
        """取消息附带的第一张图片的本地路径或远程 URL。

        优先返回本地已存在路径（读取可靠），否则返回远程 URL。

        Args:
            event: 消息事件

        Returns:
            图片来源（路径或 URL）；无图片返回 None
        """
        message_obj = getattr(event, "message_obj", None)
        images = getattr(message_obj, "image", None) or []
        for image in images:
            path = getattr(image, "path", None) or ""
            url = getattr(image, "url", None) or ""
            if isinstance(path, str) and path and os.path.exists(path):
                return path
            if isinstance(url, str) and url:
                return url
        return None

    async def _download_image(self, source: str, max_bytes: int) -> tuple[bytes | None, str | None]:
        """读取或下载图片字节。

        Args:
            source: 本地文件路径或 http(s) URL
            max_bytes: 允许的最大字节数，超限返回错误

        Returns:
            (图片字节, 错误信息)；成功时错误信息为 None
        """
        if os.path.exists(source):
            try:
                if os.path.getsize(source) > max_bytes:
                    return None, "图片过大，请压缩后重试。"
                with open(source, "rb") as handle:
                    return handle.read(), None
            except OSError:
                return None, "图片读取失败，请重试。"
        timeout = aiohttp.ClientTimeout(total=self._cfg_int("tagger_timeout_seconds", 30, 5, 60))
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(source) as response:
                    if response.status != 200:
                        return None, "图片下载失败，请检查链接是否有效。"
                    body = await response.read()
                    if len(body) > max_bytes:
                        return None, "图片过大，请压缩后重试。"
                    return body, None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None, "图片下载失败，请稍后重试。"

    async def _parse_image_flow(
        self, event: AstrMessageEvent, source: str, description: str
    ) -> tuple[ParsedRequest | None, str | None]:
        """图片反推完整流程：下载图片 → Tagger 反推 → LLM 整理。

        Args:
            event: 消息事件
            source: 图片来源（本地路径或 URL）
            description: 去除图片 URL 后的用户描述（可为空）

        Returns:
            (解析结果, 错误信息)；成功时错误信息为 None
        """
        max_bytes = self._cfg_int("max_image_bytes", 10 * 1024 * 1024, 1024 * 1024, 50 * 1024 * 1024)
        image_bytes, error = await self._download_image(source, max_bytes)
        if error:
            return None, error
        if self.tagger is None:
            return None, "图像反推服务未初始化，请稍后重试。"
        tags = await self.tagger.tag_image(image_bytes or b"")
        if not tags:
            return None, "图片识别失败，请检查图片清晰度后重试。"
        prompt = f"标签: {', '.join(tags)}\n描述: {description or '（无）'}"
        parsed = await self._parse_with_llm(event, prompt, system_prompt=TAGGER_ORGANIZE_SYSTEM_PROMPT)
        if parsed is None:
            return None, "提示词整理失败，请稍后重试。"
        return parsed, None

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
        """/提示词 <自然语言描述>：生成可复制的 NAI 正负面提示词；附带图片时反推图片并整理提示词。"""
        description = str(description or "").strip()
        clean_description = self._strip_image_url(description)
        image_source = await self._first_image_source(event) or self._extract_image_url(description)
        if not clean_description and not image_source:
            yield event.plain_result(HELP_TEXT)
            return
        if len(clean_description) > MAX_DESCRIPTION_LENGTH:
            yield event.plain_result("描述过长，请精简至 500 字以内。")
            return
        if not self._allowed(event):
            yield event.plain_result("当前用户未获提示词功能授权，请联系管理员。")
            return
        throttled = self._cooldown_message(self._sender_id(event))
        if throttled:
            yield event.plain_result(throttled)
            return

        if image_source:
            parsed, error = await self._parse_image_flow(event, image_source, clean_description)
            if error:
                yield event.plain_result(error)
                return
        else:
            parsed = await self._parse_with_llm(event, clean_description)
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
