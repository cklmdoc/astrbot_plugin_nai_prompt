"""NAI prompt parsing, SeaArt-first character tag lookup and safe rendering."""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

import aiohttp

POSITIVE_BASE = ["masterpiece", "best quality"]
NEGATIVE_BASE = [
    "bad_hands", "extra_fingers", "deformed_hands", "lowres", "worst_quality",
    "bad_anatomy", "blurry", "distorted", "ugly",
]
MINOR_MARKERS = ("未成年", "小学生", "初中生", "幼女", "儿童", "童颜")
ADULT_MARKERS = ("r18", "r-18", "成人向", "色情", "全裸", "裸体", "露点", "涩涩", "瑟瑟", "做爱", "性交")


SEAART_DROP_EXACT = {
    "masterpiece", "best_quality", "best quality", "absurdres", "very_aesthetic",
    "highres", "ultra_detailed", "highly_detailed", "scenery", "outdoors", "indoors",
    "looking_at_viewer", "simple_background", "white_background", "watermark", "signature",
    "text", "lowres", "worst_quality", "bad_anatomy", "bad_hands", "blurry",
}
SEAART_DROP_PREFIXES = ("artist:", "bad_", "negative_", "no_")
SEAART_TAG_LIMIT = 20
SEAART_DETAIL_LIMIT = 3


@dataclass
class CharacterRequest:
    display_name: str = ""
    danbooru_tag: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ParsedRequest:
    characters: list[CharacterRequest] = field(default_factory=list)
    shared_tags: list[str] = field(default_factory=list)
    outfit_tags: list[str] = field(default_factory=list)
    action_tags: list[str] = field(default_factory=list)
    scene_tags: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    nsfw_level: str = "safe"


@dataclass
class CharacterTags:
    display_name: str
    danbooru_tag: str
    tags: list[str]
    source: str


@dataclass
class PromptResult:
    positive: str
    negative: str
    character_tags: list[str]
    source: str
    nsfw_level: str
    used_fallback: bool = False
    multi_character_note: bool = False


def _as_tag_list(value: Any, limit: int = 80, allow_weights: bool = False) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    tag_pattern = r"(?:[a-z0-9_()\-]+|(?:0?\.[5-9]|1(?:\.\d+)?)::[a-z0-9_()\-]+::)"
    for item in value[:limit]:
        if not isinstance(item, str):
            continue
        tag = item.strip().lower().replace(" ", "_")
        if re.fullmatch(tag_pattern if allow_weights else r"[a-z0-9_()\-]+", tag) and tag:
            result.append(tag)
    return result


def _as_single_tag(value: Any) -> str:
    return _as_tag_list([value], 1)[0] if isinstance(value, str) and _as_tag_list([value], 1) else ""


def _unique(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    return [tag for tag in tags if tag and not (tag in seen or seen.add(tag))]


def extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_llm_response(text: str) -> ParsedRequest | None:
    data = extract_json(text)
    if data is None:
        return None
    required_arrays = ("characters", "shared_tags", "outfit_tags", "action_tags", "scene_tags", "style_tags")
    if any(key not in data or not isinstance(data[key], list) for key in required_arrays):
        return None
    characters: list[CharacterRequest] = []
    for item in data["characters"][:20]:
        if not isinstance(item, dict):
            return None
        if not {"display_name", "danbooru_tag", "tags"}.issubset(item):
            return None
        if not isinstance(item["display_name"], str) or not isinstance(item["danbooru_tag"], str):
            return None
        if not isinstance(item["tags"], list):
            return None
        display_name = item["display_name"].strip()[:100]
        danbooru_tag = _as_single_tag(item["danbooru_tag"])
        tags = _as_tag_list(item["tags"], 40)
        characters.append(CharacterRequest(display_name, danbooru_tag, tags))
    level = data.get("nsfw_level")
    if level not in {"safe", "suggestive", "explicit"}:
        return None
    return ParsedRequest(
        characters=characters,
        shared_tags=_as_tag_list(data["shared_tags"]),
        outfit_tags=_as_tag_list(data["outfit_tags"]),
        action_tags=_as_tag_list(data["action_tags"]),
        scene_tags=_as_tag_list(data["scene_tags"]),
        style_tags=_as_tag_list(data["style_tags"], allow_weights=True),
        nsfw_level=level,
    )


def safe_nsfw_level(description: str, requested: str, allow_adult: bool) -> str:
    if any(marker in description for marker in MINOR_MARKERS) or not allow_adult:
        return "safe"
    lowered = description.lower()
    if requested == "explicit" and any(marker in lowered for marker in ADULT_MARKERS):
        return "explicit"
    return "suggestive" if requested == "suggestive" or "擦边" in description else "safe"


def nsfw_tags(level: str) -> list[str]:
    if level == "suggestive":
        return ["underwear", "ecchi", "no_nudity", "no_nipples", "no_pussy"]
    if level == "explicit":
        return ["completely_nude", "naked", "nipples", "pussy"]
    return ["cute"]


def clean_seaart_tags(tags: list[str]) -> list[str]:
    cleaned: list[str] = []
    for tag in _unique(_as_tag_list(tags, 200)):
        if tag in SEAART_DROP_EXACT or tag.startswith(SEAART_DROP_PREFIXES):
            continue
        if any(word in tag for word in ("watermark", "signature", "quality", "resolution")):
            continue
        cleaned.append(tag)
        if len(cleaned) == SEAART_TAG_LIMIT:
            break
    return cleaned


class TagLookup:
    """SeaArt-first tag lookup with 24-hour successful-result alias caching."""

    CACHE_TTL = 86400

    def __init__(self, timeout_seconds: int = 5, proxy_url: str = ""):
        self.timeout_seconds = max(2, min(15, int(timeout_seconds)))
        self.proxy_url = proxy_url or None
        self._cache: dict[str, tuple[float, CharacterTags]] = {}

    @staticmethod
    def _key(value: str) -> str:
        return value.strip().lower().replace(" ", "_")

    async def lookup(self, request: CharacterRequest) -> CharacterTags | None:
        keys = [self._key(value) for value in (request.display_name, request.danbooru_tag) if value.strip()]
        for key in keys:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < self.CACHE_TTL:
                return cached[1]

        seaart = await self._seaart(request)
        canonical = await self._danbooru(request.danbooru_tag) if request.danbooru_tag else None
        if seaart:
            tags = _unique(([canonical] if canonical else []) + seaart)
            result = CharacterTags(request.display_name, canonical or request.danbooru_tag, tags, "SeaArt + Danbooru 角色标识" if canonical else "SeaArt")
        elif canonical:
            result = CharacterTags(request.display_name, canonical, [canonical], "Danbooru")
        else:
            return None

        for key in keys + ([self._key(canonical)] if canonical else []):
            if key:
                self._cache[key] = (time.monotonic(), result)
        return result

    async def _get(self, url: str) -> str | None:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AstrBot-NAIPrompt/1.0)"}
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, proxy=self.proxy_url) as response:
                    return await response.text() if response.status == 200 else None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

    async def _danbooru(self, candidate: str) -> str | None:
        candidate = self._key(candidate)
        if not candidate:
            return None
        url = "https://danbooru.donmai.us/tags.json?search[name_matches]=" + quote_plus(candidate) + "&search[category]=4"
        raw = await self._get(url)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and item.get("name") == candidate:
                return candidate
        return None

    @staticmethod
    def _walk(value: Any):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from TagLookup._walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from TagLookup._walk(child)

    @staticmethod
    def _title_matches(item: dict[str, Any], request: CharacterRequest) -> bool:
        haystack = " ".join(
            str(item.get(key, ""))
            for key in ("title", "name", "modelName", "character", "tags")
        ).lower()
        candidates = [request.display_name.lower(), request.danbooru_tag.lower().replace("_", " ")]
        return any(candidate and candidate in haystack for candidate in candidates)

    @staticmethod
    def _tag_value(item: dict[str, Any]) -> list[str]:
        for key, value in item.items():
            normalized = key.lower().replace("_", " ")
            if normalized in {"training tags", "prompt tags", "trainingtags", "prompttags"}:
                if isinstance(value, str):
                    return re.split(r"[,，]", value)
                if isinstance(value, list):
                    return [str(part) for part in value]
        return []

    @staticmethod
    def _detail_urls(item: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for key, value in item.items():
            if key.lower() in {"url", "detailurl", "modelurl", "link", "href"} and isinstance(value, str):
                if value.startswith("/"):
                    urls.append("https://www.seaart.ai" + value)
                elif value.startswith("https://www.seaart.ai/"):
                    urls.append(value)
        return _unique(urls)

    def _extract_seaart(self, raw: str, request: CharacterRequest) -> tuple[list[str], list[str]]:
        document = html.unescape(raw)
        direct = re.findall(r"(?:Training Tags|Prompt Tags)\s*[:：]\s*([^<\n]{3,1000})", document, re.I)
        for value in direct:
            tags = clean_seaart_tags(re.split(r"[,，]", value))
            if tags:
                return tags, []

        objects: list[dict[str, Any]] = []
        for payload in re.findall(r"<script[^>]*>(\{.*?\})</script>", document, re.S):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            objects.extend(node for node in self._walk(parsed) if isinstance(node, dict))
        urls: list[str] = []
        for item in objects:
            if not self._title_matches(item, request):
                continue
            tags = clean_seaart_tags(self._tag_value(item))
            if tags:
                return tags, []
            urls.extend(self._detail_urls(item))
        return [], _unique(urls)[:SEAART_DETAIL_LIMIT]

    async def _seaart(self, request: CharacterRequest) -> list[str] | None:
        search_terms = [request.display_name.strip(), request.danbooru_tag.strip()]
        for term in _unique([term for term in search_terms if term]):
            suffix = " AI模型" if term == request.display_name.strip() else ""
            raw = await self._get("https://www.seaart.ai/search?q=" + quote_plus(term + suffix))
            if raw is None:
                continue
            tags, detail_urls = self._extract_seaart(raw, request)
            if tags:
                return tags
            for url in detail_urls:
                detail = await self._get(url)
                if detail is None:
                    continue
                tags, _ = self._extract_seaart(detail, request)
                if tags:
                    return tags
        return None


def build_prompt(parsed: ParsedRequest, description: str, lookup_results: list[CharacterTags], allow_adult: bool, max_length: int, used_fallback: bool = False) -> PromptResult:
    level = safe_nsfw_level(description, parsed.nsfw_level, allow_adult)
    character_tags = _unique([tag for result in lookup_results for tag in result.tags])
    source = " + ".join(_unique([result.source for result in lookup_results])) or "OC 转换（未获取到可验证的角色训练标签）"
    user_character_tags = [tag for character in parsed.characters for tag in character.tags]
    all_tags = _unique(
        POSITIVE_BASE + parsed.shared_tags + character_tags + user_character_tags + parsed.outfit_tags
        + parsed.action_tags + parsed.scene_tags + parsed.style_tags + nsfw_tags(level)
    )
    max_length = max(200, min(5000, int(max_length)))
    selected: list[str] = []
    for tag in all_tags:
        if len(", ".join(selected + [tag])) > max_length:
            break
        selected.append(tag)
    return PromptResult(
        positive=", ".join(selected), negative=", ".join(NEGATIVE_BASE), character_tags=character_tags,
        source=source, nsfw_level={"safe": "全年龄", "suggestive": "擦边", "explicit": "成人向"}[level],
        used_fallback=used_fallback,
        multi_character_note=len(parsed.characters) >= 2,
    )


def format_result(result: PromptResult) -> str:
    char_tags = ", ".join(result.character_tags) if result.character_tags else "未识别到可验证角色标签"
    lines = [
        "┌─ NAI 提示词生成 ─────────", f"│ 标签来源：{result.source}", f"│ NSFW 等级：{result.nsfw_level}",
        "├─ 角色标签", f"│ {char_tags}", "├─ 正面 Prompt", "```text", result.positive, "```",
        "├─ 负面 Prompt", "```text", result.negative, "```",
    ]
    if result.multi_character_note:
        lines.extend(["├─ NAI V4 多角色建议", "│ 将每位角色专属标签分别填入 Character Prompting；动作、互动和场景保留在主 Prompt。"])
    if result.used_fallback:
        lines.append("│ 解析服务暂不可用，已使用基础标签转换。")
    lines.append("└────────────────────────")
    return "\n".join(lines)
