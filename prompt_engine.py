"""NAI prompt parsing, character tag lookup and safe prompt rendering."""

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
    "bad_hands",
    "extra_fingers",
    "deformed_hands",
    "lowres",
    "worst_quality",
    "bad_anatomy",
    "blurry",
    "distorted",
    "ugly",
]

# These terms flag a request which must never produce explicit sexual content.
MINOR_MARKERS = ("未成年", "小学生", "初中生", "幼女", "儿童", "童颜")
ADULT_MARKERS = ("r18", "r-18", "成人向", "色情", "全裸", "裸体", "露点", "涩涩", "瑟瑟", "做爱", "性交")

# Intentionally small, conservative fallback vocabulary. Unknown details are not invented.
FALLBACK_MAP: dict[str, list[str]] = {
    "绿头发": ["green_hair"], "绿色头发": ["green_hair"],
    "蓝头发": ["blue_hair"], "粉头发": ["pink_hair"],
    "白头发": ["white_hair"], "黑头发": ["black_hair"],
    "金发": ["blonde_hair"], "银发": ["silver_hair"],
    "双马尾": ["twintails"], "长发": ["long_hair"], "短发": ["short_hair"],
    "猫耳": ["cat_ears"], "兔耳": ["rabbit_ears"], "狐耳": ["fox_ears"],
    "校服": ["school_uniform"], "连衣裙": ["dress"], "裙子": ["skirt"],
    "微笑": ["smile"], "笑": ["smile"], "脸红": ["blush"],
    "拥抱": ["hug"], "牵手": ["holding_hands"], "坐": ["sitting"], "站": ["standing"],
    "樱花": ["cherry_blossoms"], "花园": ["garden"], "海边": ["beach"],
    "室内": ["indoors"], "室外": ["outdoors"], "夜晚": ["night"],
}


@dataclass
class ParsedRequest:
    character_candidates: list[str] = field(default_factory=list)
    character_groups: list[list[str]] = field(default_factory=list)
    subject_tags: list[str] = field(default_factory=list)
    outfit_tags: list[str] = field(default_factory=list)
    action_tags: list[str] = field(default_factory=list)
    scene_tags: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    nsfw_level: str = "safe"


@dataclass
class CharacterTags:
    name: str
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


def _as_tag_list(value: Any, limit: int = 80) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:limit]:
        if not isinstance(item, str):
            continue
        tag = item.strip().lower().replace(" ", "_")
        if re.fullmatch(r"[a-z0-9_()\-]+", tag) and tag:
            result.append(tag)
    return result


def _unique(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    return [tag for tag in tags if tag and not (tag in seen or seen.add(tag))]


def extract_json(text: str) -> dict[str, Any] | None:
    """Accept a JSON object optionally wrapped in a markdown code fence."""
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
    groups_raw = data.get("character_groups")
    groups: list[list[str]] = []
    if isinstance(groups_raw, list):
        groups = [_as_tag_list(group, 40) for group in groups_raw if isinstance(group, list)]
        groups = [group for group in groups if group]
    candidates = _as_tag_list(data.get("character_candidates"), 20)
    level = str(data.get("nsfw_level", "safe")).lower()
    if level not in {"safe", "suggestive", "explicit"}:
        level = "safe"
    return ParsedRequest(
        character_candidates=candidates,
        character_groups=groups,
        subject_tags=_as_tag_list(data.get("subject_tags")),
        outfit_tags=_as_tag_list(data.get("outfit_tags")),
        action_tags=_as_tag_list(data.get("action_tags")),
        scene_tags=_as_tag_list(data.get("scene_tags")),
        style_tags=_as_tag_list(data.get("style_tags")),
        nsfw_level=level,
    )


def fallback_parse(description: str) -> ParsedRequest:
    tags: list[str] = []
    for chinese, mapped in FALLBACK_MAP.items():
        if chinese in description:
            tags.extend(mapped)
    # Conservative estimate; do not claim a gender when not specified.
    if any(word in description for word in ("两个", "两位", "二人", "双人")):
        tags.insert(0, "2girls" if "少女" in description or "女孩" in description else "2people")
    elif any(word in description for word in ("少女", "女孩", "女生", "女人")):
        tags.insert(0, "1girl")
    elif any(word in description for word in ("男孩", "男生", "男人")):
        tags.insert(0, "1boy")
    else:
        tags.insert(0, "solo")
    level = "explicit" if any(marker in description.lower() for marker in ADULT_MARKERS) else "safe"
    return ParsedRequest(subject_tags=_unique(tags), nsfw_level=level)


def safe_nsfw_level(description: str, requested: str, allow_adult: bool) -> str:
    lowered = description.lower()
    if any(marker in description for marker in MINOR_MARKERS):
        return "safe"
    if not allow_adult:
        return "safe"
    if requested == "explicit" and any(marker in lowered for marker in ADULT_MARKERS):
        return "explicit"
    if requested == "suggestive" or "擦边" in description:
        return "suggestive"
    return "safe"


def nsfw_tags(level: str) -> list[str]:
    if level == "suggestive":
        return ["underwear", "ecchi", "no_nudity", "no_nipples", "no_pussy"]
    if level == "explicit":
        return ["completely_nude", "naked", "nipples", "pussy"]
    return ["cute"]


class TagLookup:
    """Network tag lookup with a successful-result-only, 24-hour memory cache."""

    CACHE_TTL = 86400

    def __init__(self, timeout_seconds: int = 5, proxy_url: str = "", seaart_enabled: bool = True):
        self.timeout_seconds = max(2, min(15, int(timeout_seconds)))
        self.proxy_url = proxy_url or None
        self.seaart_enabled = seaart_enabled
        self._cache: dict[str, tuple[float, CharacterTags]] = {}

    async def lookup(self, candidate: str) -> CharacterTags | None:
        candidate = candidate.strip().lower().replace(" ", "_")
        if not candidate:
            return None
        cached = self._cache.get(candidate)
        if cached and time.monotonic() - cached[0] < self.CACHE_TTL:
            return cached[1]
        result = await self._danbooru(candidate)
        if result is None and self.seaart_enabled:
            result = await self._seaart(candidate)
        if result is not None:
            self._cache[candidate] = (time.monotonic(), result)
        return result

    async def _get(self, url: str) -> str | None:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AstrBot-NAIPrompt/1.0)"}
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, proxy=self.proxy_url) as response:
                    if response.status != 200:
                        return None
                    return await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

    async def _danbooru(self, candidate: str) -> CharacterTags | None:
        # Exact name is deliberately used; fuzzy name_matches can return unrelated characters.
        url = "https://danbooru.donmai.us/tags.json?search[name_matches]=" + quote_plus(candidate) + "&search[category]=4"
        raw = await self._get(url)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None
        for item in data:
            name = item.get("name") if isinstance(item, dict) else None
            if isinstance(name, str) and name == candidate:
                return CharacterTags(name=candidate, tags=[name], source="Danbooru")
        return None

    async def _seaart(self, candidate: str) -> CharacterTags | None:
        # Public search fallback: it deliberately only accepts explicit, tag-shaped data.
        url = "https://www.seaart.ai/search?q=" + quote_plus(candidate)
        raw = await self._get(url)
        if raw is None:
            return None
        found = re.findall(r"(?:Training Tags|Prompt Tags)\s*[:：]\s*([^<\n]{3,500})", html.unescape(raw), re.I)
        for value in found:
            tags = _as_tag_list(re.split(r"[,，]", value), 40)
            if tags:
                return CharacterTags(name=candidate, tags=tags, source="SeaArt")
        return None


def build_prompt(parsed: ParsedRequest, description: str, lookup_results: list[CharacterTags], allow_adult: bool, max_length: int, used_fallback: bool = False) -> PromptResult:
    level = safe_nsfw_level(description, parsed.nsfw_level, allow_adult)
    character_tags = _unique([tag for result in lookup_results for tag in result.tags])
    source = " + ".join(_unique([result.source for result in lookup_results]))
    if not source:
        source = "OC 转换（未获取到可验证的角色训练标签）"

    all_tags = _unique(
        POSITIVE_BASE + parsed.subject_tags + character_tags + parsed.outfit_tags
        + parsed.action_tags + parsed.scene_tags + parsed.style_tags + nsfw_tags(level)
    )
    max_length = max(200, min(5000, int(max_length)))
    selected: list[str] = []
    for tag in all_tags:
        candidate = ", ".join(selected + [tag])
        if len(candidate) > max_length:
            break
        selected.append(tag)
    positive = ", ".join(selected)
    return PromptResult(
        positive=positive,
        negative=", ".join(NEGATIVE_BASE),
        character_tags=character_tags,
        source=source,
        nsfw_level={"safe": "全年龄", "suggestive": "擦边", "explicit": "成人向"}[level],
        used_fallback=used_fallback,
        multi_character_note=len(lookup_results) >= 2 or len(parsed.character_groups) >= 2,
    )


def format_result(result: PromptResult) -> str:
    char_tags = ", ".join(result.character_tags) if result.character_tags else "未识别到可验证角色标签"
    lines = [
        "┌─ NAI 提示词生成 ─────────",
        f"│ 标签来源：{result.source}",
        f"│ NSFW 等级：{result.nsfw_level}",
        "├─ 角色标签",
        f"│ {char_tags}",
        "├─ 正面 Prompt",
        "```text",
        result.positive,
        "```",
        "├─ 负面 Prompt",
        "```text",
        result.negative,
        "```",
    ]
    if result.multi_character_note:
        lines.extend([
            "├─ NAI V4 多角色建议",
            "│ 将每位角色专属标签分别填入 Character Prompting；动作、互动和场景保留在主 Prompt。",
        ])
    if result.used_fallback:
        lines.append("│ 解析服务暂不可用，已使用基础标签转换。")
    lines.append("└────────────────────────")
    return "\n".join(lines)
