"""LLM JSON validation, DanbooruSearch character enrichment and NAI rendering."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

DANBOORU_SEARCH_API_DEFAULT = "https://sakizuki-danboorusearch.hf.space/api"
CHARACTER_SCORE_THRESHOLD = 0.55
RELATED_TAG_LIMIT = 12

POSITIVE_BASE = ["masterpiece", "best quality"]
NEGATIVE_BASE = [
    "bad_hands", "extra_fingers", "deformed_hands", "lowres", "worst_quality",
    "bad_anatomy", "blurry", "distorted", "ugly",
]

# These apply only to API-returned related tags, never to user/LLM tags.
RELATED_DROP_EXACT = {
    "masterpiece", "best_quality", "best quality", "absurdres", "very_aesthetic",
    "highres", "ultra_detailed", "highly_detailed", "scenery", "outdoors", "indoors",
    "looking_at_viewer", "simple_background", "white_background", "watermark", "signature",
    "text", "lowres", "worst_quality", "bad_anatomy", "bad_hands", "blurry",
}
RELATED_DROP_PREFIXES = ("artist:", "bad_", "negative_", "no_")


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
    canonical_tag: str
    tags: list[str]


@dataclass
class PromptResult:
    positive: str
    negative: str
    character_tags: list[str]
    source: str
    nsfw_level: str
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
        if tag and re.fullmatch(tag_pattern if allow_weights else r"[a-z0-9_()\-]+", tag):
            result.append(tag)
    return result


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    return [value for value in values if value and not (value in seen or seen.add(value))]


def extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def parse_llm_response(text: str) -> ParsedRequest | None:
    data = extract_json(text)
    if data is None:
        return None
    required_arrays = ("characters", "shared_tags", "outfit_tags", "action_tags", "scene_tags", "style_tags")
    if any(key not in data or not isinstance(data[key], list) for key in required_arrays):
        return None
    if data.get("nsfw_level") not in {"safe", "suggestive", "explicit"}:
        return None
    characters: list[CharacterRequest] = []
    for item in data["characters"][:20]:
        if not isinstance(item, dict) or not {"display_name", "danbooru_tag", "tags"}.issubset(item):
            return None
        if not isinstance(item["display_name"], str) or not isinstance(item["danbooru_tag"], str) or not isinstance(item["tags"], list):
            return None
        canonical_hint = _as_tag_list([item["danbooru_tag"]], 1)
        characters.append(CharacterRequest(
            display_name=item["display_name"].strip()[:100],
            danbooru_tag=canonical_hint[0] if canonical_hint else "",
            tags=_as_tag_list(item["tags"], 40),
        ))
    return ParsedRequest(
        characters=characters,
        shared_tags=_as_tag_list(data["shared_tags"]),
        outfit_tags=_as_tag_list(data["outfit_tags"]),
        action_tags=_as_tag_list(data["action_tags"]),
        scene_tags=_as_tag_list(data["scene_tags"]),
        style_tags=_as_tag_list(data["style_tags"], allow_weights=True),
        nsfw_level=data["nsfw_level"],
    )


def resolve_nsfw_level(parsed: ParsedRequest, allow_adult: bool) -> str:
    if not allow_adult:
        return "safe"
    return parsed.nsfw_level


def nsfw_tags(level: str) -> list[str]:
    if level == "suggestive":
        return ["underwear", "ecchi", "no_nudity", "no_nipples", "no_pussy"]
    if level == "explicit":
        return ["completely_nude", "naked", "nipples", "pussy"]
    return ["cute"]


def clean_related_tags(tags: list[str]) -> list[str]:
    cleaned: list[str] = []
    for tag in _unique(_as_tag_list(tags, 200)):
        if tag in RELATED_DROP_EXACT or tag.startswith(RELATED_DROP_PREFIXES):
            continue
        if any(word in tag for word in ("watermark", "signature", "quality", "resolution")):
            continue
        cleaned.append(tag)
        if len(cleaned) >= RELATED_TAG_LIMIT:
            break
    return cleaned


class DanbooruSearchLookup:
    """Public DanbooruSearch API client with successful-result-only 24-hour cache."""

    CACHE_TTL = 86400

    def __init__(self, api_url: str = DANBOORU_SEARCH_API_DEFAULT, timeout_seconds: int = 5):
        self.api_url = (api_url or DANBOORU_SEARCH_API_DEFAULT).rstrip("/")
        self.timeout_seconds = max(2, min(15, int(timeout_seconds)))
        self._cache: dict[str, tuple[float, CharacterTags]] = {}

    @staticmethod
    def _key(value: str) -> str:
        return value.strip().lower()

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {"User-Agent": "AstrBot-NAIPrompt/1.0"}
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.post(f"{self.api_url}/{endpoint}", json=payload) as response:
                    if response.status != 200:
                        return None
                    data = await response.json(content_type=None)
                    return data if isinstance(data, dict) else None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None

    async def lookup(self, request: CharacterRequest, show_nsfw: bool) -> CharacterTags | None:
        key = self._key(request.display_name)
        if not key:
            return None
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self.CACHE_TTL:
            return cached[1]

        search_data = await self._post("search", {
            "query": request.display_name,
            "top_k": 5,
            "limit": 20,
            "show_nsfw": show_nsfw,
            "target_categories": ["Character"],
            "group_mode": "off",
        })
        if search_data is None:
            return None
        candidates = [
            item for item in search_data.get("results", [])
            if isinstance(item, dict) and item.get("category") == "Character"
            and isinstance(item.get("tag"), str)
        ]
        if not candidates:
            return None
        candidate = max(candidates, key=lambda item: float(item.get("final_score", 0) or 0))
        try:
            score = float(candidate.get("final_score", 0))
        except (TypeError, ValueError):
            return None
        if score < CHARACTER_SCORE_THRESHOLD:
            return None
        canonical_tag = candidate["tag"]

        # Related tags are optional enhancement. A failure never drops canonical identity.
        related_data = await self._post("related", {
            "tags": [canonical_tag],
            "limit": 50,
            "show_nsfw": show_nsfw,
            "target_categories": ["General"],
        })
        related_tags: list[str] = []
        if related_data is not None:
            related_tags = clean_related_tags([
                item.get("tag", "") for item in related_data.get("results", [])
                if isinstance(item, dict) and item.get("category") == "General"
            ])
        result = CharacterTags(request.display_name, canonical_tag, [canonical_tag] + related_tags)
        self._cache[key] = (time.monotonic(), result)
        return result


def build_prompt(parsed: ParsedRequest, lookup_results: list[CharacterTags], allow_adult: bool, max_length: int) -> PromptResult:
    level = resolve_nsfw_level(parsed, allow_adult)
    resolved_tags = _unique([tag for result in lookup_results for tag in result.tags])
    user_character_tags = [tag for character in parsed.characters for tag in character.tags]
    all_tags = _unique(
        POSITIVE_BASE + parsed.shared_tags + resolved_tags + user_character_tags + parsed.outfit_tags
        + parsed.action_tags + parsed.scene_tags + parsed.style_tags + nsfw_tags(level)
    )
    max_length = max(200, min(5000, int(max_length)))
    selected: list[str] = []
    for tag in all_tags:
        if len(", ".join(selected + [tag])) > max_length:
            break
        selected.append(tag)
    source = "DanbooruSearch 角色标签" if lookup_results else "LLM 转译"
    return PromptResult(
        positive=", ".join(selected),
        negative=", ".join(NEGATIVE_BASE),
        character_tags=resolved_tags,
        source=source,
        nsfw_level={"safe": "全年龄", "suggestive": "擦边", "explicit": "成人向"}[level],
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
    lines.append("└────────────────────────")
    return "\n".join(lines)
