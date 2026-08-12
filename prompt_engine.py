"""LLM JSON validation, DanbooruSearch character enrichment and NAI rendering."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

DANBOORU_SEARCH_API_DEFAULT = "https://sakizuki-danboorusearch.hf.space/api"
CHARACTER_SCORE_THRESHOLD = 0.55
RELATED_TAG_LIMIT = 12

TAGGER_API_DEFAULT = "https://smilingwolf-wd-tagger.hf.space"
TAGGER_MODEL = "SmilingWolf/wd-swinv2-tagger-v3"

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


class ImageTaggerClient:
    """公网 WD14 图像标签反推客户端，兼容 Gradio 新协议（4.x/6.x 两步流程）与 3.x 单次请求。

    将图片字节反推为 Danbooru/NAI 风格标签列表，供 /提示词 命令的图片分支使用。
    优先尝试新协议（POST 提交 + SSE 拉取结果），失败后回退 3.x 单次请求。

    属性:
        api_url: Tagger 服务基址，末尾 / 会自动忽略
        timeout_seconds: 单次请求超时秒数（5-60）
        confidence_threshold: 标签置信度阈值（0-1），低于该值的标签被过滤
    """

    def __init__(
        self,
        api_url: str = TAGGER_API_DEFAULT,
        timeout_seconds: int = 30,
        confidence_threshold: float = 0.35,
    ):
        """初始化图片反推客户端。

        Args:
            api_url: Tagger API 基址，可填自建镜像
            timeout_seconds: 请求超时秒数
            confidence_threshold: 标签置信度阈值
        """
        self.api_url = (api_url or TAGGER_API_DEFAULT).rstrip("/")
        self.timeout_seconds = max(5, min(60, int(timeout_seconds)))
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))

    @staticmethod
    def _data_uri(image_bytes: bytes) -> str:
        """将图片字节转为带 MIME 嗅探的 base64 data URI，供 Gradio file 组件接收。

        Args:
            image_bytes: 原始图片字节

        Returns:
            形如 data:image/jpeg;base64,xxx 的 data URI 字符串
        """
        mime = "image/jpeg"
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
            mime = "image/gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime = "image/webp"
        return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    def _payload(self, data_uri: str) -> dict[str, Any]:
        """构建 Gradio 新协议命名参数请求体（图片 + 模型 + 双阈值）。

        Args:
            data_uri: 图片 data URI

        Returns:
            命名参数字典，image 为 gradio.FileData 对象，模型与阈值使用配置值
        """
        threshold = self.confidence_threshold
        return {
            "image": {"path": data_uri, "meta": {"_type": "gradio.FileData"}},
            "model_repo": TAGGER_MODEL,
            "general_thresh": threshold,
            "general_mcut_enabled": False,
            "character_thresh": threshold,
            "character_mcut_enabled": False,
        }

    @staticmethod
    def _parse_sse_data(text: str) -> Any:
        """从 Gradio 4.x SSE 响应文本中提取最后一个 data 负载。

        Args:
            text: SSE 流响应文本

        Returns:
            JSON 解析后的最后一个 data 值；无有效负载返回 None
        """
        result: Any = None
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                result = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
        return result

    @staticmethod
    def _extract_tags(data: Any) -> list[str]:
        """从 Gradio 响应 data 中提取合法的 Danbooru 标签列表。

        新协议输出为 4 元素数组 [标签串, Rating, General, Character]，仅取 data[0] 标签串；
        同时兼容旧协议形态（逗号拼接字符串 / {"label": ...} 列表）。

        Args:
            data: Gradio 响应的 data 字段

        Returns:
            去重后的标签列表
        """
        tags: list[str] = []

        def append_tag(value: Any) -> None:
            """清洗并追加单个标签。

            Args:
                value: 原始标签值，非字符串直接忽略
            """
            if not isinstance(value, str):
                return
            tag = value.strip().lower().replace(" ", "_")
            if tag and re.fullmatch(r"[a-z0-9_()\-]+", tag):
                tags.append(tag)

        if not isinstance(data, list) or not data:
            return []
        if isinstance(data[0], str):
            for part in data[0].split(","):
                append_tag(part)
            return _unique(tags)
        for entry in data:
            if isinstance(entry, str):
                for part in entry.split(","):
                    append_tag(part)
            elif isinstance(entry, list):
                for item in entry:
                    if isinstance(item, dict) and isinstance(item.get("label"), str):
                        append_tag(item["label"])
                    else:
                        append_tag(item)
            elif isinstance(entry, dict) and isinstance(entry.get("label"), str):
                append_tag(entry["label"])
        return _unique(tags)

    async def _predict_v4(self, session: aiohttp.ClientSession, payload: dict[str, Any]) -> list[str] | None:
        """走 Gradio 新协议两步流程：POST 提交拿 event_id，GET 拉取 SSE 结果。

        Args:
            session: 共享 aiohttp 会话
            payload: 命名参数请求体

        Returns:
            标签列表；失败返回 None
        """
        try:
            async with session.post(f"{self.api_url}/gradio_api/call/v2/predict", json=payload) as response:
                if response.status != 200:
                    return None
                body = await response.json(content_type=None)
            event_id = body.get("event_id") if isinstance(body, dict) else None
            if not event_id:
                return None
            # SSE 拉取 URL 通常与提交 URL 同前缀；部分镜像仅暴露非 v2 前缀，逐一尝试
            for prefix in (f"{self.api_url}/gradio_api/call/v2/predict", f"{self.api_url}/gradio_api/call/predict"):
                async with session.get(f"{prefix}/{event_id}") as response:
                    if response.status != 200:
                        continue
                    text = await response.text()
                    return self._extract_tags(self._parse_sse_data(text))
            return None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None

    async def _predict_v3(self, session: aiohttp.ClientSession, payload: dict[str, Any]) -> list[str] | None:
        """走 Gradio 3.x 单次流程：POST /api/predict 直接返回 JSON。

        Args:
            session: 共享 aiohttp 会话
            payload: 预测请求体

        Returns:
            标签列表；失败返回 None
        """
        try:
            async with session.post(f"{self.api_url}/api/predict", json=payload) as response:
                if response.status != 200:
                    return None
                body = await response.json(content_type=None)
            data = body.get("data") if isinstance(body, dict) else body
            return self._extract_tags(data)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None

    async def tag_image(self, image_bytes: bytes) -> list[str]:
        """将图片字节反推为 Danbooru 标签列表。

        优先尝试 Gradio 新协议（两步流程），失败后回退 3.x；响应为空时用最小请求体重试一次
        （部分简化镜像只暴露图片一个入参）。全部失败返回空列表。

        Args:
            image_bytes: 图片原始字节

        Returns:
            反推得到的标签列表，可能为空
        """
        data_uri = self._data_uri(image_bytes)
        payload = self._payload(data_uri)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {"User-Agent": "AstrBot-NAIPrompt/1.0"}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tags = await self._predict_v4(session, payload)
            if tags is None:
                tags = await self._predict_v3(session, payload)
            if not tags:
                minimal = {"image": {"path": data_uri, "meta": {"_type": "gradio.FileData"}}}
                tags = await self._predict_v4(session, minimal)
                if tags is None:
                    tags = await self._predict_v3(session, minimal)
        return tags or []


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
