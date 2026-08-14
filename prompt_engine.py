"""LLM JSON validation, DanbooruSearch character enrichment and NAI rendering."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger("astrbot_plugin_nai_prompt.prompt_engine")

DANBOORU_SEARCH_API_DEFAULT = "https://sakizuki-danboorusearch.hf.space/api"
CHARACTER_SCORE_THRESHOLD = 0.55
RELATED_TAG_LIMIT = 12

TAGGER_API_DEFAULT = "https://smilingwolf-wd-tagger.hf.space"
TAGGER_MODEL = "SmilingWolf/wd-swinv2-tagger-v3"
# 角色标签阈值默认值，贴合官方 Space UI 默认（general 0.35 / character 0.85）
TAGGER_CHARACTER_THRESHOLD = 0.85

DEDUP_SYSTEM_PROMPT = """你是 NAI 标签去重器。只返回一个合法 JSON 对象，不要 Markdown、解释或额外文字。
输入是一组英文小写 NAI/Danbooru 标签，可能含 NAI 新版权重语法（如 1.2::tag::）或互动标签（如 source#hug）。

JSON schema:
{
  "deduped_tags": ["去重后的标签列表"]
}

规则：
- 合并语义相同的同义词，每个概念只保留一个最准确、最贴合输入的标签。
- 保持原有顺序，不要重排；保留权重语法。
- 不要新增输入中没有的标签，不要丢弃有实质语义差异的标签。
- 输出必须为合法 JSON。"""

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
    position: str = ""


@dataclass
class WeightEntry:
    """单条权重配置：目标标签与强调等级。

    属性:
        tag: 目标英文小写普通标签（不含权重语法）
        level: 强调等级，取值 weak/strong/very_strong
    """

    tag: str = ""
    level: str = "strong"


# 三档权重映射：新版 NAI 权重::标签:: 语法
WEIGHT_WEAK_VALUE = 0.8
WEIGHT_STRONG_VALUE = 1.2
WEIGHT_VERY_STRONG_VALUE = 1.5


@dataclass
class ParsedRequest:
    characters: list[CharacterRequest] = field(default_factory=list)
    shared_tags: list[str] = field(default_factory=list)
    outfit_tags: list[str] = field(default_factory=list)
    action_tags: list[str] = field(default_factory=list)
    scene_tags: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    nsfw_level: str = "safe"
    weights: list[WeightEntry] = field(default_factory=list)


@dataclass
class CharacterTags:
    display_name: str
    canonical_tag: str
    tags: list[str]
    copyright_tag: str = ""


@dataclass
class PromptResult:
    positive: str
    negative: str
    character_tags: list[str]
    source: str
    nsfw_level: str


def _as_tag_list(value: Any, limit: int = 80, allow_weights: bool = False) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    # 标签支持互动前缀（source#/target#/mutual#）；权重语法为 NAI 新版 权重::标签::
    tag_pattern = r"(?:[a-z0-9_()#\-]+|\d+(?:\.\d+)?::[a-z0-9_()#,\-]+::)"
    for item in value[:limit]:
        if not isinstance(item, str):
            continue
        raw = item.strip()
        # 渲染文字 Text: 内容 保留原样（NAI 特殊语法，可含大小写/标点/空格）
        if raw.lower().startswith("text:"):
            result.append(raw)
            continue
        tag = raw.lower().replace(" ", "_")
        if tag and re.fullmatch(tag_pattern if allow_weights else r"[a-z0-9_()#\-]+", tag):
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


def _parse_weights(value: Any) -> list[WeightEntry]:
    """解析 LLM 输出的 weights 字段为 WeightEntry 列表。

    Args:
        value: LLM JSON 中的 weights 字段，应为 [{tag, level}] 列表

    Returns:
        合法的 WeightEntry 列表；非法项被静默跳过
    """
    if not isinstance(value, list):
        return []
    result: list[WeightEntry] = []
    for item in value[:40]:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag")
        level = item.get("level")
        if not isinstance(tag, str) or not isinstance(level, str):
            continue
        normalized = tag.strip().lower().replace(" ", "_")
        if not re.fullmatch(r"[a-z0-9_\-]+", normalized):
            continue
        if level not in {"weak", "strong", "very_strong"}:
            continue
        result.append(WeightEntry(tag=normalized, level=level))
    return result


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
        position = item.get("position", "")
        if not isinstance(position, str):
            position = ""
        characters.append(CharacterRequest(
            display_name=item["display_name"].strip()[:100],
            danbooru_tag=canonical_hint[0] if canonical_hint else "",
            tags=_as_tag_list(item["tags"], 40),
            position=position.strip()[:20],
        ))
    return ParsedRequest(
        characters=characters,
        shared_tags=_as_tag_list(data["shared_tags"]),
        outfit_tags=_as_tag_list(data["outfit_tags"]),
        action_tags=_as_tag_list(data["action_tags"]),
        scene_tags=_as_tag_list(data["scene_tags"]),
        style_tags=_as_tag_list(data["style_tags"]),
        nsfw_level=data["nsfw_level"],
        weights=_parse_weights(data.get("weights")),
    )


def resolve_nsfw_level(parsed: ParsedRequest, allow_adult: bool) -> str:
    if not allow_adult:
        return "safe"
    return parsed.nsfw_level


def render_weighted_tag(tag: str, level: str) -> str:
    """将普通标签按等级渲染为 NAI 新版 权重::标签:: 语法。

    Args:
        tag: 英文小写普通标签
        level: 强调等级，weak/strong/very_strong

    Returns:
        weak -> 0.8::tag::，strong -> 1.2::tag::，very_strong -> 1.5::tag::；
        其它等级原样返回
    """
    if level == "weak":
        return f"{WEIGHT_WEAK_VALUE}::{tag}::"
    if level == "very_strong":
        return f"{WEIGHT_VERY_STRONG_VALUE}::{tag}::"
    if level == "strong":
        return f"{WEIGHT_STRONG_VALUE}::{tag}::"
    return tag


def _apply_weights(tags: list[str], weights: list[WeightEntry]) -> list[str]:
    """按权重配置将标签渲染为 NAI 权重语法。

    仅对普通标签（不含括号/权重语法）应用；权重表中不存在或非普通标签保持原样。

    Args:
        tags: 待套权重的标签列表（保持顺序）
        weights: 权重配置列表

    Returns:
        套用权重后的标签列表
    """
    level_map = {entry.tag: entry.level for entry in weights}
    result: list[str] = []
    for tag in tags:
        level = level_map.get(tag)
        if level and re.fullmatch(r"[a-z0-9_\-]+", tag):
            result.append(render_weighted_tag(tag, level))
        else:
            result.append(tag)
    return result


def format_character_name(canonical_tag: str, copyright_tag: str) -> str:
    """组合角色 canonical 标签与作品名为可读的 人物名(作品名) 形式。

    Args:
        canonical_tag: 角色 canonical 标签（下划线形式）
        copyright_tag: 作品名标签（可为空）

    Returns:
        人物名(作品名) 可读形式；无作品名时仅人物名
    """
    name = canonical_tag.replace("_", " ")
    if copyright_tag:
        return f"{name} ({copyright_tag.replace('_', ' ')})"
    return name


# 位置词到自然语言的映射，用于多角色位置的自然语言强化
_POSITION_WORDS = {
    "left": "left",
    "right": "right",
    "center": "center",
    "middle": "center",
    "top": "top",
    "up": "top",
    "upper": "top",
    "bottom": "bottom",
    "down": "bottom",
    "lower": "bottom",
}


def format_position(position: str) -> str:
    """将角色位置值转为自然语言短语，如 left -> on the left。

    Args:
        position: 位置值（英文，可为组合，如 top left）

    Returns:
        自然语言位置短语；空串或无法识别返回空串
    """
    words = [w for w in position.strip().lower().split() if w]
    if not words:
        return ""
    mapped = [_POSITION_WORDS.get(w) for w in words]
    if not mapped or any(m is None for m in mapped):
        return ""
    phrase = " ".join(mapped)
    if phrase == "center":
        return "in the center"
    return f"on the {phrase}"


def subject_gender(shared_tags: list[str]) -> str:
    """从共享标签中提取角色性别标签（不带数字），如 2girls -> girl。

    Args:
        shared_tags: 共享标签列表

    Returns:
        girl/boy/other；未识别返回空串
    """
    for tag in shared_tags:
        match = re.fullmatch(r"\d+(girl|boy|other)s?", tag)
        if match:
            return match.group(1)
    return ""


def looks_like_prompt(text: str) -> bool:
    """判断文本是否像本插件生成的 NAI 提示词。

    按逗号/竖线切分 token，统计符合标签语法的有效 token 数量与占比，
    双阈值（数量 ≥ 3 且占比 ≥ 60%）判定，避免关键词误判。

    Args:
        text: 待判断文本

    Returns:
        像提示词返回 True，否则 False
    """
    if not text:
        return False
    tokens = [t.strip() for t in re.split(r"[,|]", text)]
    tokens = [t for t in tokens if t]
    if len(tokens) < 3:
        return False
    tag_pattern = re.compile(r"[a-z0-9_()#\-]+")
    weight_pattern = re.compile(r"\d+(?:\.\d+)?::[a-z0-9_()#,\-]+::")
    valid = 0
    for token in tokens:
        lowered = token.lower()
        # 渲染文字 Text: 内容 视为有效
        if lowered.startswith("text:"):
            valid += 1
            continue
        tag = lowered.replace(" ", "_")
        if tag_pattern.fullmatch(tag) or weight_pattern.fullmatch(tag):
            valid += 1
    return valid >= 3 and valid >= len(tokens) * 0.6


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
            "target_categories": ["Character", "Copyright"],
            "group_mode": "off",
        })
        if search_data is None:
            return None
        results = search_data.get("results", [])
        character_candidates = [
            item for item in results
            if isinstance(item, dict) and item.get("category") == "Character"
            and isinstance(item.get("tag"), str)
        ]
        if not character_candidates:
            return None
        candidate = max(character_candidates, key=lambda item: float(item.get("final_score", 0) or 0))
        try:
            score = float(candidate.get("final_score", 0))
        except (TypeError, ValueError):
            return None
        if score < CHARACTER_SCORE_THRESHOLD:
            return None
        canonical_tag = candidate["tag"]

        # 作品名（Copyright）为可选项，取最高分；失败不影响角色识别
        copyright_tag = ""
        copyright_candidates = [
            item for item in results
            if isinstance(item, dict) and item.get("category") == "Copyright"
            and isinstance(item.get("tag"), str)
        ]
        if copyright_candidates:
            best_copyright = max(copyright_candidates, key=lambda item: float(item.get("final_score", 0) or 0))
            copyright_tag = best_copyright["tag"]

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
        result = CharacterTags(request.display_name, canonical_tag, related_tags, copyright_tag)
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
        # 最近一次失败原因，供上层在报错时展示以辅助排查
        self._last_error = ""

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

    def _payload(self, image_path: str) -> dict[str, Any]:
        """构建 Gradio 新协议命名参数请求体（图片 + 模型 + 双阈值）。

        Args:
            image_path: 图片在服务器端的路径（经 /gradio_api/upload 上传后返回），
                或 data URI（镜像不支持上传时的回退）

        Returns:
            命名参数字典，image 为 gradio.FileData 对象，模型与阈值使用配置值
        """
        threshold = self.confidence_threshold
        return {
            "image": {"path": image_path, "meta": {"_type": "gradio.FileData"}},
            "model_repo": TAGGER_MODEL,
            "general_thresh": threshold,
            "general_mcut_enabled": False,
            "character_thresh": TAGGER_CHARACTER_THRESHOLD,
            "character_mcut_enabled": False,
        }

    async def _upload_image(self, session: aiohttp.ClientSession, image_bytes: bytes) -> str:
        """按 Gradio 6 标准流程将图片 multipart 上传到 /gradio_api/upload，返回服务器端路径。

        Args:
            session: 共享 aiohttp 会话
            image_bytes: 图片原始字节

        Returns:
            服务器端文件路径；失败返回空串并在 self._last_error 记录原因
        """
        url = f"{self.api_url}/gradio_api/upload"
        form = aiohttp.FormData()
        form.add_field("files", image_bytes, filename="image.png", content_type="application/octet-stream")
        try:
            async with session.post(url, data=form) as response:
                if response.status != 200:
                    self._last_error = f"上传 HTTP {response.status}（{url}）"
                    logger.error("[Tagger] %s", self._last_error)
                    return ""
                paths = await response.json(content_type=None)
            if isinstance(paths, list) and paths and isinstance(paths[0], str):
                return paths[0]
            self._last_error = f"上传响应异常: {str(paths)[:200]}（{url}）"
            logger.error("[Tagger] %s", self._last_error)
            return ""
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            self._last_error = f"上传异常: {type(exc).__name__}: {exc}（{url}）"
            logger.error("[Tagger] %s", self._last_error)
            return ""

    @staticmethod
    def _parse_sse_data(text: str) -> Any:
        """从 Gradio 4.x/6.x SSE 响应文本中提取结果负载。

        优先取 `event: complete` 事件对应的 data（openapi 确认最终事件即 complete）；
        无事件标记时退化为最后一个 data 行；整体非 SSE 时尝试按裸 JSON 解析（部分镜像直接返回）。

        Args:
            text: SSE 流响应文本

        Returns:
            JSON 解析后的结果负载；无有效负载返回 None
        """
        complete_payload: Any = None
        fallback: Any = None
        current_event = ""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
                continue
            if not line.startswith("data:"):
                continue
            try:
                value = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if current_event == "complete":
                complete_payload = value
            fallback = value
        if complete_payload is not None:
            return complete_payload
        if fallback is not None:
            return fallback
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _scan_complete_data(text: str) -> Any:
        """从累积的 SSE 文本中查找完整出现的 event: complete 事件，返回其 data。

        供流式读取增量调用：数据未收完整时返回 None，追加后再调用即可。
        半行（chunk 边界切割）因 JSON 解析失败或关键字不完整会被安全忽略。

        Args:
            text: 已累积的 SSE 文本

        Returns:
            complete 事件的 data；尚未出现完整 complete 事件返回 None
        """
        current_event = ""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                try:
                    value = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                if current_event == "complete":
                    return value
        return None

    async def _fetch_sse(self, session: aiohttp.ClientSession, url: str) -> tuple[Any, str]:
        """流式拉取 Gradio SSE，收到 event: complete 即返回其 data，不等待流结束。

        Args:
            session: 共享 aiohttp 会话
            url: SSE 拉取 URL

        Returns:
            (data, error)：data 为 complete 事件的 JSON；失败时 data 为 None 并附错误说明
        """
        # SSE 拉取单独放宽总超时：模型加载/推理可能超过默认超时，等待 complete 后立即返回
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds + 60)
        buffer = ""
        try:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    return None, f"SSE 拉取 HTTP {response.status}（{url}）"
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    value = self._scan_complete_data(buffer)
                    if value is not None:
                        return value, ""
            # 流被服务端关闭后兜底解析
            value = self._scan_complete_data(buffer)
            if value is not None:
                return value, ""
            value = self._parse_sse_data(buffer)
            if value is not None:
                return value, ""
            return None, f"SSE 流结束但未收到有效数据（{url}）"
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            return None, f"SSE 拉取异常: {type(exc).__name__}: {exc}（{url}）"

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
        # 兼容部分版本把 outputs 再包一层：data = [[标签串, LabelData, ...]]，展平后再处理
        if len(data) == 1 and isinstance(data[0], list) and data[0] and isinstance(data[0][0], str):
            data = data[0]
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

        失败时在 self._last_error 记录具体原因，并输出日志便于排查。

        Args:
            session: 共享 aiohttp 会话
            payload: 命名参数请求体

        Returns:
            标签列表；失败返回 None
        """
        submit_url = f"{self.api_url}/gradio_api/call/v2/predict"
        try:
            async with session.post(submit_url, json=payload) as response:
                if response.status != 200:
                    self._last_error = f"提交请求 HTTP {response.status}（{submit_url}）"
                    logger.error("[Tagger] %s", self._last_error)
                    return None
                body = await response.json(content_type=None)
            event_id = body.get("event_id") if isinstance(body, dict) else None
            if not event_id:
                self._last_error = f"提交响应缺少 event_id：{str(body)[:200]}"
                logger.error("[Tagger] %s", self._last_error)
                return None
            # 标准 SSE 端点为 /gradio_api/call/predict/{event_id}（openapi 确认）；
            # v2 前缀在部分镜像也有效，作为兼容候选。解析为空时继续尝试下一个前缀。
            for prefix in (f"{self.api_url}/gradio_api/call/predict", f"{self.api_url}/gradio_api/call/v2/predict"):
                data, err = await self._fetch_sse(session, f"{prefix}/{event_id}")
                if data is None:
                    self._last_error = err
                    logger.error("[Tagger] %s", self._last_error)
                    continue
                tags = self._extract_tags(data)
                if tags:
                    return tags
                self._last_error = f"SSE 结果为空（{prefix}/{event_id}）"
                logger.error("[Tagger] %s，SSE data: %s", self._last_error, str(data)[:300])
            return None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            self._last_error = f"请求异常: {type(exc).__name__}: {exc}"
            logger.error("[Tagger] %s（%s）", self._last_error, submit_url)
            return None

    async def _predict_v3(self, session: aiohttp.ClientSession, payload: dict[str, Any]) -> list[str] | None:
        """走 Gradio 3.x 单次流程：POST /api/predict 直接返回 JSON。

        失败时在 self._last_error 记录具体原因，并输出日志便于排查。

        Args:
            session: 共享 aiohttp 会话
            payload: 预测请求体

        Returns:
            标签列表；失败返回 None
        """
        try:
            async with session.post(f"{self.api_url}/api/predict", json=payload) as response:
                if response.status != 200:
                    self._last_error = f"v3 提交 HTTP {response.status}（{self.api_url}/api/predict）"
                    logger.error("[Tagger] %s", self._last_error)
                    return None
                body = await response.json(content_type=None)
            data = body.get("data") if isinstance(body, dict) else body
            tags = self._extract_tags(data)
            if not tags:
                self._last_error = "v3 响应 data 为空"
                logger.error("[Tagger] %s，响应: %s", self._last_error, str(body)[:300])
            return tags
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            self._last_error = f"v3 请求异常: {type(exc).__name__}: {exc}"
            logger.error("[Tagger] %s", self._last_error)
            return None

    async def tag_image(self, image_bytes: bytes) -> list[str]:
        """将图片字节反推为 Danbooru 标签列表。

        按 Gradio 6 标准流程先上传图片拿到服务器路径，再走新协议两步流程；
        上传失败时回退为内联 base64 data URI；响应为空时用最小请求体重试一次。
        全部失败返回空列表。

        Args:
            image_bytes: 图片原始字节

        Returns:
            反推得到的标签列表，可能为空
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {"User-Agent": "AstrBot-NAIPrompt/1.0"}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            image_path = await self._upload_image(session, image_bytes)
            if not image_path:
                # 上传失败（无鉴权/镜像不支持），回退为直接内联 base64 data URI
                image_path = self._data_uri(image_bytes)
            payload = self._payload(image_path)
            tags = await self._predict_v4(session, payload)
            if tags is None:
                # v4 为正确协议，其失败原因优先保留；v3 大多因 Space 不支持而 404/405，仅作兜底
                v4_error = self._last_error or ""
                tags = await self._predict_v3(session, payload)
                if not tags:
                    self._last_error = v4_error or self._last_error or ""
            if not tags:
                minimal = {"image": {"path": image_path, "meta": {"_type": "gradio.FileData"}}}
                v4_error = self._last_error or ""
                tags = await self._predict_v4(session, minimal)
                if tags is None:
                    minimal_v4_error = self._last_error or ""
                    tags = await self._predict_v3(session, minimal)
                    if not tags:
                        self._last_error = minimal_v4_error or v4_error or self._last_error or ""
        if not tags:
            logger.error("[Tagger] 图片反推最终失败，原因: %s", self._last_error or "未知")
        return tags or []


async def _dedupe_tags_semantic(provider: Any, tags: list[str]) -> list[str]:
    """通过 LLM 对标签做语义去重，合并同义词并保持原顺序与权重语法。

    Args:
        provider: LLM provider，需支持 text_chat(prompt, contexts, system_prompt)
        tags: 待去重的标签列表

    Returns:
        语义去重后的标签列表；LLM 失败或无有效输出时返回原列表（降级不去重）
    """
    if not tags or provider is None:
        return tags
    try:
        response = await provider.text_chat(
            prompt=", ".join(tags),
            contexts=[],
            system_prompt=DEDUP_SYSTEM_PROMPT,
        )
        text = getattr(response, "completion_text", "") if response else ""
        data = extract_json(text)
        if data and isinstance(data.get("deduped_tags"), list):
            deduped = _as_tag_list(data["deduped_tags"], len(tags) + 16, allow_weights=True)
            if deduped:
                return deduped
    except Exception as exc:
        logger.warning("[NAIPrompt] LLM 语义去重失败，降级不去重: %s", exc)
    return tags


def _truncate_tags(tags: list[str], max_length: int) -> list[str]:
    """按字符上限截断标签列表；max_length <= 0 表示不截断。

    Args:
        tags: 标签列表
        max_length: 最大字符数，非正数时不截断

    Returns:
        截断后的标签列表
    """
    if max_length <= 0:
        return tags
    max_length = max(200, min(5000, int(max_length)))
    selected: list[str] = []
    for tag in tags:
        if len(", ".join(selected + [tag])) > max_length:
            break
        selected.append(tag)
    return selected


async def build_prompt(
    parsed: ParsedRequest,
    lookup_results: list[CharacterTags | None],
    allow_adult: bool,
    max_length: int,
    provider: Any = None,
) -> PromptResult:
    """按 NAI 新版格式组装最终正面提示词。

    单角色：人物名(作品名) 与特征、服装、动作、场景、风格等平铺；
    多角色：用 | 分隔 base prompt 与各角色 prompt（官方语法），位置用自然语言强化。
    权重统一用 权重::标签:: 语法，语义去重失败降级不去重。

    Args:
        parsed: LLM 解析结果
        lookup_results: DanbooruSearch 角色查询结果列表，与 parsed.characters 等长（未命中为 None）
        allow_adult: 是否允许成人向
        max_length: 正面提示词最大字符数，0 表示不截断（仅单角色生效）
        provider: LLM provider，用于语义去重；None 时跳过语义去重

    Returns:
        组装后的 PromptResult
    """
    level = resolve_nsfw_level(parsed, allow_adult)

    # 构建每个角色的可读名与专属标签（含权重套用）
    char_entries: list[tuple[str, list[str], str]] = []
    char_names: list[str] = []
    for index, char in enumerate(parsed.characters):
        result = lookup_results[index] if index < len(lookup_results) else None
        if result is not None:
            name = format_character_name(result.canonical_tag, result.copyright_tag)
            tags = _unique(result.tags + char.tags)
        else:
            name = char.display_name.strip()
            tags = _unique(char.tags)
        tags = _apply_weights(tags, parsed.weights)
        char_entries.append((name, tags, char.position.strip()))
        if name:
            char_names.append(name)

    # 共享标签（人数、服装、动作、场景、风格、NSFW）
    shared_tags = _unique(
        parsed.shared_tags + parsed.outfit_tags + parsed.action_tags
        + parsed.scene_tags + parsed.style_tags + nsfw_tags(level)
    )
    shared_tags = _apply_weights(shared_tags, parsed.weights)

    if len(char_entries) >= 2:
        # 多角色：| 分隔 base prompt 与各角色 prompt（官方语法）
        gender = subject_gender(shared_tags)
        base = await _dedupe_tags_semantic(provider, shared_tags)
        char_prompts: list[str] = []
        for name, tags, position in char_entries:
            parts: list[str] = []
            if gender and gender not in tags:
                parts.append(gender)
            if name:
                parts.append(name)
            parts.extend(tags)
            pos = format_position(position)
            if pos:
                parts.append(pos)
            char_prompts.append(", ".join(parts))
        positive = " | ".join([", ".join(base)] + char_prompts)
    else:
        # 单角色或原创：平铺（角色名不参与去重/截断，始终在最前）
        if char_entries:
            name, tags, _ = char_entries[0]
            body = tags + shared_tags
            body = await _dedupe_tags_semantic(provider, body)
            body = _truncate_tags(body, max_length)
            positive = ", ".join(([name] if name else []) + body)
        else:
            body = await _dedupe_tags_semantic(provider, shared_tags)
            body = _truncate_tags(body, max_length)
            positive = ", ".join(body)

    source = "DanbooruSearch 角色标签" if any(r is not None for r in lookup_results) else "LLM 转译"
    return PromptResult(
        positive=positive,
        negative="",
        character_tags=char_names,
        source=source,
        nsfw_level={"safe": "全年龄", "suggestive": "擦边", "explicit": "成人向"}[level],
    )


def format_result(result: PromptResult) -> str:
    """将 PromptResult 格式化为极简纯文本：正面提示词 + 一行元信息注释。

    Args:
        result: 组装结果

    Returns:
        正面提示词纯文本，末尾附带来源、NSFW、角色的一行注释
    """
    meta = [f"来源:{result.source}", f"NSFW:{result.nsfw_level}"]
    if result.character_tags:
        meta.append(f"角色:{', '.join(result.character_tags)}")
    return f"{result.positive}\n（{' | '.join(meta)}）"


# 生图服务默认值与占位符（本地 AstrBot 生图插件不校验 Key/模型名）
IMAGE_GEN_API_DEFAULT = "http://127.0.0.1:8765"
IMAGE_GEN_MODEL = "astrbot-image"
IMAGE_GEN_API_KEY = "astrbot"
IMAGE_GEN_SIZE = "1024x1024"


class ImageGeneratorClient:
    """OpenAI Images API 兼容的生图客户端，调用本地 AstrBot 生图插件生成示例图。

    属性:
        api_url: 生图服务基址（OpenAI Images API 兼容）
        timeout_seconds: 单次生图请求超时秒数
        _last_error: 最近一次失败原因
    """

    def __init__(self, api_url: str = IMAGE_GEN_API_DEFAULT, timeout_seconds: int = 180):
        """初始化生图客户端。

        Args:
            api_url: 生图服务基址，末尾 / 会自动忽略
            timeout_seconds: 请求超时秒数
        """
        self.api_url = (api_url or IMAGE_GEN_API_DEFAULT).rstrip("/")
        self.timeout_seconds = max(30, min(600, int(timeout_seconds)))
        self._last_error = ""

    async def generate(self, prompt: str, count: int) -> list[bytes]:
        """调用生图服务生成 count 张图片。

        Args:
            prompt: 生图用的正面提示词
            count: 生成张数（夹取到 1-4）

        Returns:
            图片字节列表；失败返回空列表并在 self._last_error 记录原因
        """
        if not prompt:
            return []
        count = max(1, min(4, int(count)))
        url = f"{self.api_url}/v1/images/generations"
        payload = {
            "model": IMAGE_GEN_MODEL,
            "prompt": prompt,
            "n": count,
            "size": IMAGE_GEN_SIZE,
            "response_format": "b64_json",
        }
        headers = {
            "Authorization": f"Bearer {IMAGE_GEN_API_KEY}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        self._last_error = f"生图 HTTP {response.status}（{url}）"
                        logger.error("[ImageGen] %s", self._last_error)
                        return []
                    body = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            self._last_error = f"生图请求异常: {type(exc).__name__}: {exc}（{url}）"
            logger.error("[ImageGen] %s", self._last_error)
            return []

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            self._last_error = f"生图响应缺少 data 列表: {str(body)[:200]}"
            logger.error("[ImageGen] %s", self._last_error)
            return []
        images: list[bytes] = []
        for item in data[:count]:
            image_bytes = await self._extract_image(item)
            if image_bytes:
                images.append(image_bytes)
        if not images:
            self._last_error = "生图响应未包含有效图片"
            logger.error("[ImageGen] %s，响应: %s", self._last_error, str(body)[:300])
        return images

    async def _extract_image(self, item: Any) -> bytes | None:
        """从单个生图结果项中提取图片字节，优先 b64_json，其次 url 下载。

        Args:
            item: data 数组中的单项

        Returns:
            图片字节；失败返回 None
        """
        if not isinstance(item, dict):
            return None
        b64 = item.get("b64_json")
        if isinstance(b64, str) and b64:
            try:
                return base64.b64decode(b64)
            except (ValueError, TypeError):
                return None
        url = item.get("url")
        if isinstance(url, str) and url:
            return await self._download_image(url)
        return None

    async def _download_image(self, url: str) -> bytes | None:
        """下载远程图片 URL 返回字节。

        Args:
            url: 图片 URL

        Returns:
            图片字节；失败返回 None
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    return await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None
