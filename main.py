"""AstrBot NAI prompt helper plugin."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from collections import deque

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig

from .prompt_engine import (
    DANBOORU_SEARCH_API_DEFAULT,
    IMAGE_GEN_API_DEFAULT,
    TAGGER_API_DEFAULT,
    CharacterTags,
    DanbooruSearchLookup,
    ImageGeneratorClient,
    ImageTaggerClient,
    ParsedRequest,
    build_prompt,
    extract_json,
    fetch_url_bytes,
    format_result,
    looks_like_prompt,
    parse_llm_response,
    resolve_nsfw_level,
)

HELP_TEXT = """用法：/提示词 <自然语言描述>

支持图片反推：命令后附带图片，或在文字中附图片链接，即可反推并整理 NAI 提示词（可附加文字微调）。
支持强调/弱化：用"突出/强调/重点/弱化/淡化"等词控制标签权重。

示例：
/提示词 流萤穿校服，在樱花树下微笑
/提示词 突出红色长发，弱化背景，穿校服
/提示词 + 图片 → 反推图片中的角色与特征"""
MAX_DESCRIPTION_LENGTH = 500
IMAGE_URL_PATTERN = re.compile(r"https?://[^\s]+?\.(?:png|jpe?g|webp|gif|bmp)(?:\?[^\s]*)?", re.IGNORECASE)


def _normalize_prompt_text(text: str) -> str:
    """折叠空白用于引用溯源匹配；不改变提示词语义。"""
    return " ".join(str(text).split())

LLM_SYSTEM_PROMPT = """你是 NAI 标签提示词解析器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。
将用户自然语言转换为紧凑、英文小写 NAI 新版格式标签数据。

JSON schema:
{
 "characters": [
   {
     "display_name": "角色名（中文或英文常用名）；原创角色填空字符串",
     "danbooru_tag": "留空字符串；角色标签与作品名由插件查询服务确定",
     "tags": ["该角色专属标签，含外貌特征；多角色互动时含互动标签"],
     "position": "多角色时该角色位置，如 left/right/center/up/down 及其组合；未提及填空字符串"
   }
 ],
 "shared_tags": ["人数和共享主体标签，如 1girl"],
 "outfit_tags": ["用户明确指定的服装"],
 "action_tags": ["动作和互动"],
 "scene_tags": ["场景、道具、明确光照或时间"],
 "style_tags": ["少量风格标签"],
 "nsfw_level": "safe|suggestive|explicit",
 "weights": [
   {"tag": "与上述某个标签完全一致的英文小写标签", "level": "weak|strong|very_strong"}
 ]
}

规则：
- 角色仅在明确提及既有角色时填写 display_name（角色最终会以 人物名(作品名) 形式由插件自动组合）；原创 OC 的 characters 可为空数组。
- danbooru_tag 必须留空字符串。
- 普通标签必须英文小写下划线；不要写完整句子。
- 互动标签：两个及以上角色有互动时，在对应角色 tags 写 source#动作（发起者）/ target#动作（承受者）/ mutual#动作（互相），如 source#hug。
- 渲染文字：用户要求角色说话/画面文字时，在 scene_tags 或 style_tags 写 Text: 内容；不想要文字写 no text。
- 情绪词：可加入少量情绪描述标签增强表现力。
- 不得添加用户未明确描述的服装、道具、天气、光照、时间或外观。
- 不得输出 masterpiece、best_quality、画师标签、负面词、尺寸或比例词。
- 不要堆叠同义词/重复/无意义标签；每个概念只保留一个最准确标签，描述清楚构图即可。
- 未提及的数组必须返回空数组。
- 强调/弱化（weights）：仅当用户明确表达强调或弱化意图时填写。
  - 突出/强调/重点 → strong；非常/极其/最/强烈 → very_strong；弱化/淡化/忽略/不要 → weak。
  - tag 必须是上述数组中已存在的、完全一致的英文小写普通标签（不含权重语法）。
  - 用户未表达强调意图时 weights 返回空数组。"""

FORMAT_RETRY_PROMPT = """上一次输出不符合指定 JSON schema。
请只返回完整、合法的 JSON 对象，不要 Markdown、解释或其它文字。
必须包含 characters、shared_tags、outfit_tags、action_tags、scene_tags、style_tags、nsfw_level、weights；
characters 的每一项必须包含 display_name、danbooru_tag、tags、position；
weights 的每一项必须包含 tag、level（level 取 weak/strong/very_strong）。"""

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
     "danbooru_tag": "留空字符串；角色标签与作品名由插件查询服务确定",
     "tags": ["该角色专属标签，含外貌特征；多角色互动时含互动标签"],
     "position": "多角色时该角色位置，如 left/right/center/up/down 及其组合；未提及填空字符串"
   }
 ],
 "shared_tags": ["人数和共享主体标签，如 1girl"],
 "outfit_tags": ["服装标签"],
 "action_tags": ["动作和互动标签"],
 "scene_tags": ["场景、道具、光照或时间标签"],
 "style_tags": ["风格标签"],
 "nsfw_level": "safe|suggestive|explicit",
 "weights": [
   {"tag": "与上述某个标签完全一致的英文小写标签", "level": "weak|strong|very_strong"}
 ]
}

规则：
- 以反推标签为事实来源，将标签按语义归类到上述数组；识别出的角色名标签填入 characters 的 display_name（角色最终会以 人物名(作品名) 形式由插件自动组合）。
- 用户描述与标签冲突时以用户描述为准（如“去掉校服”“换白色头发”）。
- 丢弃画师名、masterpiece、best_quality 等通用质量标签。
- nsfw_level 根据标签判定：含 naked/nipples/pussy 等为 explicit；含 underwear/ecchi 等为 suggestive；否则 safe。
- 互动标签：多角色有互动时，在对应角色 tags 写 source#动作（发起者）/ target#动作（承受者）/ mutual#动作（互相）。
- 渲染文字：画面/用户要求文字时，在 scene_tags 或 style_tags 写 Text: 内容；不想要文字写 no text。
- 情绪词：可加入少量情绪描述标签增强表现力。
- 普通标签必须英文小写下划线；不要写完整句子。
- 未提及的数组必须返回空数组。
- 主动优化（weights）：识别图片主体与核心特征，自动给核心特征标签加权（核心服装/显著特征 → strong，次要背景/弱化元素 → weak），并补全反推不出的风格、构图、光照、景别缺失标签到 style_tags/scene_tags。
  - 仅补全风格/构图/光照/景别，不脑补具体道具、服装、角色等事实性标签。
  - 用户描述中有强调/弱化意图时，同样按 strong/very_strong/weak 填入 weights。
  - tag 必须是上述数组中已存在的、完全一致的英文小写普通标签（不含权重语法）。
  - 无需要时 weights 返回空数组。"""

INCREMENTAL_EDIT_SYSTEM_PROMPT = """你是 NAI 提示词增量编辑器。根据用户的修改指令，修改给定的 NAI 提示词。只输出修改后的完整提示词文本，不要 Markdown、解释或额外文字。

输入格式：
原提示词: <完整 NAI 提示词>
修改指令: <用户的修改要求>

规则：
- 只修改用户指令涉及的部分，保留其余标签、权重语法（权重::标签::）、| 分隔结构与互动标签不变。
- 修改后输出完整的提示词（不要只输出改动部分）。
- 保持英文小写标签与原有格式。
- 用户要求去掉某标签时直接删除；要求修改时替换；要求新增时补到合适位置。
- 不要添加用户未要求的标签。"""


class NaiPromptPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.config = config or {}
        self.lookup: DanbooruSearchLookup | None = None
        self.tagger: ImageTaggerClient | None = None
        self.generator: ImageGeneratorClient | None = None
        self._last_request: dict[str, float] = {}
        # 查询/冲突过滤并发的共享信号量，避免多角色请求打爆外部服务
        self._lookup_sem = asyncio.Semaphore(4)
        # 最近输出的提示词缓冲（时间戳, 会话标识, 文本），供 self_id 缺失时引用溯源
        self._recent_prompts: deque[tuple[float, str, str]] = deque(maxlen=50)

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
        self.generator = ImageGeneratorClient(
            api_url=self._cfg_str("image_api_url", IMAGE_GEN_API_DEFAULT),
            timeout_seconds=180,
            allowed_hosts=tuple(self._cfg_list("allowed_image_hosts")),
        )
        logger.info("[NAIPrompt] 插件已加载")

    async def terminate(self) -> None:
        self._last_request.clear()
        self._recent_prompts.clear()
        self.lookup = None
        self.tagger = None
        self.generator = None
        logger.info("[NAIPrompt] 插件已停止")

    def _cfg_str(self, key: str, default: str = "") -> str:
        value = self.config.get(key, default) if hasattr(self.config, "get") else default
        return str(value).strip() if value is not None else default

    def _cfg_list(self, key: str, default: list | None = None) -> list[str]:
        """读取列表配置项；兼容字符串单值形态，去除空项与首尾空白。"""
        value = self.config.get(key, default) if hasattr(self.config, "get") else default
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

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
        # 管理员始终放行，不受白名单开关与列表约束
        if self._is_admin(event):
            return True
        # 白名单开关未启用时，开放给所有用户
        if not self._cfg_bool("enable_whitelist", False):
            return True
        # 白名单已启用：仅白名单内用户放行；名单为空时仅管理员可用
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

    @staticmethod
    def _message_components(event: AstrMessageEvent) -> list:
        """获取消息链组件列表，兼容不同 AstrBot 版本的取数 API。

        Args:
            event: 消息事件

        Returns:
            消息组件列表（可能为空）
        """
        for method in ("get_message", "get_messages"):
            getter = getattr(event, method, None)
            if not callable(getter):
                continue
            try:
                chain = getter()
            except Exception:
                continue
            if isinstance(chain, list):
                return chain
        message = getattr(getattr(event, "message_obj", None), "message", None)
        return message if isinstance(message, list) else []

    async def _first_image_source(self, event: AstrMessageEvent) -> str | None:
        """取消息附带的第一张图片的本地路径或远程 URL。

        通过消息链 Image 组件定位图片（AstrBot 标准 API），
        优先返回本地已存在路径（读取可靠），否则返回远程 URL；无图片返回 None。

        Args:
            event: 消息事件

        Returns:
            图片来源（路径或 URL）；无图片返回 None
        """
        for component in self._message_components(event):
            is_image = isinstance(component, Image) or component.__class__.__name__.lower() == "image"
            if not is_image:
                continue
            path = getattr(component, "path", None) or ""
            url = getattr(component, "url", None) or ""
            file_ref = getattr(component, "file", None) or ""
            if isinstance(path, str) and path and os.path.exists(path):
                return path
            # 部分适配器仅提供需鉴权的远程 URL，优先用适配器落盘接口取本地路径
            converter = getattr(component, "convert_to_file_path", None)
            if callable(converter):
                try:
                    local = await converter()
                except Exception:
                    local = None
                if isinstance(local, str) and local and os.path.exists(local):
                    return local
            if isinstance(url, str) and url:
                return url
            if isinstance(file_ref, str) and file_ref and (
                os.path.exists(file_ref) or file_ref.startswith(("http://", "https://"))
            ):
                return file_ref
        # 兜底：部分平台版本将图片挂在 message_obj.image
        for image in getattr(getattr(event, "message_obj", None), "image", None) or []:
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
        allowed_hosts = frozenset(self._cfg_list("allowed_image_hosts"))
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                status, body = await fetch_url_bytes(session, source, allowed_hosts=allowed_hosts)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None, "图片下载失败，请稍后重试。"
        if status is None:
            # 安全策略拦截（SSRF 防护）；不向用户泄露拦截细节，仅服务端留痕
            logger.warning("[NAIPrompt] 图片 URL 被安全策略拦截: %s", source)
            return None, "图片链接被拒绝，请更换图片源。"
        if status != 200:
            return None, "图片下载失败，请检查链接是否有效。"
        if len(body) > max_bytes:
            return None, "图片过大，请压缩后重试。"
        return body, None

    async def _parse_image_flow(
        self, event: AstrMessageEvent, source: str, description: str
    ) -> tuple[ParsedRequest | None, str | None]:
        """图片反推完整流程：下载图片 → Tagger 反推 → LLM 整理。

        整理阶段按 image_auto_optimize 配置决定是否自动优化（补全缺失标签 + 自动加权核心特征）。

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
            reason = getattr(self.tagger, "_last_error", "") or ""
            suffix = f"（{reason}）" if reason else ""
            return None, f"图片识别失败{suffix}，请检查图片清晰度后重试。"
        prompt = f"标签: {', '.join(tags)}\n描述: {description or '（无）'}"
        system_prompt = TAGGER_ORGANIZE_SYSTEM_PROMPT
        if not self._cfg_bool("image_auto_optimize", True):
            # 关闭自动优化：仅忠实整理反推标签，不补全、不自动加权核心特征；
            # 但用户描述中的强调/弱化意图仍按 weights 规则生效
            system_prompt += "\n\n本次请勿补全缺失标签、勿自动加权核心特征，仅忠实整理反推标签；但用户描述中的强调/弱化意图仍按规则填写 weights。"
        parsed = await self._parse_with_llm(event, prompt, system_prompt=system_prompt)
        if parsed is None:
            return None, "提示词整理失败，请稍后重试。"
        return parsed, None

    async def _lookup_characters(self, parsed: ParsedRequest):
        """查询每个角色的 DanbooruSearch 结果。

        Args:
            parsed: LLM 解析结果

        Returns:
            与 parsed.characters 等长的结果列表，未命中项为 None
        """
        if self.lookup is None:
            return [None] * len(parsed.characters)
        final_level = resolve_nsfw_level(parsed, self._cfg_bool("allow_adult_prompts", True))
        show_nsfw = final_level == "explicit"

        async def limited(item) -> CharacterTags | None:
            async with self._lookup_sem:
                return await self.lookup.lookup(item, show_nsfw)

        results = await asyncio.gather(*(limited(item) for item in parsed.characters), return_exceptions=True)
        return [r if isinstance(r, CharacterTags) else None for r in results]

    async def _filter_conflicting_tags(
        self, event: AstrMessageEvent, parsed: ParsedRequest, lookup_results: list[CharacterTags | None]
    ) -> list[CharacterTags | None]:
        """通过 LLM 过滤角色关联标签中与用户指定服装/动作冲突的默认标签。

        仅当用户明确指定了 outfit_tags 或 action_tags 时触发。
        按角色分别调用 LLM，并行执行；失败时降级为清空关联标签（仅保留 canonical）。

        Args:
            event: 消息事件
            parsed: LLM 解析结果，包含 outfit_tags 和 action_tags
            lookup_results: DanbooruSearch 查询结果列表，与 parsed.characters 等长（未命中为 None）

        Returns:
            过滤后的结果列表（未命中项保持 None）
        """
        if not parsed.outfit_tags and not parsed.action_tags:
            return lookup_results

        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if not provider:
            logger.warning("[NAIPrompt] 无可用 LLM provider，标签冲突过滤降级")
            return [
                CharacterTags(r.display_name, r.canonical_tag, [], r.copyright_tag) if r is not None else None
                for r in lookup_results
            ]

        async def filter_one(result: CharacterTags | None) -> CharacterTags | None:
            if result is None:
                return None
            prompt = json.dumps({
                "outfit_tags": parsed.outfit_tags,
                "action_tags": parsed.action_tags,
                "character_tags": result.tags,
            }, ensure_ascii=False)
            try:
                async with self._lookup_sem:
                    response = await provider.text_chat(prompt=prompt, contexts=[], system_prompt=CONFLICT_FILTER_SYSTEM_PROMPT)
                text = getattr(response, "completion_text", "") if response else ""
                data = extract_json(text)
                if data and isinstance(data.get("filtered_tags"), list):
                    filtered = [str(t) for t in data["filtered_tags"] if isinstance(t, str)]
                    if filtered:
                        logger.info("[NAIPrompt] 角色 %s 标签冲突过滤完成: %d -> %d", result.display_name, len(result.tags), len(filtered))
                        return CharacterTags(result.display_name, result.canonical_tag, filtered, result.copyright_tag)
            except Exception as exc:
                logger.warning("[NAIPrompt] 角色 %s 标签冲突过滤异常: %s", result.display_name, exc)
            # 降级：清空关联标签，仅保留 canonical
            return CharacterTags(result.display_name, result.canonical_tag, [], result.copyright_tag)

        results = await asyncio.gather(*(filter_one(r) for r in lookup_results), return_exceptions=True)
        return [r if isinstance(r, CharacterTags) else None for r in results]

    @staticmethod
    def _save_temp_image(image_bytes: bytes) -> str | None:
        """将图片字节写入临时文件并返回路径。

        Args:
            image_bytes: 图片字节

        Returns:
            临时文件路径；失败返回 None
        """
        suffix = ".jpg" if image_bytes[:3] == b"\xff\xd8\xff" else ".png"
        try:
            fd, path = tempfile.mkstemp(prefix="nai_prompt_", suffix=suffix)
            with os.fdopen(fd, "wb") as handle:
                handle.write(image_bytes)
            return path
        except OSError:
            return None

    async def _generate_images(self, event: AstrMessageEvent, positive: str):
        """按配置生成示例图并通过图片消息回传；失败静默降级。

        Args:
            event: 消息事件
            positive: 正面提示词（生图 prompt）

        Yields:
            图片消息结果
        """
        if not positive or self.generator is None:
            return
        count = self._cfg_int("image_count", 1, 1, 4)
        images = await self.generator.generate(positive, count)
        for image_bytes in images:
            path = self._save_temp_image(image_bytes)
            if path:
                yield event.image_result(path)

    def _remember_prompt(self, text: str, event: AstrMessageEvent) -> None:
        """记录最近输出的提示词，供引用溯源匹配使用。"""
        if not text:
            return
        session_id = str(getattr(event, "unified_msg_origin", "") or "")
        self._recent_prompts.append((time.monotonic(), session_id, text))

    def _match_recent_prompt(self, text: str, session_id: str) -> bool:
        """在最近输出缓冲区中查找与 text 双向包含匹配的记录。

        Args:
            text: 被引用的提示词文本
            session_id: 当前会话标识（可能为空）

        Returns:
            命中返回 True，否则 False
        """
        now = time.monotonic()
        normalized = _normalize_prompt_text(text)
        if len(normalized) < 20:
            return False
        matched_ts: float | None = None
        for ts, record_session, stored in self._recent_prompts:
            if now - ts > 600:
                continue
            if record_session and session_id and record_session != session_id:
                continue
            stored_normalized = _normalize_prompt_text(stored)
            if normalized in stored_normalized or stored_normalized in normalized:
                if matched_ts is None or ts > matched_ts:
                    matched_ts = ts
        return matched_ts is not None

    async def _reply_prompt_source(self, event: AstrMessageEvent) -> str | None:
        """从消息链中提取被引用回复的提示词文本。

        优先以"被引用消息发送者是机器人自己"（self_id 可用时）为准；
        self_id 缺失时回退为内容溯源匹配（命中最近输出缓冲区）。

        Args:
            event: 消息事件

        Returns:
            被引用的提示词文本；无有效引用返回 None
        """
        try:
            self_id = str(getattr(getattr(event, "message_obj", None), "self_id", "") or "")
        except Exception:
            self_id = ""
        session_id = str(getattr(event, "unified_msg_origin", "") or "")
        for component in self._message_components(event):
            if component.__class__.__name__.lower() != "reply":
                continue
            sender_id = str(getattr(component, "sender_id", "") or "")
            message_str = str(getattr(component, "message_str", "") or getattr(component, "text", "") or "")
            if not message_str or not looks_like_prompt(str(message_str)):
                continue
            # 被引用消息的会话标识（部分适配器提供），不一致时拒绝跨会话引用
            quoted_session = str(
                getattr(component, "group_id", "") or getattr(component, "session_id", "") or ""
            )
            if self_id and sender_id and sender_id == self_id:
                if quoted_session and session_id and quoted_session != session_id:
                    continue
                return str(message_str)
            if not self_id and self._match_recent_prompt(str(message_str), session_id):
                return str(message_str)
        return None

    async def _edit_prompt(self, event: AstrMessageEvent, quoted_prompt: str, instruction: str):
        """基于被引用提示词与修改指令，通过 LLM 输出修改后的提示词并联动示例图。

        Args:
            event: 消息事件
            quoted_prompt: 被引用的原提示词
            instruction: 修改指令

        Yields:
            修改后的提示词结果（及可选的示例图）
        """
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if not provider:
            yield event.plain_result("当前无可用的 LLM 服务，无法进行增量修改。")
            return
        prompt = f"原提示词: {quoted_prompt}\n修改指令: {instruction}"
        try:
            response = await provider.text_chat(prompt=prompt, contexts=[], system_prompt=INCREMENTAL_EDIT_SYSTEM_PROMPT)
            text = getattr(response, "completion_text", "") if response else ""
        except Exception as exc:
            logger.warning("[NAIPrompt] 增量修改失败: %s", exc)
            text = ""
        text = (text or "").strip()
        # 去除可能的 Markdown 代码块包裹
        if text.startswith("```"):
            text = text.strip("`").strip()
        if not text:
            yield event.plain_result("增量修改失败，请稍后重试。")
            return
        self._remember_prompt(text, event)
        yield event.plain_result(text)
        if self._cfg_bool("enable_image_generation", False):
            async for image_result in self._generate_images(event, text):
                yield image_result

    @filter.command("提示词")
    async def prompt_command(self, event: AstrMessageEvent, description: str = ""):
        """/提示词 <自然语言描述>：生成可复制的 NAI 正负面提示词；附带图片时反推图片并整理提示词。"""
        description = str(description or "").strip()
        clean_description = self._strip_image_url(description)
        # 增量修改：检测到引用机器人提示词且带修改指令时，进入修改模式
        quoted_prompt = await self._reply_prompt_source(event)
        if quoted_prompt and clean_description:
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
            async for item in self._edit_prompt(event, quoted_prompt, clean_description):
                yield item
            return
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
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        result = await build_prompt(
            parsed=parsed,
            lookup_results=characters,
            allow_adult=self._cfg_bool("allow_adult_prompts", True),
            max_length=self._cfg_int("max_prompt_length", 0, 0, 5000),
            provider=provider,
        )
        text_result = format_result(result)
        yield event.plain_result(text_result)
        self._remember_prompt(text_result, event)
        # 按配置生成示例图；生图失败静默降级，不影响已返回的文本提示词
        if self._cfg_bool("enable_image_generation", False):
            async for image_result in self._generate_images(event, result.positive):
                yield image_result
