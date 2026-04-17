import asyncio
import copy
import json
import os
import shutil
import time
from typing import Any, Callable, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp
from astrbot.api.provider import LLMResponse
from astrbot.core.message.message_event_result import MessageEventResult

from .provider_utils import find_provider


_RECORD_ASR_CACHE_TTL_SEC = 10 * 60
_RECORD_ASR_CACHE_MAX_SIZE = 100
_record_asr_cache: dict[str, dict[str, Any]] = {}


def normalize_blocked_command_text(text: str) -> str:
    """标准化阻断命令文本：去边界空白、折叠多空白、转小写。"""
    return " ".join(str(text or "").strip().lower().split())

def is_valid_blocked_command_match(text: str, command: str) -> bool:
    """严格匹配命令（全文一致或命令后跟空格）。"""
    if not text.startswith(command):
        return False
    if len(text) == len(command):
        return True
    return text[len(command)] == " "

def normalize_blocked_command_values(commands: Any) -> list[str]:
    """标准化并去重（保留原顺序）的阻断命令列表。"""
    normalized: list[str] = []
    for raw_cmd in commands or []:
        if not isinstance(raw_cmd, str):
            continue
        normalized_cmd = normalize_blocked_command_text(raw_cmd)
        if normalized_cmd:
            normalized.append(normalized_cmd)

    deduped: list[str] = []
    seen: set[str] = set()
    for cmd in normalized:
        if cmd in seen:
            continue
        seen.add(cmd)
        deduped.append(cmd)
    return deduped

def get_blocked_commands(get_cfg: Callable[[str, Any], Any]) -> list[str]:
    """读取并标准化阻断命令列表配置。"""
    blocked_commands = get_cfg("blocked_commands")
    if isinstance(blocked_commands, (list, tuple, set)):
        return normalize_blocked_command_values(blocked_commands)
    return []

def match_blocked_command(msg: str, blocked_commands: list[str]) -> str:
    """返回命中命令；未命中返回空。"""
    normalized_msg = normalize_blocked_command_text(msg)
    if not normalized_msg:
        return ""

    for normalized_cmd in blocked_commands:
        if is_valid_blocked_command_match(normalized_msg, normalized_cmd):
            return normalized_cmd
    return ""


class EffectiveDialogHistory:
    def __init__(self, max_turns: int = 3):
        self._max_turns = max(1, int(max_turns))
        self._turns_by_session: dict[str, list[dict[str, Any]]] = {}

    @staticmethod
    def normalize_text(text: Any) -> str:
        normalized = " ".join(str(text or "").replace("\u200b", " ").split())
        return normalized.strip()

    def outline_components(self, components: list[Any]) -> str:
        parts: list[str] = []
        for comp in components or []:
            seg_type = ""
            if isinstance(comp, dict):
                seg_type = str(comp.get("type") or "").strip().lower()
                data = comp.get("data") or {}
                if seg_type == "text":
                    text = self.normalize_text((data or {}).get("text"))
                    if text:
                        parts.append(text)
                    continue
                if seg_type == "image":
                    parts.append("[图片]")
                elif seg_type == "video":
                    parts.append("[视频]")
                elif seg_type == "file":
                    name = self.normalize_text((data or {}).get("name"))
                    parts.append(f"[文件:{name}]" if name else "[文件]")
                elif seg_type == "forward":
                    parts.append("[合并转发]")
                elif seg_type == "json":
                    parts.append("[JSON]")
                elif seg_type == "record":
                    parts.append("[语音]")
                continue

            if isinstance(comp, Comp.Plain):
                text = self.normalize_text(getattr(comp, "text", ""))
                if text:
                    parts.append(text)
            elif isinstance(comp, Comp.Image):
                parts.append("[图片]")
            elif isinstance(comp, Comp.Video):
                parts.append("[视频]")
            elif isinstance(comp, Comp.File):
                name = self.normalize_text(getattr(comp, "name", ""))
                parts.append(f"[文件:{name}]" if name else "[文件]")
            elif isinstance(comp, Comp.Forward):
                parts.append("[合并转发]")
            elif isinstance(comp, Comp.Json):
                parts.append("[JSON]")
            elif isinstance(comp, Comp.Record):
                parts.append("[语音]")
        return self.normalize_text(" ".join(parts))

    def build_user_text(self, msg: str, components: list[Any]) -> str:
        text = self.normalize_text(msg)
        if text:
            return text
        return self.outline_components(components)

    def extract_assistant_text(self, event: AstrMessageEvent) -> str:
        result = event.get_result()
        chain = list(getattr(result, "chain", None) or [])
        if not chain:
            return ""

        parts: list[str] = []
        for comp in chain:
            if isinstance(comp, (Comp.Reply, Comp.At, Comp.AtAll)):
                continue
            if isinstance(comp, Comp.Plain):
                text = self.normalize_text(getattr(comp, "text", ""))
                if text:
                    parts.append(text)
            else:
                comp_type = str(getattr(comp, "type", "") or comp.__class__.__name__).strip()
                if comp_type:
                    parts.append(f"[{comp_type}]")
        return self.normalize_text(" ".join(parts))

    def extract_assistant_text_from_response(self, resp: LLMResponse) -> str:
        if resp.result_chain and getattr(resp.result_chain, "chain", None):
            parts: list[str] = []
            for comp in list(resp.result_chain.chain or []):
                if isinstance(comp, Comp.Plain):
                    text = self.normalize_text(getattr(comp, "text", ""))
                    if text:
                        parts.append(text)
                else:
                    comp_type = str(getattr(comp, "type", "") or comp.__class__.__name__).strip()
                    if comp_type:
                        parts.append(f"[{comp_type}]")
            normalized = self.normalize_text(" ".join(parts))
            if normalized:
                return normalized
        return self.normalize_text(getattr(resp, "completion_text", ""))

    async def get_history_messages(
        self,
        event: AstrMessageEvent,
        count: int,
    ) -> list[str]:
        turns = self._turns_by_session.get(event.unified_msg_origin, [])
        if not turns:
            return []

        contexts: list[str] = []
        for turn in turns:
            user_text = self.normalize_text(turn.get("user"))
            assistant_text = self.normalize_text(turn.get("assistant"))
            user_name = self.normalize_text(turn.get("user_name"))
            assistant_name = self.normalize_text(turn.get("assistant_name"))
            if user_text:
                if user_name:
                    contexts.append(f"{user_name}: {user_text}")
                else:
                    contexts.append(f"user: {user_text}")
            if assistant_text:
                if assistant_name:
                    contexts.append(f"{assistant_name}: {assistant_text}")
                else:
                    contexts.append(f"assistant: {assistant_text}")
        return contexts[-count:] if count else contexts

    def clear_session(self, umo: str) -> int:
        if not umo:
            return 0
        turns = self._turns_by_session.pop(umo, None)
        return len(turns) if turns else 0

    def append_turn(
        self,
        umo: str,
        user_text: str,
        assistant_text: str,
        user_name: str = "",
        assistant_name: str = "",
    ) -> None:
        user_text = self.normalize_text(user_text)
        assistant_text = self.normalize_text(assistant_text)
        if (not umo) or (not user_text) or (not assistant_text):
            return

        user_name = self.normalize_text(user_name)
        assistant_name = self.normalize_text(assistant_name)
        turns = self._turns_by_session.setdefault(umo, [])
        turns.append(
            {
                "user": user_text,
                "assistant": assistant_text,
                "user_name": user_name,
                "assistant_name": assistant_name,
                "ts": time.time(),
            }
        )
        self._turns_by_session[umo] = turns[-self._max_turns:]


def clear_effective_dialog_history(history: EffectiveDialogHistory, umo: str) -> int:
    if not history:
        return 0
    return history.clear_session(umo)


def resolve_provider(context: Any, provider_id: str):
    """按配置的 id 或名称查找可用 Provider。"""
    return find_provider(context, provider_id)


def get_llm_provider(
    context: Any,
    event: Optional[AstrMessageEvent] = None,
    umo: Optional[str] = None,
):
    """获取当前会话使用的 LLM Provider。"""
    try:
        using_umo = umo or (getattr(event, "unified_msg_origin", None) if event is not None else None)
        if using_umo:
            return context.get_using_provider(umo=using_umo)
        return context.get_using_provider()
    except Exception:
        return None


def get_stt_provider(
    context: Any,
    get_cfg: Callable[[str, Any], Any],
    event: Optional[AstrMessageEvent] = None,
    umo: Optional[str] = None,
):
    """获取配置的 STT Provider，未命中时回退到当前会话 Provider。"""
    asr_pid = get_cfg("asr_provider_id")
    p = resolve_provider(context, asr_pid)
    if p:
        logger.debug(f"[LLMEnhancement] 成功匹配到指定 STT Provider: {asr_pid}")
        return p

    if asr_pid:
        logger.warning(f"[LLMEnhancement] 未找到指定 STT Provider: {asr_pid}")

    try:
        using_umo = umo or (getattr(event, "unified_msg_origin", None) if event is not None else None)
        if using_umo:
            return context.get_using_stt_provider(umo=using_umo)
        return context.get_using_stt_provider()
    except Exception:
        return None


def get_vision_provider(
    context: Any,
    get_cfg: Callable[[str, Any], Any],
    event: Optional[AstrMessageEvent] = None,
    umo: Optional[str] = None,
):
    """获取配置的视觉 Provider，未命中时回退到当前会话 LLM Provider。"""
    image_pid = get_cfg("image_provider_id")
    p = resolve_provider(context, image_pid)
    if p:
        logger.debug(f"[LLMEnhancement] 成功匹配到指定 Vision Provider: {image_pid}")
        return p

    if image_pid:
        logger.warning(f"[LLMEnhancement] 未找到指定 Vision Provider: {image_pid}，回退到会话 Provider")

    return get_llm_provider(context=context, event=event, umo=umo)


def _build_record_cache_key(segment: Any) -> str:
    if isinstance(segment, dict):
        data = segment.get("data") or {}
        if isinstance(data, dict):
            for key in ("file_id", "file", "url", "path", "file_path", "id"):
                value = data.get(key)
                if value not in (None, ""):
                    return str(value)
        for key in ("file_id", "file", "url", "path", "file_path", "id"):
            value = segment.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    for attr in ("file_id", "file", "url", "path", "file_path", "id"):
        value = getattr(segment, attr, None)
        if value not in (None, ""):
            return str(value)
    return ""


def _get_record_asr_cache(cache_key: str) -> str:
    if not cache_key:
        return ""
    item = _record_asr_cache.get(cache_key)
    if not isinstance(item, dict):
        return ""
    expire_at = float(item.get("expire_at", 0.0) or 0.0)
    if expire_at <= time.time():
        _record_asr_cache.pop(cache_key, None)
        return ""
    return str(item.get("text") or "")


def _set_record_asr_cache(cache_key: str, text: str) -> None:
    if not cache_key:
        return
    normalized = str(text or "").strip()
    if not normalized:
        return
    _record_asr_cache[cache_key] = {
        "text": normalized,
        "expire_at": time.time() + _RECORD_ASR_CACHE_TTL_SEC,
    }
    if len(_record_asr_cache) > _RECORD_ASR_CACHE_MAX_SIZE:
        now_ts = time.time()
        expired_keys = [
            k
            for k, v in _record_asr_cache.items()
            if float((v or {}).get("expire_at", 0.0) or 0.0) <= now_ts
        ]
        for k in expired_keys:
            _record_asr_cache.pop(k, None)
        if len(_record_asr_cache) > _RECORD_ASR_CACHE_MAX_SIZE:
            _record_asr_cache.clear()


def _normalize_record_source(value: Any) -> str:
    return str(value or "").strip()


async def resolve_record_file_path(
    event: Optional[AstrMessageEvent],
    segment: Any,
) -> tuple[Optional[str], bool]:
    """将语音 segment 解析为本地文件路径，返回 `(path, should_cleanup)`。"""
    def _build_candidates(data: dict) -> list[str]:
        candidates: list[str] = []
        url = _normalize_record_source(data.get("url"))
        if url:
            candidates.append(url)
        file_value = _normalize_record_source(data.get("file"))
        if file_value:
            candidates.append(file_value)
        path_value = _normalize_record_source(data.get("path"))
        if path_value:
            candidates.append(path_value)
        file_id = _normalize_record_source(data.get("file_id"))
        if file_id:
            candidates.append(file_id)
        return candidates

    async def _resolve_from_candidates(candidates: list[str]) -> tuple[Optional[str], bool]:
        for cand in candidates:
            if cand.startswith("http"):
                try:
                    record = Comp.Record.fromURL(cand)
                    path = await record.convert_to_file_path()
                    return path, True
                except Exception:
                    continue
            if cand.startswith("file://"):
                try:
                    record = Comp.Record(file=cand)
                    path = await record.convert_to_file_path()
                    return path, False
                except Exception:
                    continue
            if os.path.exists(cand):
                return os.path.abspath(cand), False

        # 兜底：针对不是本地路径的 file id，尝试调用平台 get_record 获取文件
        if candidates and event is not None and getattr(event, "bot", None) is not None:
            api = getattr(event.bot, "api", None)
            if api and hasattr(api, "call_action"):
                try:
                    resp = await api.call_action("get_record", file=candidates[0])
                    if isinstance(resp, dict):
                        resp_file = _normalize_record_source(resp.get("file"))
                        resp_url = _normalize_record_source(resp.get("url"))
                        if resp_file and os.path.exists(resp_file):
                            return os.path.abspath(resp_file), False
                        if resp_url:
                            record = Comp.Record.fromURL(resp_url)
                            path = await record.convert_to_file_path()
                            return path, True
                except Exception:
                    pass
        return None, False

    try:
        if isinstance(segment, Comp.Record):
            data = {
                "file": getattr(segment, "file", None),
                "url": getattr(segment, "url", None),
                "path": getattr(segment, "path", None) or getattr(segment, "file_path", None),
                "file_id": getattr(segment, "file_id", None),
            }
            candidates = _build_candidates(data)
            if not candidates:
                return None, False
            return await _resolve_from_candidates(candidates)

        if not isinstance(segment, dict):
            return None, False
        seg_type = str(segment.get("type") or "").lower()
        if seg_type != "record":
            return None, False
        data = segment.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        candidates = _build_candidates(data)
        if not candidates:
            return None, False
        return await _resolve_from_candidates(candidates)
    except Exception:
        return None, False
    return None, False


async def transcribe_record_segment(
    *,
    context: Any,
    get_cfg: Callable[[str, Any], Any],
    event: Optional[AstrMessageEvent],
    segment: Any,
) -> tuple[str, list[str]]:
    """尽可能转写单个语音 segment，返回 `(text, cleanup_paths)`。"""
    if not bool(get_cfg("record_parse_enable", True)):
        return "", []
    cache_key = _build_record_cache_key(segment)
    cached_text = _get_record_asr_cache(cache_key)
    if cached_text:
        return cached_text, []
    stt = get_stt_provider(context, get_cfg, event=event)
    if not stt:
        return "", []
    record_path, should_cleanup = await resolve_record_file_path(event, segment)
    if not record_path:
        return "", []
    text = ""
    try:
        if hasattr(stt, "get_text"):
            text = await stt.get_text(record_path)
        elif hasattr(stt, "speech_to_text"):
            res = await stt.speech_to_text(record_path)
            if isinstance(res, dict):
                text = str(res.get("text") or "")
            elif hasattr(res, "text"):
                text = str(res.text)
            else:
                text = str(res or "")
        else:
            text = ""
    except Exception as e:
        text = ""
    cleanup_paths = [record_path] if should_cleanup else []
    text = str(text or "").strip()
    if text:
        _set_record_asr_cache(cache_key or record_path, text)
    return text, cleanup_paths


async def transcribe_record_from_chain(
    *,
    context: Any,
    get_cfg: Callable[[str, Any], Any],
    event: Optional[AstrMessageEvent],
    chain: Any,
) -> tuple[str, list[str]]:
    """在消息链中查找第一个语音 segment 并执行转写。"""
    for seg in list(chain or []):
        if isinstance(seg, Comp.Record) or (
            isinstance(seg, dict) and str(seg.get("type") or "").lower() == "record"
        ):
            text, cleanup_paths = await transcribe_record_segment(
                context=context,
                get_cfg=get_cfg,
                event=event,
                segment=seg,
            )
            return text, cleanup_paths
    return "", []


async def cleanup_paths_later(cleanup_paths: list[str], delay_sec: int = 120):
    """延迟清理临时路径，采用 fire-and-forget 方式执行。"""
    if not cleanup_paths:
        return

    snapshot_paths = [p for p in dict.fromkeys(cleanup_paths) if p]
    if not snapshot_paths:
        return

    async def final_cleanup(paths: list[str]):
        await asyncio.sleep(delay_sec)
        for p in paths:
            try:
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                logger.debug(f"[LLMEnhancement] cleanup path 失败: {p}, err={e}")

    asyncio.create_task(final_cleanup(snapshot_paths))


def _history_segment_to_text(seg: Any) -> str:
    if isinstance(seg, str):
        return seg
    if not isinstance(seg, dict):
        return ""

    seg_type = str(seg.get("type") or "").strip().lower()
    data = seg.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    if seg_type == "text":
        return str(data.get("text") or seg.get("text") or "")
    if seg_type == "at":
        qq = str(data.get("qq") or seg.get("qq") or "").strip()
        return "@全体成员" if qq.lower() == "all" else (f"@{qq}" if qq else "@")
    if seg_type == "reply":
        rid = str(data.get("id") or seg.get("id") or "").strip()
        return f"[回复:{rid}]" if rid else "[回复]"
    if seg_type == "image":
        return "[图片]"
    if seg_type == "video":
        return "[视频]"
    if seg_type == "record":
        return "[语音]"
    if seg_type == "file":
        name = str(data.get("name") or seg.get("name") or "").strip()
        return f"[文件:{name}]" if name else "[文件]"
    if seg_type == "face":
        return "[表情]"
    if seg_type == "json":
        return "[JSON]"
    if seg_type == "xml":
        return "[XML]"
    if seg_type == "forward":
        return "[合并转发]"
    return ""


def _history_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(_history_segment_to_text(seg) for seg in content).strip()
    if isinstance(content, dict):
        return _history_segment_to_text(content).strip()
    return str(content or "").strip()


async def get_history_messages(
    context: Any,
    event: AstrMessageEvent,
    role: str | None = "assistant",
    count: int | None = 0,
    with_role_prefix: bool = False,
) -> list[str]:
    """按角色读取当前会话历史消息内容。"""
    try:
        umo = event.unified_msg_origin
        curr_cid = await context.conversation_manager.get_curr_conversation_id(umo)
        if not curr_cid:
            return []
        conversation = await context.conversation_manager.get_conversation(umo, curr_cid)
        if not conversation:
            return []

        history = json.loads(conversation.history or "[]")
        role_filter = str(role or "").strip().lower()
        allow_all_roles = role is None or role_filter in {"", "*", "all"}

        contexts: list[str] = []
        for record in history:
            rec_role = str(record.get("role") or "").strip().lower()
            if (not allow_all_roles) and rec_role != role_filter:
                continue

            text = _history_content_to_text(record.get("content"))
            if not text:
                continue

            if with_role_prefix:
                role_label = rec_role or "unknown"
                contexts.append(f"{role_label}: {text}")
            else:
                contexts.append(text)

        return contexts[-count:] if count else contexts
    except Exception as e:
        logger.error(f"获取历史消息失败：{e}")
        return []





def get_discarded_response_ttl_sec(
    get_cfg: Callable[[str, Any], Any],
    default_ttl_sec: float,
) -> float:
    raw = get_cfg("dynamic_discarded_response_ttl_sec", default_ttl_sec)
    try:
        value = float(raw)
    except Exception:
        value = default_ttl_sec
    return max(30.0, min(1800.0, value))


def clear_discarded_response_cache(member: Any) -> None:
    member.dynamic_discarded_response_cache = {}


def get_valid_discarded_response_cache(member: Any, ttl_sec: float) -> Optional[dict[str, Any]]:
    cache = member.dynamic_discarded_response_cache or {}
    if not isinstance(cache, dict) or not cache:
        return None

    ts = float(cache.get("ts", 0.0) or 0.0)
    if ts <= 0.0 or (time.time() - ts) > ttl_sec:
        clear_discarded_response_cache(member)
        return None

    chain = cache.get("chain") or []
    if not isinstance(chain, list) or not chain:
        clear_discarded_response_cache(member)
        return None

    return cache


def is_chain_effectively_empty(chain: list[Any]) -> bool:
    if not chain:
        return True
    for comp in chain:
        if isinstance(comp, Comp.Plain):
            if str(getattr(comp, "text", "") or "").strip():
                return False
            continue
        return False
    return True


def extract_chain_from_llm_response(resp: LLMResponse) -> list[Any]:
    chain: list[Any] = []
    if resp.result_chain and getattr(resp.result_chain, "chain", None):
        chain = list(resp.result_chain.chain or [])
    elif str(resp.completion_text or "").strip():
        chain = [Comp.Plain(text=str(resp.completion_text or "").strip())]

    if not chain:
        return []

    filtered: list[Any] = []
    for comp in chain:
        if isinstance(comp, Comp.Plain) and (not str(getattr(comp, "text", "") or "").strip()):
            continue
        filtered.append(comp)
    if not filtered:
        return []

    return copy.deepcopy(filtered)


def store_discarded_response_cache(member: Any, response_seq: int, resp: LLMResponse) -> bool:
    chain = extract_chain_from_llm_response(resp)
    if not chain:
        return False
    incoming_seq = int(response_seq or 0)
    existing_cache = member.dynamic_discarded_response_cache or {}
    if isinstance(existing_cache, dict) and existing_cache:
        try:
            existing_seq = int(existing_cache.get("seq", 0) or 0)
        except Exception:
            existing_seq = 0
        # Keep the latest discarded response by sequence to avoid stale overwrite.
        if existing_seq > incoming_seq:
            return False
    member.dynamic_discarded_response_cache = {
        "ts": time.time(),
        "seq": incoming_seq,
        "chain": chain,
    }
    return True


def is_llm_response_empty_without_tool(resp: LLMResponse) -> bool:
    if resp.tools_call_name or resp.tools_call_args:
        return False
    if resp.result_chain and getattr(resp.result_chain, "chain", None):
        return is_chain_effectively_empty(list(resp.result_chain.chain or []))
    return not str(resp.completion_text or "").strip()


def looks_like_error_result(chain: list[Any]) -> bool:
    plain_text = "".join(
        str(getattr(comp, "text", "") or "")
        for comp in chain
        if isinstance(comp, Comp.Plain)
    ).strip()
    if not plain_text:
        return False

    normalized = " ".join(plain_text.split())
    lowered = normalized.lower()

    def _starts_with_error_header(text: str, header: str) -> bool:
        return (
            text == header
            or text.startswith(f"{header}:")
            or text.startswith(f"{header}：")
        )

    if _starts_with_error_header(normalized, "AstrBot 请求失败"):
        return True
    if _starts_with_error_header(normalized, "LLM 响应错误"):
        return True
    if _starts_with_error_header(lowered, "error occurred while processing agent request"):
        return True
    return False


async def apply_discarded_response_fallback(
    event: AstrMessageEvent,
    member: Any,
    gid: str,
    uid: str,
    reason: str,
    ttl_sec: float,
) -> bool:
    cache: Optional[dict[str, Any]] = None
    async with member.lock:
        cache = get_valid_discarded_response_cache(member, ttl_sec=ttl_sec)
    if not cache:
        return False

    cached_chain = copy.deepcopy(cache.get("chain") or [])
    if not cached_chain:
        return False

    result = event.get_result()
    if result is None:
        event.set_result(MessageEventResult(chain=cached_chain))
    else:
        result.chain = cached_chain
        event.set_result(result)

    event.set_extra("_llme_discard_cache_clear_after_sent", True)
    event.set_extra("_llme_discard_cache_used", True)
    logger.info(
        "[LLMEnhancement] 使用被丢弃响应缓存进行回退："
        f"group={gid or 'private'}, uid={uid}, reason={reason}, cached_seq={int(cache.get('seq') or 0)}"
    )
    return True
