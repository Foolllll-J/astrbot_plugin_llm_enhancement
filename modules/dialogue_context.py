import asyncio
import datetime
import json
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Callable, Optional
from urllib.parse import unquote

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp
from astrbot.api.provider import ProviderRequest
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from .state_manager import GroupState, StateManager
from .runtime_helpers import (
    _fetch_messages_by_ids,
    _is_unavailable_get_msg_payload,
    _normalize_emoji_summary,
    cleanup_paths_later,
    transcribe_record_from_chain,
    clear_effective_dialog_history,
    append_text_part_to_request,
)
from .wake_logic import build_media_trigger_message

_CONTEXT_IMAGE_CAPTION_PROMPT = "请用中文描述这张图片。包含：主体、数量、位置关系、动作、场景背景、画面中的文字内容、情绪氛围。控制在120字以内。如果识图异常请直接回复【处理失败】。"
_NON_TEXT_PARSE_OPTIONS = {"image", "forward", "url", "file", "json", "record"}
_LAST_USER_INTERACTION_TTL_SEC = 24 * 60 * 60
_CONTEXT_IMAGE_CAPTION_CACHE_TTL_SEC = 86400
_CONTEXT_IMAGE_CAPTION_CACHE_MAX_SIZE = 100
_context_image_caption_cache: dict[str, dict[str, Any]] = {}
_context_image_caption_semaphore = asyncio.Semaphore(3)

_CONTEXT_EMOJI_CAPTION_CACHE_TTL_SEC = 86400
_CONTEXT_EMOJI_CAPTION_CACHE_MAX_SIZE = 200
_context_emoji_cache: dict[str, dict[str, Any]] = {}
_CONTEXT_EMOJI_CAPTION_PROMPT = "请识别这个表情包/贴图传达的情感和含义，关注表情、动作、画面中的文字和使用场景，用中文简要描述。控制在60字以内。如果识图异常请直接回复【处理失败】。"


def is_context_injection_enabled(get_cfg: Any) -> bool:
    """判断上下文注入功能是否已启用。"""
    mode = str(get_cfg("context_injection_mode", "off") or "off").strip().lower()
    return mode in {"before_active_only", "before_all_wake"}


def get_context_injection_mode(get_cfg: Any) -> str:
    """获取上下文注入模式（off / before_active_only / before_all_wake）。"""
    mode = str(get_cfg("context_injection_mode", "off") or "off").strip().lower()
    if mode not in {"off", "before_active_only", "before_all_wake"}:
        return "off"
    return mode


def get_context_injection_format(get_cfg: Any) -> str:
    """获取上下文注入格式（simple / detailed）。"""
    fmt = str(get_cfg("context_injection_format", "simple") or "simple").strip().lower()
    if fmt not in {"simple", "detailed"}:
        return "simple"
    return fmt


def get_context_injection_max_messages(get_cfg: Any) -> int:
    """获取上下文注入的最大消息数。"""
    raw = get_cfg("context_injection_max_messages", 20)
    try:
        value = int(raw)
    except Exception:
        value = 20
    return max(0, value)


def get_wake_judge_context_count(get_cfg: Any) -> int:
    """获取唤醒判断时使用的上下文消息数量。"""
    return max(0, int(get_cfg("wake_judge_context_count", 10)))


def prune_group_context_state(
    *,
    group_state: GroupState,
    get_cfg: Any,
    now_ts: Optional[float] = None,
) -> None:
    """裁剪群组上下文状态：限制消息数量并清理过期的用户交互记录。"""
    max_len = get_context_injection_max_messages(get_cfg)
    ts_now = float(now_ts if now_ts is not None else time.time())

    messages = list(group_state.context_messages or [])
    if max_len <= 0:
        group_state.context_messages = []
        group_state.last_user_interaction = {}
        return
    if len(messages) > max_len:
        messages = messages[-max_len:]
    group_state.context_messages = messages

    # 同步清理 last_user_interaction，避免长期累积。
    valid_uids: set[str] = set()
    for item in messages:
        uid = str(item.get("uid") or "").strip()
        if uid:
            valid_uids.add(uid)

    cleaned_last_interaction: dict[str, float] = {}
    for uid, ts in dict(group_state.last_user_interaction or {}).items():
        uid_text = str(uid or "").strip()
        if not uid_text:
            continue
        if uid_text not in valid_uids:
            continue
        try:
            ts_value = float(ts or 0.0)
        except Exception:
            continue
        if ts_value <= 0:
            continue
        if (ts_now - ts_value) > _LAST_USER_INTERACTION_TTL_SEC:
            continue
        cleaned_last_interaction[uid_text] = ts_value
    group_state.last_user_interaction = cleaned_last_interaction


def get_context_non_text_parse_options(get_cfg: Any) -> set[str]:
    """获取非文本上下文注入的解析选项集合（url / json / file 等）。"""
    raw = get_cfg("context_injection_non_text_parse_options", [])
    if not isinstance(raw, list):
        return set()
    options: set[str] = set()
    for item in raw:
        v = str(item or "").strip().lower()
        if v in _NON_TEXT_PARSE_OPTIONS:
            options.add(v)
    return options


def get_wake_history_messages(
    *,
    group_state: GroupState,
    get_cfg: Any,
    count: int = 0,
) -> list[str]:
    """
    为唤醒判定/唤醒延长提供上下文（固定使用简洁样式）。
    内部会约束：wake_judge_context_count <= context_injection_max_messages。
    """
    prune_group_context_state(group_state=group_state, get_cfg=get_cfg)
    context_messages = list(group_state.context_messages or [])
    if not context_messages:
        return []

    limit = min(
        get_wake_judge_context_count(get_cfg),
        get_context_injection_max_messages(get_cfg),
    )
    if count:
        try:
            limit = min(limit, max(1, int(count)))
        except Exception:
            pass
    if limit <= 0:
        return []

    max_line_len = int(get_cfg("context_line_max_length", 120) or 120)
    selected = context_messages[-limit:]
    return [
        _render_context_line(item, detailed=False, limit=max_line_len)
        for item in selected
        if str(item.get("text") or "").strip()
    ]


def _extract_file_names_from_chain(message_chain: Any) -> list[str]:
    file_names: list[str] = []
    seen: set[str] = set()
    try:
        for seg in list(message_chain or []):
            name = ""
            if isinstance(seg, Comp.File):
                name = str(getattr(seg, "name", "") or "").strip()
            elif isinstance(seg, dict):
                if str(seg.get("type") or "").lower() != "file":
                    continue
                data = seg.get("data") or {}
                if isinstance(data, dict):
                    name = str(
                        data.get("name")
                        or data.get("file_name")
                        or data.get("file")
                        or "",
                    ).strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            file_names.append(name)
    except Exception:
        return file_names
    return file_names


def compute_active_wake_adjustment(
    *,
    group_state: GroupState,
    current_uid: str,
    bot_id: str,
    at_targets: list[dict[str, str]],
    at_all: bool,
    reply_to_id: str,
    include_state_bias: bool = True,
    get_cfg: Any = None,
) -> tuple[float, list[str]]:
    """
    计算主动唤醒降权系数（仅用于非 @bot 场景）。
    返回 (factor, reasons)。
    """
    factor = 1.0
    reasons: list[str] = []

    bid = str(bot_id or "").strip()
    current_uid_text = str(current_uid or "").strip()
    reply_uid = str(reply_to_id or "").strip()

    has_at_other = False
    for target in list(at_targets or []):
        tid = str((target or {}).get("id") or "").strip()
        if not tid:
            continue
        if bid and tid == bid:
            continue
        has_at_other = True
        break

    if has_at_other:
        factor *= 0.60
        reasons.append("at_other")

    # @全体属于显式艾特语义，不纳入主动唤醒降权逻辑。

    if reply_uid and ((not bid) or reply_uid != bid):
        factor *= 0.65
        reasons.append("reply_other")

    # 近 8 条非 bot 上下文中若存在多人发言，进一步轻降权。
    recent = list(group_state.context_messages or [])[-8:]
    non_bot_uids: set[str] = set()
    for item in recent:
        if bool(item.get("is_bot", False)):
            continue
        uid = str(item.get("uid") or "").strip()
        if uid:
            non_bot_uids.add(uid)
    if len(non_bot_uids) >= 2:
        # 当前用户之外还有其他用户发言，说明更可能是多人对话空气。
        if any(uid != current_uid_text for uid in non_bot_uids):
            factor *= 0.85
            reasons.append("multi_user_recent")

    if include_state_bias:
        # Bot 最近回复对象偏置：若当前发言人不是最近回复目标，则轻降权（仅用于相关性主动唤醒）。
        last_replied_uid = str(
            getattr(group_state, "context_bot_last_replied_to_uid", "") or ""
        ).strip()
        if (
            last_replied_uid
            and current_uid_text
            and current_uid_text != last_replied_uid
        ):
            factor *= 0.90
            reasons.append("not_last_replied_target")

        # 当前用户最近互动时间过旧时，进一步降权。
        last_interaction = 0.0
        try:
            last_interaction = float(
                (group_state.last_user_interaction or {}).get(current_uid_text, 0.0)
                or 0.0
            )
        except Exception:
            last_interaction = 0.0
        if last_interaction > 0:
            idle_sec = max(0.0, time.time() - last_interaction)
            if idle_sec > 180:
                factor *= 0.85
                reasons.append("user_interaction_stale")

    silence_bonus = _compute_active_wake_silence_bonus(
        group_state=group_state,
        include_state_bias=include_state_bias,
        get_cfg=get_cfg,
    )
    if silence_bonus > 0:
        factor *= 1.0 + silence_bonus
        reasons.append(f"silence_bonus+{silence_bonus:.2f}")

    factor = max(0.05, min(1.30, factor))
    return factor, reasons


def _compute_active_wake_silence_bonus(
    *,
    group_state: GroupState,
    include_state_bias: bool,
    get_cfg: Any = None,
) -> float:
    steps = [20, 80, 150]
    values = [0.05, 0.10, 0.20]
    cap = 0.20
    relevant_ratio = 0.7

    if get_cfg is None:
        return 0.0
    if not bool(get_cfg("active_wake_silence_bonus_enabled", True)):
        return 0.0
    if cap <= 0:
        return 0.0

    msg_count = max(0, int(getattr(group_state, "active_wake_new_msg_count", 0) or 0))
    if msg_count <= 0:
        return 0.0

    bonus = 0.0
    for idx, step in enumerate(steps):
        if msg_count < step:
            continue
        bonus = max(bonus, float(values[idx]))

    if bonus <= 0:
        return 0.0

    if include_state_bias:
        bonus *= relevant_ratio

    return max(0.0, min(cap, bonus))


def _extract_first_image_file_and_url(message_chain: Any) -> tuple[str, str, str]:
    """提取第一张图片的 path/file/url，兼容 Reply.chain 中的嵌套图片。"""
    try:
        for seg in message_chain or []:
            if isinstance(seg, Comp.Image):
                path_val = str(getattr(seg, "path", "") or "").strip()
                file_val = str(getattr(seg, "file", "") or "").strip()
                url_val = str(getattr(seg, "url", "") or "").strip()
                if path_val or file_val or url_val:
                    return path_val, file_val, url_val
                try:
                    seg_dict = seg.toDict()
                    if isinstance(seg_dict, dict):
                        data = seg_dict.get("data") or {}
                        if isinstance(data, dict):
                            path_val = str(data.get("path") or "").strip()
                            file_val = str(data.get("file") or "").strip()
                            url_val = str(data.get("url") or "").strip()
                            if path_val or file_val or url_val:
                                return path_val, file_val, url_val
                except Exception:
                    pass
            if isinstance(seg, dict) and seg.get("type") == "image":
                data = seg.get("data") or {}
                if isinstance(data, dict):
                    path_val = str(data.get("path") or "").strip()
                    file_val = str(data.get("file") or "").strip()
                    url_val = str(data.get("url") or "").strip()
                    if path_val or file_val or url_val:
                        return path_val, file_val, url_val
            if isinstance(seg, Comp.Reply) and getattr(seg, "chain", None):
                nested = _extract_first_image_file_and_url(seg.chain)
                if any(nested):
                    return nested
    except Exception:
        return "", "", ""
    return "", "", ""


def _build_image_caption_cache_key(image_file: str, image_url: str) -> str:
    file_key = str(image_file or "").strip()
    url_key = str(image_url or "").strip()
    if file_key:
        return f"file:{file_key}"
    if url_key:
        return f"url:{url_key}"
    return ""


def _get_cached_image_caption(cache_key: str) -> str:
    key = str(cache_key or "").strip()
    if not key:
        return ""
    item = _context_image_caption_cache.get(key) or {}
    try:
        expire_at = float(item.get("expire_at", 0.0) or 0.0)
    except Exception:
        expire_at = 0.0
    if expire_at <= time.time():
        _context_image_caption_cache.pop(key, None)
        return ""
    text = str(item.get("caption") or "").strip()
    if text:
        item["expire_at"] = time.time() + _CONTEXT_IMAGE_CAPTION_CACHE_TTL_SEC
        logger.debug(f"[图片转述] 图片转述缓存命中(已刷新TTL): {cache_key}")
    return text


def _set_cached_image_caption(cache_key: str, caption: str) -> None:
    key = str(cache_key or "").strip()
    val = str(caption or "").strip()
    if not key or not val:
        return
    _context_image_caption_cache[key] = {
        "caption": val,
        "expire_at": time.time() + _CONTEXT_IMAGE_CAPTION_CACHE_TTL_SEC,
    }
    logger.debug(f"[图片转述] 图片转述已缓存: {cache_key}")
    if len(_context_image_caption_cache) > _CONTEXT_IMAGE_CAPTION_CACHE_MAX_SIZE:
        oldest_key = next(iter(_context_image_caption_cache))
        del _context_image_caption_cache[oldest_key]


def _get_cached_emoji(cache_key: str) -> str:
    key = str(cache_key or "").strip()
    if not key:
        return ""
    item = _context_emoji_cache.get(key) or {}
    try:
        expire_at = float(item.get("expire_at", 0.0) or 0.0)
    except Exception:
        expire_at = 0.0
    if expire_at <= time.time():
        _context_emoji_cache.pop(key, None)
        return ""
    text = str(item.get("caption") or "").strip()
    if text:
        item["expire_at"] = time.time() + _CONTEXT_EMOJI_CAPTION_CACHE_TTL_SEC
        logger.debug(f"[表情包转述] 表情包转述缓存命中(已刷新TTL): {cache_key}")
    return text


def _set_cached_emoji(cache_key: str, caption: str) -> None:
    key = str(cache_key or "").strip()
    val = str(caption or "").strip()
    if not key or not val:
        return
    _context_emoji_cache[key] = {
        "caption": val,
        "expire_at": time.time() + _CONTEXT_EMOJI_CAPTION_CACHE_TTL_SEC,
    }
    logger.debug(f"[表情包转述] 表情包转述已缓存: {cache_key}")
    if len(_context_emoji_cache) > _CONTEXT_EMOJI_CAPTION_CACHE_MAX_SIZE:
        oldest_key = next(iter(_context_emoji_cache))
        del _context_emoji_cache[oldest_key]


def _normalize_local_image_path(path_or_uri: str) -> str:
    raw = str(path_or_uri or "").strip()
    if not raw:
        return ""
    if raw.startswith("file://"):
        candidate = unquote(raw[7:])
        if candidate.startswith("/") and len(candidate) > 3 and candidate[2] == ":":
            candidate = candidate[1:]
    else:
        candidate = unquote(raw)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    return ""


def _normalize_image_ref(image_ref: Any) -> str:
    """只接受有效的本地图片、HTTP 地址或可直接传给 Provider 的 base64。"""
    value = str(image_ref or "").strip()
    if not value:
        return ""
    local_path = _normalize_local_image_path(value)
    if local_path:
        return local_path
    lower_value = value.lower()
    if lower_value.startswith(("http://", "https://", "base64://")):
        return value
    if lower_value.startswith("data:image/"):
        comma_index = value.find(",")
        if comma_index > 0 and ";base64" in value[:comma_index].lower():
            payload = value[comma_index + 1 :].strip()
            if payload:
                return f"base64://{payload}"
    return ""


def _is_local_image_ref(value: Any) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    return (
        raw.lower().startswith("file://")
        or os.path.isabs(raw)
        or (len(raw) > 2 and raw[1] == ":" and raw[2] in {"\\", "/"})
    )


def _iter_image_segment_data(message_chain: Any):
    for segment in message_chain or []:
        if isinstance(segment, Comp.Image):
            yield {
                key: str(getattr(segment, key, "") or "").strip()
                for key in ("path", "file", "url")
            }
        elif isinstance(segment, dict) and str(segment.get("type") or "").lower() == "image":
            data = segment.get("data") or {}
            if isinstance(data, dict):
                yield {
                    key: str(data.get(key) or "").strip()
                    for key in ("path", "file", "url")
                }
        elif isinstance(segment, Comp.Reply) and getattr(segment, "chain", None):
            yield from _iter_image_segment_data(segment.chain)


def snapshot_image_files_for_context(
    message_chain: Any,
    raw_image_datas: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """在后台注入启动前复制仍存在的图片，避免核心事件清理导致竞态。"""
    snapshot_datas: list[dict[str, Any]] = []
    snapshot_paths: list[str] = []
    seen_paths: set[str] = set()
    sources: list[dict[str, Any]] = []
    for image_data in _iter_image_segment_data(message_chain):
        sources.append(image_data)
    for image_data in raw_image_datas or []:
        if isinstance(image_data, dict):
            sources.append(image_data)

    for image_data in sources:
        local_path = ""
        for key in ("path", "file", "url"):
            local_path = _normalize_local_image_path(image_data.get(key))
            if local_path:
                break
        if not local_path or local_path in seen_paths:
            continue
        seen_paths.add(local_path)
        suffix = os.path.splitext(local_path)[1] or ".img"
        fd, snapshot_path = tempfile.mkstemp(
            prefix="llme_context_image_",
            suffix=suffix,
        )
        os.close(fd)
        try:
            try:
                # 同一临时目录通常支持硬链接，能在不复制大文件的情况下跨过核心清理。
                os.remove(snapshot_path)
                os.link(local_path, snapshot_path)
            except OSError:
                shutil.copyfile(local_path, snapshot_path)
        except Exception:
            try:
                os.remove(snapshot_path)
            except OSError:
                pass
            continue
        snapshot_paths.append(snapshot_path)
        snapshot_datas.append(
            {"path": snapshot_path, "file": snapshot_path, "url": snapshot_path}
        )

    merged_datas = [
        data for data in (raw_image_datas or []) if isinstance(data, dict)
    ]
    merged_datas.extend(snapshot_datas)
    return merged_datas, snapshot_paths


def _is_emoji_image_data(seg_data: dict[str, Any]) -> bool:
    sub_type = seg_data.get("sub_type", seg_data.get("subType"))
    try:
        if sub_type is not None and int(sub_type) == 1:
            return True
    except (TypeError, ValueError):
        pass

    for key in ("emoji_id", "emojiId", "emojiID"):
        value = seg_data.get(key)
        if value not in (None, ""):
            return True
    for key in ("emoji_package_id", "emojiPackageId", "emojiPackageID"):
        value = seg_data.get(key)
        if value not in (None, ""):
            return True
    package = seg_data.get("emoji_package") or seg_data.get("emojiPackage")
    if isinstance(package, dict):
        if package.get("id") or package.get("package_id") or package.get("packageId"):
            return True

    for key in ("url", "file"):
        value = seg_data.get(key)
        if isinstance(value, str) and value:
            lowered = value.lower()
            if "gxh.vip.qq.com/club/item/parcel/item/" in lowered:
                return True
            if ("emoji_id=" in lowered) or ("emojiid=" in lowered):
                return True
            if "/emoji" in lowered or "/sticker" in lowered or "/face" in lowered:
                return True
            if key == "file" and lowered.startswith("fb-") and lowered.endswith(".gif"):
                return True

    for key in ("type", "image_type", "imageType"):
        value = seg_data.get(key)
        if isinstance(value, str) and value.strip().lower() in {
            "emoji",
            "sticker",
            "face",
            "meme",
        }:
            return True

    summary = seg_data.get("summary")
    if isinstance(summary, str):
        lowered = summary.lower()
        if ("表情" in summary) or ("emoji" in lowered) or ("sticker" in lowered):
            return True
        text = summary.strip()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if inner:
                lowered_inner = inner.lower()
                if (
                    ("图片" not in inner)
                    and ("视频" not in inner)
                    and ("image" not in lowered_inner)
                    and ("video" not in lowered_inner)
                ):
                    return True
    return False


def _extract_emoji_summary_from_data(seg_data: dict[str, Any]) -> str:
    summary = seg_data.get("summary")
    if isinstance(summary, str):
        text = _normalize_emoji_summary(summary)
        if text:
            return text
    return ""


def get_emoji_summary_from_sources(
    message_chain: Any,
    raw_image_datas: Optional[list[dict[str, Any]]] = None,
) -> str:
    """从消息链或原始图片数据中提取表情包/贴图的文字描述。"""
    for data in list(raw_image_datas or []):
        if _is_emoji_image_data(data):
            text = _extract_emoji_summary_from_data(data)
            if text:
                return text
    try:
        for seg in message_chain or []:
            if isinstance(seg, dict) and str(seg.get("type") or "").lower() == "image":
                data = seg.get("data") or {}
                if isinstance(data, dict) and _is_emoji_image_data(data):
                    text = _extract_emoji_summary_from_data(data)
                    if text:
                        return text
            elif isinstance(seg, dict) and str(seg.get("type") or "").lower() in (
                "mface",
                "marketface",
            ):
                data = seg.get("data") or {}
                name = None
                if isinstance(data, dict):
                    for k in ("name", "summary", "emoji_id", "id"):
                        v = data.get(k)
                        if isinstance(v, str) and v.strip():
                            name = v.strip()
                            break
                if name:
                    return _normalize_emoji_summary(name)
            elif isinstance(seg, Comp.Image):
                seg_data = _extract_image_data_from_component(seg)
                if _is_emoji_image_data(seg_data):
                    text = _extract_emoji_summary_from_data(seg_data)
                    if text:
                        return text
    except Exception:
        pass
    return ""


def _merge_image_caption_and_emoji_summary(caption: str, emoji_summary: str) -> str:
    cap = str(caption or "").strip()
    summary = str(emoji_summary or "").strip()
    if not summary:
        return cap
    if not cap:
        return summary
    if summary in cap:
        return cap
    return f"{cap}；表情：{summary}"


def append_emoji_summary_suffix(base_text: str, emoji_summary: str) -> str:
    """在文本末尾追加表情包描述的附注信息。"""
    base = str(base_text or "").strip()
    summary = str(emoji_summary or "").strip()
    if not base or not summary:
        return base
    if summary in base:
        return base
    return f"{base}（表情：{summary}）"


def _extract_image_data_from_component(comp: Any) -> dict[str, Any]:
    seg_data: dict[str, Any] = {}
    for attr in (
        "subType",
        "sub_type",
        "summary",
        "file",
        "url",
        "type",
        "imageType",
        "image_type",
    ):
        value = getattr(comp, attr, None)
        if value not in (None, ""):
            seg_data[attr] = value
    try:
        comp_dict = comp.toDict()
        if isinstance(comp_dict, dict):
            data = comp_dict.get("data")
            if isinstance(data, dict):
                for key, value in data.items():
                    if value not in (None, "") and key not in seg_data:
                        seg_data[key] = value
    except Exception:
        pass
    return seg_data


def extract_raw_image_datas_from_event(event: Any) -> list[dict[str, Any]]:
    """从事件的原始消息中提取所有图片组件的 data 字典列表。"""
    raw_message = getattr(getattr(event, "message_obj", None), "raw_message", None)
    if raw_message is None:
        return []
    try:
        segments = raw_message.get("message")
    except Exception:
        segments = getattr(raw_message, "message", None)
    if not isinstance(segments, list):
        return []
    image_datas: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("type") or "").lower() != "image":
            continue
        data = seg.get("data") or {}
        if isinstance(data, dict) and data:
            image_datas.append(data)
    return image_datas


def get_image_component_label(
    message_chain: Any, raw_image_datas: Optional[list[dict[str, Any]]] = None
) -> str:
    """判断图片组件的标签：是表情包还是普通图片。"""
    fallback_image_data: dict[str, Any] = {}
    for data in list(raw_image_datas or []):
        if _is_emoji_image_data(data):
            return "表情包"
        if (not fallback_image_data) and isinstance(data, dict):
            fallback_image_data = dict(data)
    try:
        for seg in message_chain or []:
            if isinstance(seg, dict) and str(seg.get("type") or "").lower() == "image":
                data = seg.get("data") or {}
                if _is_emoji_image_data(data):
                    return "表情包"
                if not fallback_image_data:
                    fallback_image_data = dict(data)
            elif isinstance(seg, Comp.Image):
                seg_data = _extract_image_data_from_component(seg)
                if _is_emoji_image_data(seg_data):
                    return "表情包"
                if (not fallback_image_data) and seg_data:
                    fallback_image_data = dict(seg_data)
    except Exception:
        pass
    return "图片"


def extract_addressing_signals(
    message_chain: Any,
    bot_id: str = "",
) -> tuple[list[dict[str, str]], bool, bool, str, str]:
    """提取消息链中的寻址信号：@列表、是否@机器人、是否@全体、回复目标。"""
    at_targets: list[dict[str, str]] = []
    at_bot = False
    at_all = False
    reply_to_id = ""
    reply_msg_id = ""
    bid = str(bot_id or "").strip()
    try:
        for seg in message_chain or []:
            if isinstance(seg, Comp.At):
                target_id = str(getattr(seg, "qq", "") or "").strip()
                target_name = str(getattr(seg, "name", "") or "").strip()
                if target_id:
                    at_targets.append({"id": target_id, "name": target_name})
                if bid and target_id == bid:
                    at_bot = True
            elif isinstance(seg, Comp.AtAll):
                at_all = True
            elif isinstance(seg, Comp.Reply):
                sender_id = str(getattr(seg, "sender_id", "") or "").strip()
                if sender_id:
                    reply_to_id = sender_id
                rid = str(
                    getattr(seg, "id", "") or getattr(seg, "message_id", "") or ""
                ).strip()
                if rid:
                    reply_msg_id = rid
            elif isinstance(seg, dict):
                seg_type = seg.get("type")
                data = seg.get("data") or {}
                if seg_type == "at":
                    target_id = str(data.get("qq") or data.get("id") or "").strip()
                    target_name = str(data.get("name") or "").strip()
                    if target_id:
                        at_targets.append({"id": target_id, "name": target_name})
                    if bid and target_id == bid:
                        at_bot = True
                elif seg_type == "atall":
                    at_all = True
                elif seg_type == "reply":
                    sender_id = str(
                        data.get("sender_id") or data.get("user_id") or ""
                    ).strip()
                    if sender_id:
                        reply_to_id = sender_id
                    rid = str(data.get("id") or data.get("message_id") or "").strip()
                    if rid:
                        reply_msg_id = rid
    except Exception:
        return [], False, False, "", ""
    return at_targets, at_bot, at_all, reply_to_id, reply_msg_id


def _summary_from_raw_message_chain(chain: Any) -> str:
    texts: list[str] = []
    has_image = False
    has_video = False
    has_forward = False
    has_json = False
    has_record = False
    file_name = ""
    emoji_summary = get_emoji_summary_from_sources(chain)
    try:
        for seg in chain or []:
            if isinstance(seg, dict):
                seg_type = str(seg.get("type") or "").lower()
                data = seg.get("data") or {}
                if seg_type == "text":
                    t = str(data.get("text") or "").strip()
                    if t:
                        texts.append(t)
                elif seg_type == "image":
                    has_image = True
                elif seg_type == "video":
                    has_video = True
                elif seg_type == "record":
                    has_record = True
                elif seg_type == "forward":
                    has_forward = True
                elif seg_type == "json":
                    has_json = True
                elif seg_type == "file":
                    file_name = str(data.get("name") or data.get("file") or "").strip()
            else:
                if isinstance(seg, Comp.Plain):
                    t = str(getattr(seg, "text", "") or "").strip()
                    if t:
                        texts.append(t)
                elif isinstance(seg, Comp.Image):
                    has_image = True
                elif isinstance(seg, Comp.Video):
                    has_video = True
                elif isinstance(seg, Comp.Record):
                    has_record = True
                elif isinstance(seg, Comp.Forward):
                    has_forward = True
                elif isinstance(seg, Comp.Json):
                    has_json = True
                elif isinstance(seg, Comp.File):
                    file_name = str(
                        getattr(seg, "name", "") or getattr(seg, "file", "") or ""
                    ).strip()
    except Exception:
        pass

    text = " ".join([t for t in texts if t]).strip()
    if text:
        return append_emoji_summary_suffix(_clip(text, 120), emoji_summary)
    image_label = get_image_component_label(chain)
    placeholder, _ = build_media_trigger_message(
        sender_name="对方",
        has_image_component=has_image,
        has_video_component=has_video,
        has_file_component=bool(file_name),
        has_forward_component=has_forward,
        has_json_component=has_json,
        has_record_component=has_record,
        file_name=file_name,
        image_label=image_label,
    )
    if placeholder:
        return append_emoji_summary_suffix(placeholder, emoji_summary)
    return "[引用消息]"


async def try_build_reply_preview(event: Any, reply_msg_id: str) -> str:
    """尝试根据回复消息 ID 构建引用消息的预览文本。"""
    rid = str(reply_msg_id or "").strip()
    if not rid:
        return ""
    try:
        client = getattr(event, "bot", None)
        api = getattr(client, "api", None)
        if api is None:
            return ""
        original_msg = await api.call_action("get_msg", message_id=rid)
        if _is_unavailable_get_msg_payload(original_msg):
            return ""
        if not isinstance(original_msg, dict):
            return ""
        sender = original_msg.get("sender", {}) or {}
        sender_name = str(
            sender.get("nickname")
            or sender.get("card")
            or sender.get("user_id")
            or "对方"
        ).strip()
        chain = original_msg.get("message") or []
        summary = _summary_from_raw_message_chain(chain)
        return f"{sender_name}: {summary}"
    except Exception as e:
        logger.debug(f"[上下文注入] 获取引用消息摘要失败：{e}")
        return ""


def _provider_id(provider: Any) -> str:
    provider_cfg = getattr(provider, "provider_config", None)
    if isinstance(provider_cfg, dict):
        return str(
            provider_cfg.get("id")
            or provider_cfg.get("provider_id")
            or provider_cfg.get("name")
            or ""
        ).strip()
    return str(
        getattr(provider, "id", None)
        or getattr(provider, "provider_id", None)
        or getattr(provider, "name", None)
        or ""
    ).strip()


def _provider_supports_image_input(provider: Any) -> bool:
    provider_cfg = getattr(provider, "provider_config", None)
    if not isinstance(provider_cfg, dict):
        return False
    modalities = provider_cfg.get("modalities", None)
    # 与 AstrBot 核心保持一致：空列表表示旧配置/未配置，按支持多模态兼容。
    if modalities == []:
        return True
    return isinstance(modalities, list) and "image" in modalities


def _append_provider_candidate(candidates: list[Any], provider: Any) -> None:
    if provider is None or not hasattr(provider, "text_chat"):
        return
    provider_key = _provider_id(provider) or str(id(provider))
    if any(((_provider_id(item) or str(id(item))) == provider_key) for item in candidates):
        return
    candidates.append(provider)


def select_image_caption_provider(
    *,
    get_cfg: Any,
    provider_by_id_resolver: Any,
    default_provider_resolver: Any,
    preferred_provider_id: str = "",
    framework_provider_settings: Optional[dict[str, Any]] = None,
    provider_candidates_resolver: Any = None,
) -> Any:
    """按显式配置、框架回退列表和可用 Provider 自动选择视觉 Provider。"""
    framework_settings = (
        framework_provider_settings
        if isinstance(framework_provider_settings, dict)
        else {}
    )
    candidates: list[Any] = []
    provider_ids: list[str] = []

    def add_provider_id(raw_id: Any) -> None:
        provider_id = str(raw_id or "").strip()
        if provider_id and provider_id not in provider_ids:
            provider_ids.append(provider_id)

    add_provider_id(preferred_provider_id)
    add_provider_id(
        get_cfg("context_injection_image_caption_provider_id", "")
        if callable(get_cfg)
        else ""
    )
    add_provider_id(framework_settings.get("default_image_caption_provider_id", ""))
    add_provider_id(get_cfg("video_image_provider_id") if callable(get_cfg) else "")

    fallback_ids = framework_settings.get("fallback_chat_models", [])
    if isinstance(fallback_ids, list):
        for fallback_id in fallback_ids:
            add_provider_id(fallback_id)

    for provider_id in provider_ids:
        try:
            _append_provider_candidate(
                candidates,
                provider_by_id_resolver(provider_id),
            )
        except Exception:
            continue

    if callable(default_provider_resolver):
        try:
            _append_provider_candidate(candidates, default_provider_resolver())
        except Exception:
            pass

    if callable(provider_candidates_resolver):
        try:
            all_providers = provider_candidates_resolver() or []
            if isinstance(all_providers, (list, tuple)):
                for provider in all_providers:
                    _append_provider_candidate(candidates, provider)
        except Exception:
            pass

    for provider in candidates:
        if _provider_supports_image_input(provider):
            return provider
    return None


async def try_get_image_caption(
    *,
    event: Any = None,
    message_chain: Any,
    raw_image_datas: Optional[list[dict[str, Any]]] = None,
    get_cfg: Any,
    provider_by_id_resolver: Any,
    default_provider_resolver: Any,
    timeout_sec: float = 60.0,
    preferred_provider_id: str = "",
    allow_default_provider: bool = True,
    emoji_mode: bool = False,
    framework_provider_settings: Optional[dict[str, Any]] = None,
    provider_candidates_resolver: Any = None,
) -> str:
    """尝试调用视觉模型为图片生成文字描述，优先使用表情包摘要缓存。"""
    emoji_summary = get_emoji_summary_from_sources(
        message_chain, raw_image_datas=raw_image_datas
    )
    image_path, image_file, image_url = _extract_first_image_file_and_url(message_chain)
    _raw_cache_datas = raw_image_datas or extract_raw_image_datas_from_event(event)
    _cache_file, _cache_url, _cache_path = "", "", ""
    for _d in _raw_cache_datas or []:
        if isinstance(_d, dict):
            _cache_path = str(_d.get("path") or "").strip()
            _cache_file = str(_d.get("file") or "").strip()
            _cache_url = str(_d.get("url") or "").strip()
            if _cache_file or _cache_url or _cache_path:
                break
    if not image_path and not image_file and not image_url:
        image_path, image_file, image_url = _cache_path, _cache_file, _cache_url
    if not image_path and not image_file and not image_url:
        return emoji_summary or ""
    cache_key = _build_image_caption_cache_key(
        # 当合并消息逐张调用时，message_chain 中的标识属于当前图片。
        # 只有当前组件没有标识时，才使用事件级原始数据作为兜底；不能
        # 始终优先使用事件中第一张图片，否则后续图片会复用第一张的缓存。
        image_file or image_path,
        image_url,
    )
    _is_emoji = False
    if emoji_mode:
        _is_emoji = bool(emoji_summary)
        if not _is_emoji:
            for _d in _raw_cache_datas or []:
                if isinstance(_d, dict) and _is_emoji_image_data(_d):
                    _is_emoji = True
                    break
    if _is_emoji:
        cached_caption = _get_cached_emoji(cache_key)
        if cached_caption:
            return _merge_image_caption_and_emoji_summary(cached_caption, emoji_summary)
    else:
        cached_caption = _get_cached_image_caption(cache_key)
        if cached_caption:
            return _merge_image_caption_and_emoji_summary(cached_caption, emoji_summary)

    logger.debug(
        f"{'[表情包转述]' if _is_emoji else '[图片转述]'} 转述生成: cache_key={cache_key}"
    )

    provider = select_image_caption_provider(
        get_cfg=get_cfg,
        provider_by_id_resolver=provider_by_id_resolver,
        default_provider_resolver=(
            default_provider_resolver if allow_default_provider else None
        ),
        preferred_provider_id=preferred_provider_id,
        framework_provider_settings=framework_provider_settings,
        provider_candidates_resolver=provider_candidates_resolver,
    )
    if not provider or (not hasattr(provider, "text_chat")):
        return emoji_summary or ""

    prompt = (
        _CONTEXT_EMOJI_CAPTION_PROMPT if _is_emoji else _CONTEXT_IMAGE_CAPTION_PROMPT
    )
    if not prompt:
        return emoji_summary or ""

    image_input = await _resolve_image_input_from_segment_data(
        event,
        {"path": image_path, "file": image_file, "url": image_url},
    )
    if not image_input:
        for raw_data in _raw_cache_datas or []:
            if not isinstance(raw_data, dict):
                continue
            image_input = await _resolve_image_input_from_segment_data(
                event,
                raw_data,
            )
            if image_input:
                break
    temp_image_path = ""
    if image_input.lower().startswith(("http://", "https://")):
        # 兜底：下载外链到本地临时文件再传给模型。
        try:
            from .media_parser import download_media_to_temp

            source = image_input
            if not source:
                return emoji_summary or ""
            temp_image_path = str(
                await download_media_to_temp(
                    source,
                    size_mb_limit=10,
                )
                or "",
            ).strip()
            image_input = _normalize_local_image_path(temp_image_path)
        except Exception as e:
            logger.debug(
                f"[上下文注入] 图像转述失败(download): {type(e).__name__}: {e}"
            )
            image_input = ""
    if not image_input:
        return (emoji_summary + " [处理失败]") if emoji_summary else "[处理失败]"

    resized_tmp_path = ""
    if image_input and os.path.exists(image_input):
        try:
            from PIL import Image

            img = Image.open(image_input)
            max_dim = 2048
            if img.width > max_dim or img.height > max_dim:
                ratio = max_dim / max(img.width, img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                fd, resized_tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                img.save(resized_tmp_path, "JPEG", quality=95)
                image_input = resized_tmp_path
        except Exception as e:
            logger.debug(f"[上下文注入] 图片压缩失败: {e}")

    try:
        async with _context_image_caption_semaphore:
            if timeout_sec > 0:
                resp = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=prompt,
                        session_id=uuid.uuid4().hex,
                        image_urls=[image_input],
                        persist=False,
                    ),
                    timeout=max(5.0, float(timeout_sec)),
                )
            else:
                resp = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=prompt,
                        session_id=uuid.uuid4().hex,
                        image_urls=[image_input],
                        persist=False,
                    ),
                    timeout=60,
                )
            text = str(getattr(resp, "completion_text", "") or "").strip()
            if "处理失败" in text:
                caption = "[处理失败]"
            else:
                caption = text[:200]
            if caption:
                if _is_emoji:
                    _set_cached_emoji(
                        cache_key,
                        _merge_image_caption_and_emoji_summary(caption, emoji_summary),
                    )
                else:
                    _set_cached_image_caption(cache_key, caption)
            if _is_emoji:
                return _merge_image_caption_and_emoji_summary(caption, emoji_summary)
            return _merge_image_caption_and_emoji_summary(caption, emoji_summary)
    except Exception as e:
        logger.debug(f"[上下文注入] 图像转述失败(model): {type(e).__name__}: {e}")
        return (emoji_summary + " [处理失败]") if emoji_summary else "[处理失败]"
    finally:
        if temp_image_path:
            try:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
            except Exception:
                pass
        if resized_tmp_path:
            try:
                if os.path.exists(resized_tmp_path):
                    os.remove(resized_tmp_path)
            except Exception:
                pass


async def _resolve_image_input_from_segment_data(
    event: Any,
    seg_data: dict[str, Any],
) -> str:
    if not isinstance(seg_data, dict):
        return ""
    path_val = str(
        seg_data.get("path") or seg_data.get("file_path") or ""
    ).strip()
    file_val = str(seg_data.get("file") or "").strip()
    url_val = str(seg_data.get("url") or "").strip()

    for candidate in (path_val, file_val, url_val):
        normalized = _normalize_image_ref(candidate)
        if normalized:
            return normalized

    # 失效的绝对路径不能作为 OneBot file 参数，也不能交给 aiohttp 当 URL。
    api_candidates = [
        candidate
        for candidate in (file_val, url_val)
        if candidate and not _is_local_image_ref(candidate)
    ]
    if api_candidates and isinstance(event, AiocqhttpMessageEvent):
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None) if bot else None
        if api is not None and hasattr(api, "call_action"):
            actions: list[tuple[str, dict[str, Any]]] = []
            group_id = ""
            try:
                group_id = str(event.get_group_id() or "").strip()
            except Exception:
                pass
            for image_ref in api_candidates:
                actions.extend(
                    [
                        ("get_image", {"file": image_ref}),
                        ("get_image", {"file_id": image_ref}),
                        ("get_image", {"id": image_ref}),
                        ("get_image", {"image": image_ref}),
                        ("get_file", {"file_id": image_ref}),
                        ("get_file", {"file": image_ref}),
                    ]
                )
                if group_id:
                    group_value: Any = (
                        int(group_id) if group_id.isdigit() else group_id
                    )
                    actions.append(
                        (
                            "get_group_file_url",
                            {"group_id": group_value, "file_id": image_ref},
                        )
                    )
                actions.append(("get_private_file_url", {"file_id": image_ref}))

            for action, params in actions:
                try:
                    image_resp = await api.call_action(action, **params)
                except Exception:
                    continue
                response_dicts = [image_resp]
                if isinstance(image_resp, dict) and isinstance(
                    image_resp.get("data"), dict
                ):
                    response_dicts.append(image_resp["data"])
                for response in response_dicts:
                    if not isinstance(response, dict):
                        continue
                    for key in ("path", "file", "url"):
                        normalized = _normalize_image_ref(response.get(key))
                        if normalized:
                            return normalized
    return ""

async def inject_merged_images_by_provider(
    *,
    event: Any,
    req: ProviderRequest,
    merged_msg_ids: list[str],
    current_provider: Any,
    get_cfg: Any,
    framework_provider_settings: Optional[dict[str, Any]] = None,
    provider_by_id_resolver: Any,
    default_provider_resolver: Any,
    provider_candidates_resolver: Any = None,
) -> dict[str, Any]:
    """将合并消息中的图片补入 ProviderRequest 的 image_urls，或降级为文字描述。返回注入结果统计。"""
    batch_msg_ids = [
        str(mid).strip() for mid in (merged_msg_ids or []) if str(mid or "").strip()
    ]
    if not batch_msg_ids:
        return {"mode": "none", "image_count": 0, "caption_count": 0}

    original_msgs = await _fetch_messages_by_ids(event, batch_msg_ids)
    if not original_msgs:
        return {"mode": "none", "image_count": 0, "caption_count": 0}

    image_segments: list[dict[str, Any]] = []
    for original_msg in original_msgs:
        for segment in original_msg.get("message") or []:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("type") or "").strip().lower() != "image":
                continue
            seg_data = segment.get("data") or {}
            if not isinstance(seg_data, dict):
                continue
            image_segments.append(segment)

    if not image_segments:
        return {"mode": "none", "image_count": 0, "caption_count": 0}

    if _provider_supports_image_input(current_provider):
        if not hasattr(req, "image_urls") or req.image_urls is None:
            req.image_urls = []
        existing_urls = {
            str(item).strip()
            for item in list(req.image_urls or [])
            if str(item).strip()
        }
        injected_count = 0
        for segment in image_segments:
            seg_data = segment.get("data") or {}
            image_input = await _resolve_image_input_from_segment_data(event, seg_data)
            if not image_input or image_input in existing_urls:
                continue
            req.image_urls.append(image_input)
            existing_urls.add(image_input)
            injected_count += 1
        if injected_count > 0:
            logger.debug(
                "[LLMEnhancement] 已将合并消息中的图片补入 req.image_urls："
                f"group={getattr(event, 'get_group_id', lambda: None)() or 'private'}, "
                f"uid={getattr(event, 'get_sender_id', lambda: None)()}, count={injected_count}",
            )
        return {"mode": "image", "image_count": injected_count, "caption_count": 0}

    provider_settings = (
        framework_provider_settings
        if isinstance(framework_provider_settings, dict)
        else {}
    )
    caption_provider = select_image_caption_provider(
        get_cfg=get_cfg,
        provider_by_id_resolver=provider_by_id_resolver,
        default_provider_resolver=default_provider_resolver,
        framework_provider_settings=provider_settings,
        provider_candidates_resolver=provider_candidates_resolver,
    )
    caption_provider_id = _provider_id(caption_provider)
    if caption_provider is None:
        logger.debug(
            "[LLMEnhancement] 当前 Provider 不支持视觉，且没有可用的图像转述 Provider，跳过合并图片注入："
            f"group={getattr(event, 'get_group_id', lambda: None)() or 'private'}, "
            f"uid={getattr(event, 'get_sender_id', lambda: None)()}, image_count={len(image_segments)}",
        )
        return {"mode": "skip", "image_count": len(image_segments), "caption_count": 0}

    caption_count = 0
    for segment in image_segments:
        caption = await try_get_image_caption(
            event=event,
            message_chain=[segment],
            raw_image_datas=None,
            get_cfg=get_cfg,
            provider_by_id_resolver=provider_by_id_resolver,
            default_provider_resolver=default_provider_resolver,
            preferred_provider_id=caption_provider_id,
            allow_default_provider=False,
            timeout_sec=0,
            framework_provider_settings=provider_settings,
            provider_candidates_resolver=provider_candidates_resolver,
        )
        if not caption:
            continue
        caption_text = f"<image_caption>{caption}</image_caption>"
        if not append_text_part_to_request(req, caption_text, mark_temp=False):
            req.prompt = f"{(req.prompt or '').strip()}\n\n{caption_text}".strip()
        caption_count += 1

    if caption_count > 0:
        logger.debug(
            "[LLMEnhancement] 当前 Provider 不支持视觉，已改为注入合并图片转述："
            f"group={getattr(event, 'get_group_id', lambda: None)() or 'private'}, "
            f"uid={getattr(event, 'get_sender_id', lambda: None)()}, "
            f"image_count={len(image_segments)}, caption_count={caption_count}, "
            f"caption_provider={caption_provider_id}",
        )
        return {
            "mode": "caption",
            "image_count": len(image_segments),
            "caption_count": caption_count,
        }

    logger.debug(
        "[LLMEnhancement] 当前 Provider 不支持视觉，但合并图片转述未产出内容："
        f"group={getattr(event, 'get_group_id', lambda: None)() or 'private'}, "
        f"uid={getattr(event, 'get_sender_id', lambda: None)()}, "
        f"image_count={len(image_segments)}, caption_provider={caption_provider_id}",
    )
    return {"mode": "caption", "image_count": len(image_segments), "caption_count": 0}


def _extract_forward_id_from_chain(message_chain: Any) -> str:
    try:
        for seg in message_chain or []:
            if isinstance(seg, Comp.Forward):
                fid = str(getattr(seg, "id", "") or "").strip()
                if fid:
                    return fid
            elif isinstance(seg, dict):
                if str(seg.get("type") or "").lower() != "forward":
                    continue
                data = seg.get("data") or {}
                fid = str(data.get("id") or "").strip()
                if fid:
                    return fid
    except Exception:
        return ""
    return ""


async def build_text_context_enrichment(
    *,
    event: Any,
    get_cfg: Any,
    message_chain: Any,
    parse_options: set[str],
) -> str:
    """构建文本上下文的富化内容：解析 URL、JSON 卡片、文件信息等。"""
    parts: list[str] = []

    # URL 解析（文本链路天然可命中，非文本链路也会复用该函数）。
    if "url" in parse_options and bool(get_cfg("url_parse_enable", True)):
        try:
            from .url_parser import extract_url_infos_from_chain

            url_result = await extract_url_infos_from_chain(
                event=event,
                chain=list(message_chain or []),
                timeout_sec=max(2, int(get_cfg("inject_url_timeout_sec", 8) or 8)),
                max_download_kb=max(
                    32, int(get_cfg("inject_url_max_download_kb", 512) or 512)
                ),
                block_private_network=bool(
                    get_cfg("inject_url_block_private_network", True)
                ),
                blocked_domains=get_cfg("inject_url_blocked_domains", []) or [],
                url_parse_provider=get_cfg("url_parse_provider", "exa") or "exa",
                url_parse_api_keys=get_cfg("url_parse_api_keys", []) or [],
            )
            if getattr(url_result, "details", None):
                parts.append(
                    f"URL解析: {_clip('；'.join(list(url_result.details)[:2]), 220)}"
                )
        except Exception:
            pass

    # JSON 卡片解析（复用引用解析中的同源能力）。
    if "json" in parse_options and bool(get_cfg("json_parse_enable", True)):
        try:
            from .json_parser import extract_json_infos_from_chain

            json_news_texts, json_infos = extract_json_infos_from_chain(
                list(message_chain or [])
            )
            json_news_texts = list(dict.fromkeys(json_news_texts))
            json_infos = list(dict.fromkeys(json_infos))
            if json_infos or json_news_texts:
                json_parts: list[str] = []
                if json_infos:
                    json_parts.append(f"卡片信息: {_clip('；'.join(json_infos), 220)}")
                if json_news_texts:
                    json_parts.append(
                        f"正文摘录: {_clip('；'.join(json_news_texts[:5]), 220)}"
                    )
                parts.append("JSON解析: " + " | ".join(json_parts))
        except Exception:
            pass

    # 文件解析（仅在主配置开启文件文本注入且长度>0时生效）。
    if "file" in parse_options and bool(get_cfg("file_parse_enable", True)):
        try:
            file_names = _extract_file_names_from_chain(message_chain)
            if file_names:
                parts.append("文件信息: " + _clip("；".join(file_names[:3]), 220))
        except Exception:
            pass

    return " | ".join(parts)


async def build_non_text_context_text(
    *,
    event: Any,
    context: Any,
    get_cfg: Any,
    message_chain: Any,
    sender_name: str,
    msg_id: str,
    parse_options: set[str],
    raw_image_datas: Optional[list[dict[str, Any]]] = None,
    provider_by_id_resolver: Any,
    default_provider_resolver: Any,
    emoji_mode: bool = False,
    framework_provider_settings: Optional[dict[str, Any]] = None,
    provider_candidates_resolver: Any = None,
) -> str:
    """为非文本消息（图片、语音等）构建用于上下文注入的文字描述。"""
    parts: list[str] = []
    has_record_component = False
    try:
        for seg in message_chain or []:
            if isinstance(seg, Comp.Record):
                has_record_component = True
                break
            if isinstance(seg, dict) and str(seg.get("type") or "").lower() == "record":
                has_record_component = True
                break
    except Exception:
        has_record_component = False

    # image: 复用图片转述逻辑
    if "image" in parse_options:
        image_caption = await try_get_image_caption(
            event=event,
            message_chain=message_chain,
            raw_image_datas=raw_image_datas,
            get_cfg=get_cfg,
            provider_by_id_resolver=provider_by_id_resolver,
            default_provider_resolver=default_provider_resolver,
            emoji_mode=emoji_mode,
            framework_provider_settings=framework_provider_settings,
            provider_candidates_resolver=provider_candidates_resolver,
        )
        if image_caption:
            image_label = get_image_component_label(
                message_chain, raw_image_datas=raw_image_datas
            )
            parts.append(f"{image_label}转述: {image_caption}")

    # forward: 仅抽取聊天记录文本，不做媒体解析
    if "forward" in parse_options and bool(get_cfg("forward_parse_enable", True)):
        forward_id = _extract_forward_id_from_chain(message_chain)
        if forward_id:
            try:
                from .forward_parser import extract_forward_content

                max_forward_messages = int(
                    get_cfg("forward_message_max_count", 50) or 50
                )
                nested_parse_depth = int(get_cfg("nested_parse_depth", 5) or 5)
                enable_json_parse = bool(get_cfg("json_parse_enable", True))
                (
                    extracted_texts,
                    _image_urls,
                    _video_sources,
                ) = await extract_forward_content(
                    event.bot,
                    forward_id=forward_id,
                    max_message_count=max_forward_messages,
                    nested_parse_depth=nested_parse_depth,
                    enable_json_parse=enable_json_parse,
                )
                if extracted_texts:
                    preview_count = max(1, int(max_forward_messages))
                    preview_texts = extracted_texts[:preview_count]
                    # 按聊天记录解析配置动态放宽摘要长度，避免较长转发被额外截断过多。
                    clip_limit = max(220, min(4000, preview_count * 120))
                    parts.append(
                        f"聊天记录抽取: {_clip('；'.join(preview_texts), clip_limit)}"
                    )
            except Exception:
                pass

    if "record" in parse_options and has_record_component:
        record_text = ""
        if bool(get_cfg("record_parse_enable", True)):
            record_text, cleanup_paths = await transcribe_record_from_chain(
                context=context,
                get_cfg=get_cfg,
                event=event,
                chain=message_chain,
            )
            if cleanup_paths:
                await cleanup_paths_later(cleanup_paths)
        if record_text:
            parts.append(f"语音转写: {_clip(record_text, 220)}")
        else:
            parts.append("语音消息")

    # url/file/json: 复用现有解析逻辑
    structured_enrichment = await build_text_context_enrichment(
        event=event,
        get_cfg=get_cfg,
        message_chain=message_chain,
        parse_options=parse_options,
    )
    if structured_enrichment:
        parts.append(structured_enrichment)

    if parts:
        if len(parts) == 1:
            only = parts[0]
            if only == "语音消息":
                return "发送了1条语音消息"
            if only.startswith("语音转写:"):
                return f"发送了1条语音消息，{only}"
            if only.startswith("表情包转述:"):
                return f"发送了1个表情包，{only}"
            if only.startswith("图片转述:"):
                return f"发送了1张图片，{only}"
            if only.startswith("聊天记录抽取:"):
                return f"发送了聊天记录，{only}"
            if only.startswith("URL解析:"):
                return f"发送了1个链接，{only}"
            if only.startswith("JSON解析:"):
                return f"发送了分享卡片，{only}"
            if only.startswith("文件信息:"):
                return f"发送了1个文件，{only}"
        return f"发送了复合媒体消息，解析结果：{' | '.join(parts)}"
    return ""


def build_context_text(
    message_text: str,
    sender_name: str,
    *,
    image_caption: str,
    has_image_component: bool,
    has_video_component: bool,
    has_file_component: bool,
    has_forward_component: bool,
    has_json_component: bool,
    has_record_component: bool,
    file_name: str,
    image_label: str = "图片",
    emoji_summary: str = "",
) -> str:
    """构建单条上下文消息的最终文本，融合图片描述、表情摘要和媒体标记。"""
    text = str(message_text or "").strip()
    if has_image_component and image_caption:
        if text:
            return append_emoji_summary_suffix(
                f"{text} [{image_label}描述: {image_caption}]", emoji_summary
            )
        return append_emoji_summary_suffix(
            f"{sender_name}发送了1张{image_label}：{image_caption}",
            emoji_summary,
        )
    if text:
        return append_emoji_summary_suffix(text, emoji_summary)
    media_text, _ = build_media_trigger_message(
        sender_name=sender_name,
        has_image_component=has_image_component,
        has_video_component=has_video_component,
        has_file_component=has_file_component,
        has_forward_component=has_forward_component,
        has_json_component=has_json_component,
        has_record_component=has_record_component,
        file_name=file_name,
        image_label=image_label,
    )
    return append_emoji_summary_suffix(str(media_text or "").strip(), emoji_summary)


def _clip(text: str, limit: int = 120) -> str:
    t = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(t) <= limit:
        return t
    return t[: max(20, limit - 1)] + "…"


def _is_active_wake_reason(wake_reason: str) -> bool:
    reason = str(wake_reason or "")
    active_markers = (
        "提及唤醒",
        "话题相关性",
        "答疑唤醒",
        "无聊唤醒",
        "概率唤醒",
        "动态合并跟进",
        "动态重排跟进",
    )
    return any(m in reason for m in active_markers)


def _classify_active_wake_reason(wake_reason: str) -> str:
    reason = str(wake_reason or "").strip()
    if reason.startswith("提及唤醒"):
        return "mention"
    if "话题相关" in reason:
        return "relevant"
    if reason.startswith("答疑唤醒"):
        return "ask"
    if reason.startswith("无聊唤醒"):
        return "bored"
    if reason.startswith("概率唤醒"):
        return "prob"
    return ""


def build_active_wake_prompt_note(
    *,
    direct_wake: bool,
    wake_reason: str,
) -> str:
    """构建主动唤醒的提示说明文本，供附加到请求中。"""
    if direct_wake:
        return ""
    wake_type = _classify_active_wake_reason(wake_reason)
    if not wake_type:
        return ""

    reason_map = {
        "mention": "当前内容属于对你的间接提及，提及了你的代称或与你相关的内容。",
        "relevant": "当前用户所发内容与当前话题相关，你基于话题相关性主动接话；请自然承接上下文。",
        "ask": "有用户在求解答，你决定主动为其解答或回复。",
        "bored": "有用户表示无聊，你决定活跃下气氛。",
        "prob": "你作为对话的旁听者，决定自然地参与他人正在进行中的对话。",
    }
    note_body = reason_map.get(wake_type, "").strip()
    if not note_body:
        return ""

    return f"\n\n[主动唤醒说明]\n本次回复属于主动参与，用户未显式@你。\n{note_body}"


def inject_active_wake_note_into_request(
    *,
    req: Any,
    direct_wake: bool,
    wake_reason: str,
) -> tuple[bool, str, str]:
    """将主动唤醒说明注入到请求中（优先使用 TextPart，降级到 prompt）。"""
    block = build_active_wake_prompt_note(
        direct_wake=direct_wake,
        wake_reason=wake_reason,
    ).strip()
    if not block:
        return False, "skip:not_active_wake", ""

    wake_type = _classify_active_wake_reason(wake_reason) or "other"
    if not append_text_part_to_request(req, block, mark_temp=True):
        req.prompt = f"{(req.prompt or '').strip()}\n\n{block}".strip()
    return True, f"active_wake:{wake_type}", block


def should_inject_context(
    *,
    get_cfg: Any,
    direct_wake: bool,
    wake_reason: str,
) -> bool:
    """根据配置和唤醒类型判断是否应该向请求注入上下文。"""
    mode = get_context_injection_mode(get_cfg)
    if mode == "off":
        return False
    if mode == "before_all_wake":
        return True
    # before_active_only 模式
    if direct_wake:
        return False
    return _is_active_wake_reason(wake_reason)


def _dedup_simple_notice_text(sender: str, text: str) -> str:
    sender_text = str(sender or "").strip()
    value = str(text or "").strip()
    if not sender_text or not value:
        return value
    for marker in ("通知: ", "申请: "):
        if marker not in value:
            continue
        left, right = value.split(marker, 1)
        right = right.strip()
        if not right.startswith(sender_text):
            continue
        tail = right[len(sender_text) :].lstrip()
        if tail:
            return f"{left}{marker}{tail}"
    return value


def _render_context_line(
    item: dict[str, Any], *, detailed: bool, limit: int = 120
) -> str:
    sender = str(item.get("sender_name") or item.get("uid") or "未知用户").strip()
    uid_text = str(item.get("uid") or "").strip()
    ts = item.get("ts")
    time_text = "--:--:--"
    try:
        if ts is not None:
            time_text = datetime.datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        pass
    if not detailed:
        source_text = str(item.get("source") or "").strip().lower()
        text_raw = str(item.get("text") or "")
        is_notice_line = source_text.startswith("notice_") or source_text.startswith(
            "request_"
        )
        sender_with_uid = f"{sender}(uid={uid_text})" if uid_text else sender
        if is_notice_line:
            text = _clip(text_raw, limit)
            return f"[{time_text}] {sender_with_uid}: {text}"
        text = _clip(_dedup_simple_notice_text(sender, text_raw), limit)
        reply_preview = _clip(str(item.get("reply_preview") or ""), 80)
        suffix = f" | 引用: {reply_preview}" if reply_preview else ""
        return f"[{time_text}] {sender_with_uid}: {text}{suffix}"

    detail_parts: list[str] = [f"sender={sender}"]
    msg_id_text = str(item.get("msg_id") or "").strip()
    source_text = str(item.get("source") or "").strip()
    reply_to_id = str(item.get("reply_to_id") or "").strip()
    reply_msg_id = str(item.get("reply_msg_id") or "").strip()
    reply_preview = str(item.get("reply_preview") or "").strip()
    text_value = str(item.get("text") or "").strip()

    if uid_text:
        detail_parts.append(f"uid={uid_text}")
    if msg_id_text:
        detail_parts.append(f"msg_id={msg_id_text}")
    if bool(item.get("is_bot", False)):
        detail_parts.append("is_bot=True")
    if source_text:
        detail_parts.append(f"source={source_text}")
    if bool(item.get("at_bot", False)):
        detail_parts.append("at_bot=True")
    if bool(item.get("at_all", False)):
        detail_parts.append("at_all=True")

    at_targets = item.get("at_targets") or []
    if at_targets:
        try:
            at_targets_text = json.dumps(
                at_targets, ensure_ascii=False, separators=(",", ":")
            )
            detail_parts.append(f"at_targets={at_targets_text}")
        except Exception:
            pass
    if reply_to_id:
        detail_parts.append(f"reply_to_id={reply_to_id}")
    if reply_msg_id:
        detail_parts.append(f"reply_msg_id={reply_msg_id}")
    if reply_preview:
        detail_parts.append(f"reply_preview={reply_preview}")
    if text_value:
        detail_parts.append(f"text={text_value}")

    return f"[{time_text}] " + " ".join(detail_parts)


def _raw_value(raw: Any, key: str, default: Any = None) -> Any:
    if raw is None:
        return default
    try:
        if isinstance(raw, dict):
            return raw.get(key, default)
        if hasattr(raw, "get"):
            return raw.get(key, default)
        return getattr(raw, key, default)
    except Exception:
        return default


def _resolve_uid_label(group_state: GroupState, uid: str) -> str:
    uid_text = str(uid or "").strip()
    if not uid_text:
        return "某成员"
    for item in reversed(list(group_state.context_messages or [])):
        if str(item.get("uid") or "").strip() != uid_text:
            continue
        sender_name = str(item.get("sender_name") or "").strip()
        if sender_name:
            return sender_name
    return f"用户{uid_text}"


def _extract_poke_action_text(raw_info: Any) -> str:
    if not isinstance(raw_info, list):
        return ""
    texts: list[str] = []
    for node in raw_info:
        if not isinstance(node, dict):
            continue
        if str(node.get("type") or "").strip().lower() != "nor":
            continue
        txt = str(node.get("txt") or "").strip()
        if txt:
            texts.append(txt)
    if not texts:
        return ""
    return _clip("".join(texts), 80)


def _extract_poke_action_parts(raw_info: Any) -> list[str]:
    if not isinstance(raw_info, list):
        return []
    texts: list[str] = []
    for node in raw_info:
        if not isinstance(node, dict):
            continue
        if str(node.get("type") or "").strip().lower() != "nor":
            continue
        txt = str(node.get("txt") or "").strip()
        if txt:
            texts.append(txt)
    return texts


def append_notice_context_from_raw(
    *,
    group_state: GroupState,
    raw_message: Any,
    get_cfg: Any,
) -> tuple[bool, str]:
    """从原始消息中解析系统通知（如群成员变动）并追加到群组上下文。返回（是否成功, 来源标识）。"""
    if not is_context_injection_enabled(get_cfg):
        return False, "context_disabled"

    post_type = str(_raw_value(raw_message, "post_type") or "").strip().lower()
    if post_type not in {"notice", "request"}:
        return False, "not_notice_or_request"

    group_id = str(_raw_value(raw_message, "group_id") or "").strip()
    if not group_id:
        return False, "no_group_id"

    uid_text = ""
    sender_name = "系统通知"
    message_text = ""
    msg_id = ""
    reply_to_id = ""
    source = ""
    parse_features: list[str] = []

    raw_ts = _raw_value(raw_message, "time")
    try:
        now_ts = float(raw_ts) if raw_ts is not None else time.time()
    except Exception:
        now_ts = time.time()

    if post_type == "notice":
        notice_type = str(_raw_value(raw_message, "notice_type") or "").strip().lower()
        sub_type = str(_raw_value(raw_message, "sub_type") or "").strip().lower()
        user_id = str(_raw_value(raw_message, "user_id") or "").strip()
        operator_id = str(_raw_value(raw_message, "operator_id") or "").strip()
        sender_id = str(_raw_value(raw_message, "sender_id") or "").strip()
        target_id = str(_raw_value(raw_message, "target_id") or "").strip()
        source = f"notice_{notice_type or 'unknown'}"
        parse_features = ["notice"]
        if notice_type:
            parse_features.append(notice_type)

        if notice_type in {"group_recall", "friend_recall"}:
            msg_id = str(_raw_value(raw_message, "message_id") or "").strip()
            actor_uid = operator_id or user_id
            actor_label = _resolve_uid_label(group_state, actor_uid)
            target_label = _resolve_uid_label(group_state, user_id)
            uid_text = actor_uid
            sender_name = actor_label
            if actor_uid and user_id and actor_uid != user_id:
                message_text = (
                    f"撤回通知: {actor_label}撤回了 {target_label} 的一条消息"
                )
                reply_to_id = user_id
            else:
                message_text = f"撤回通知: {actor_label}撤回了自己的一条消息"
        elif notice_type == "group_increase":
            target_label = _resolve_uid_label(group_state, user_id)
            reply_to_id = user_id
            if sub_type == "invite":
                actor_uid = operator_id
                actor_label = (
                    _resolve_uid_label(group_state, actor_uid)
                    if actor_uid
                    else "管理员"
                )
                uid_text = actor_uid or user_id
                sender_name = actor_label
                message_text = f"入群通知: {actor_label}邀请 {target_label} 加入了群聊"
            elif sub_type == "approve":
                uid_text = user_id
                sender_name = target_label
                message_text = f"入群通知: {target_label}申请入群并已通过"
            else:
                uid_text = user_id
                sender_name = target_label
                message_text = f"入群通知: {target_label}加入了群聊"
        elif notice_type == "group_decrease":
            target_label = _resolve_uid_label(group_state, user_id)
            reply_to_id = user_id
            if sub_type == "kick":
                actor_uid = operator_id if operator_id and operator_id != "0" else ""
                actor_label = (
                    _resolve_uid_label(group_state, actor_uid)
                    if actor_uid
                    else "管理员"
                )
                uid_text = actor_uid or user_id
                sender_name = actor_label
                message_text = f"退群通知: {actor_label}将 {target_label} 移出了群聊"
            elif sub_type == "leave":
                uid_text = user_id
                sender_name = target_label
                message_text = f"退群通知: {target_label}退出了群聊"
            elif sub_type == "kick_me":
                actor_uid = operator_id if operator_id and operator_id != "0" else ""
                actor_label = (
                    _resolve_uid_label(group_state, actor_uid)
                    if actor_uid
                    else "管理员"
                )
                uid_text = actor_uid or user_id
                sender_name = actor_label
                message_text = f"退群通知: {actor_label}将机器人移出了群聊"
            else:
                uid_text = user_id
                sender_name = target_label
                message_text = f"退群通知: {target_label}离开了群聊"
        elif notice_type == "group_admin":
            target_label = _resolve_uid_label(group_state, user_id)
            uid_text = operator_id or user_id
            sender_name = _resolve_uid_label(group_state, uid_text)
            reply_to_id = user_id
            if sub_type == "set":
                if operator_id and operator_id != user_id:
                    actor_label = _resolve_uid_label(group_state, operator_id)
                    message_text = (
                        f"管理通知: {actor_label}将 {target_label} 设为了管理员"
                    )
                else:
                    message_text = f"管理通知: {target_label}被设为了管理员"
            elif sub_type == "unset":
                if operator_id and operator_id != user_id:
                    actor_label = _resolve_uid_label(group_state, operator_id)
                    message_text = (
                        f"管理通知: {actor_label}取消了 {target_label} 的管理员身份"
                    )
                else:
                    message_text = f"管理通知: {target_label}被取消了管理员身份"
            else:
                message_text = f"管理通知: {target_label}管理员身份发生变更"
        elif notice_type == "group_ban":
            actor_uid = operator_id if operator_id and operator_id != "0" else ""
            actor_label = (
                _resolve_uid_label(group_state, actor_uid) if actor_uid else "管理员"
            )
            target_label = _resolve_uid_label(group_state, user_id)
            uid_text = actor_uid or user_id
            sender_name = actor_label
            reply_to_id = user_id
            try:
                duration = int(_raw_value(raw_message, "duration") or 0)
            except Exception:
                duration = 0
            if sub_type in {"lift_ban", "unban"} or duration <= 0:
                message_text = f"禁言通知: {actor_label}取消了 {target_label} 的禁言"
            else:
                message_text = (
                    f"禁言通知: {actor_label}禁言了 {target_label} {duration}秒"
                )
        elif notice_type == "notify" and sub_type == "poke":
            actor_uid = user_id
            target_uid = target_id
            actor_label = _resolve_uid_label(group_state, actor_uid)
            target_label = _resolve_uid_label(group_state, target_uid)
            raw_info = _raw_value(raw_message, "raw_info")
            action_parts = _extract_poke_action_parts(raw_info)
            action_text = _extract_poke_action_text(raw_info)
            uid_text = actor_uid
            sender_name = actor_label
            reply_to_id = target_uid
            target_display = (
                "自己"
                if (actor_uid and target_uid and actor_uid == target_uid)
                else target_label
            )
            if action_parts:
                action_head = str(action_parts[0] or "").strip()
                action_tail = str("".join(action_parts[1:]) or "").strip()
                if action_head and action_tail:
                    message_text = f"互动通知: {actor_label} {action_head} {target_display} {action_tail}"
                elif action_head:
                    message_text = (
                        f"互动通知: {actor_label} {action_head} {target_display}"
                    )
                elif action_text:
                    message_text = (
                        f"互动通知: {actor_label} {action_text} {target_display}"
                    )
                else:
                    message_text = f"互动通知: {actor_label} 戳了戳 {target_display}"
            elif action_text:
                message_text = f"互动通知: {actor_label} {action_text} {target_display}"
            else:
                message_text = f"互动通知: {actor_label} 戳了戳 {target_display}"
        elif notice_type == "essence":
            msg_id = str(_raw_value(raw_message, "message_id") or "").strip()
            actor_uid = operator_id or user_id or sender_id
            sender_uid = sender_id or user_id
            actor_label = _resolve_uid_label(group_state, actor_uid)
            sender_label = _resolve_uid_label(group_state, sender_uid)
            uid_text = actor_uid
            sender_name = actor_label
            reply_to_id = sender_uid
            if sub_type == "add":
                if actor_uid and sender_uid and actor_uid != sender_uid:
                    message_text = (
                        f"精华通知: {actor_label}将 {sender_label} 的消息设为了精华"
                    )
                else:
                    message_text = f"精华通知: {sender_label}的一条消息被设为了精华"
            elif sub_type in {"del", "delete", "remove"}:
                if actor_uid and sender_uid and actor_uid != sender_uid:
                    message_text = (
                        f"精华通知: {actor_label}取消了 {sender_label} 的精华消息"
                    )
                else:
                    message_text = f"精华通知: {sender_label}的一条精华消息被取消"
            else:
                message_text = f"精华通知: {sender_label}的精华消息发生变更"
        else:
            return False, "unsupported_notice"
    else:
        request_type = (
            str(_raw_value(raw_message, "request_type") or "").strip().lower()
        )
        sub_type = str(_raw_value(raw_message, "sub_type") or "").strip().lower()
        if request_type != "group" or sub_type != "add":
            return False, "unsupported_request"
        user_id = str(_raw_value(raw_message, "user_id") or "").strip()
        requester_label = _resolve_uid_label(group_state, user_id)
        comment = _clip(str(_raw_value(raw_message, "comment") or "").strip(), 120)
        uid_text = user_id
        sender_name = requester_label
        source = "request_group_add"
        parse_features = ["request", "group_add"]
        reply_to_id = user_id
        if comment:
            message_text = (
                f"入群申请: {requester_label}申请加入群聊，验证信息：{comment}"
            )
        else:
            message_text = f"入群申请: {requester_label}申请加入群聊"

    if not message_text:
        return False, "empty_notice_text"

    max_messages = get_context_injection_max_messages(get_cfg)
    ok = append_group_context_message(
        group_state,
        uid=uid_text,
        sender_name=sender_name,
        message_text=message_text,
        max_messages=max_messages,
        msg_id=msg_id,
        is_bot=False,
        source=source or "notice_unknown",
        at_targets=[],
        at_bot=False,
        at_all=False,
        reply_to_id=reply_to_id,
        reply_msg_id="",
        reply_preview="",
        parse_features=parse_features,
        now_ts=now_ts,
        get_cfg=get_cfg,
    )
    if not ok:
        return False, "append_failed"
    return True, source or "notice_unknown"


def inject_context_into_request(
    *,
    req: Any,
    group_state: GroupState,
    get_cfg: Any,
    direct_wake: bool,
    wake_reason: str,
) -> tuple[bool, str, str]:
    """将群组上下文消息列表格式化后注入到请求中。返回（是否成功, 标识, 内容摘要）。"""
    prune_group_context_state(group_state=group_state, get_cfg=get_cfg)
    if not should_inject_context(
        get_cfg=get_cfg, direct_wake=direct_wake, wake_reason=wake_reason
    ):
        return False, "mode_skip", ""

    context_messages = sorted(
        list(group_state.context_messages or []),
        key=lambda x: float(x.get("ts", 0)),
    )
    if not context_messages:
        return False, "no_context", ""

    max_keep = get_context_injection_max_messages(get_cfg)
    if max_keep <= 0:
        return False, "max_keep_zero", ""
    detailed = get_context_injection_format(get_cfg) == "detailed"
    max_line_len = int(get_cfg("context_line_max_length", 150) or 150)
    selected = context_messages[-max_keep:]
    lines = [
        _render_context_line(item, detailed=detailed, limit=max_line_len)
        for item in selected
        if str(item.get("text") or "").strip()
    ]
    if not lines:
        return False, "empty_lines", ""

    block = (
        f"\n\n[上下文注入|格式={'详细' if detailed else '简洁'}]\n"
        + "\n".join(lines)
        + "\n[说明] 以上为最近对话片段，请据此理解当前上下文。"
    )
    if not append_text_part_to_request(req, block, mark_temp=True):
        req.prompt = f"{(req.prompt or '').strip()}{block}".strip()
    return True, f"injected:{len(lines)}", block.strip()


def append_group_context_message(
    group_state: GroupState,
    *,
    uid: str,
    sender_name: str,
    message_text: str,
    max_messages: int,
    msg_id: str = "",
    is_bot: bool = False,
    source: str = "incoming",
    at_targets: Optional[list[dict[str, str]]] = None,
    at_bot: bool = False,
    at_all: bool = False,
    reply_to_id: str = "",
    reply_msg_id: str = "",
    reply_preview: str = "",
    parse_features: Optional[list[str]] = None,
    now_ts: Optional[float] = None,
    get_cfg: Any = None,
) -> bool:
    """向群组上下文中追加一条消息记录，超出限制时自动裁剪。返回是否成功。"""
    text = str(message_text or "").strip()
    if not text:
        return False
    ts = float(now_ts if now_ts is not None else time.time())
    entry: dict[str, Any] = {
        "ts": ts,
        "uid": str(uid or "").strip(),
        "sender_name": str(sender_name or "").strip() or "未知用户",
        "text": text,
        "source": str(source or "incoming"),
    }
    msg_id_text = str(msg_id or "").strip()
    reply_to_id_text = str(reply_to_id or "").strip()
    reply_msg_id_text = str(reply_msg_id or "").strip()
    reply_preview_text = str(reply_preview or "").strip()
    at_targets_list = list(at_targets or [])
    if msg_id_text:
        entry["msg_id"] = msg_id_text
    if bool(is_bot):
        entry["is_bot"] = True
    if at_targets_list:
        entry["at_targets"] = at_targets_list
    if bool(at_bot):
        entry["at_bot"] = True
    if bool(at_all):
        entry["at_all"] = True
    if reply_to_id_text:
        entry["reply_to_id"] = reply_to_id_text
    if reply_msg_id_text:
        entry["reply_msg_id"] = reply_msg_id_text
    if reply_preview_text:
        entry["reply_preview"] = reply_preview_text
    features: list[str] = []
    for item in list(parse_features or []):
        v = str(item or "").strip().lower()
        if v and v not in features:
            features.append(v)
    if features:
        entry["parse_features"] = features
    group_state.context_messages.append(entry)
    if (not bool(is_bot)) and entry["uid"]:
        group_state.last_user_interaction[entry["uid"]] = ts
    if get_cfg:
        prune_group_context_state(group_state=group_state, get_cfg=get_cfg, now_ts=ts)
    else:
        max_len = int(max_messages)
        if max_len <= 0:
            return False
        if len(group_state.context_messages) > max_len:
            group_state.context_messages = group_state.context_messages[-max_len:]
    logger.debug(
        "[上下文注入] 记录上下文："
        f"source={entry.get('source', 'incoming')}, uid={entry['uid'] or 'unknown'}, "
        f"total={len(group_state.context_messages)}",
    )
    return True


def record_wake_judge_reject_context(
    event: AstrMessageEvent,
    gid: str,
    uid: str,
    msg: str,
    get_cfg: Callable[[str, Any], Any],
) -> None:
    """唤醒判定拒绝时记录一条上下文，避免后续将“主动不回应”误判为消息丢失。

    仅在上下文注入开启时记录；同一用户连续多次拒绝只保留首条，避免刷屏占用上下文窗口。
    """
    if not gid or not is_context_injection_enabled(get_cfg):
        return
    g = StateManager.get_group(gid)
    last = g.context_messages[-1] if g.context_messages else None
    if (
        last
        and str(last.get("source") or "") == "request_judge_reject"
        and str(last.get("target_uid") or "") == str(uid or "")
    ):
        return
    sender_name = event.get_sender_name() or str(uid) or "某成员"
    preview = str(msg or "").strip()
    if len(preview) > 20:
        preview = preview[:20] + "…"
    append_group_context_message(
        g,
        uid=str(event.get_self_id() or "bot"),
        sender_name="Bot",
        message_text=f"已收到{sender_name}的消息（{preview}），但经唤醒判定主动选择不回应，未回复。",
        max_messages=get_context_injection_max_messages(get_cfg),
        is_bot=True,
        source="request_judge_reject",
        now_ts=time.time(),
        get_cfg=get_cfg,
    )
    if (
        g.context_messages
        and g.context_messages[-1].get("source") == "request_judge_reject"
    ):
        g.context_messages[-1]["target_uid"] = str(uid or "")


def clear_group_context_records(group_state: GroupState) -> int:
    """清空群组的所有上下文记录，返回被移除的消息数量。"""
    removed = len(group_state.context_messages or [])
    group_state.context_messages = []
    group_state.last_user_interaction = {}
    group_state.context_bot_last_replied_to_uid = ""
    return removed


def clear_context_records_for_group(
    *,
    group_state: GroupState,
    effective_history: Any,
    umo: str,
) -> tuple[int, int]:
    """清除群组的上下文记录和有效对话历史，返回（上下文条数, 历史条数）。"""
    removed = clear_group_context_records(group_state)
    effective_removed = clear_effective_dialog_history(effective_history, umo)
    gid = str(getattr(group_state, "gid", "") or "")
    scope_label = f"群({gid})" if gid else "群(unknown)"
    logger.debug(
        "[LLMEnhancement] 清除上下文完成："
        f"{scope_label}, removed={removed}, effective_removed={effective_removed}",
    )
    return removed, effective_removed
