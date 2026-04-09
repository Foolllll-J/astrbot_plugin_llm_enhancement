import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

from .file_parser import extract_file_infos_from_chain
from .json_parser import (
    extract_json_infos_from_chain,
    parse_forward_card_info_from_json_segment_data,
    parse_json_segment_data,
)
from .url_parser import extract_url_infos_from_chain
from .runtime_helpers import transcribe_record_segment, transcribe_record_from_chain, cleanup_paths_later

from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent

@dataclass
class ReplyParseResult:
    forward_id: Optional[str]
    json_extracted_texts: list[str] = field(default_factory=list)
    injected_json: bool = False
    injected_file: bool = False
    blocked: bool = False


@dataclass
class ReferenceContextResult:
    reply_seg: Optional[Comp.Reply] = None
    forward_id: Optional[str] = None
    injected_json: bool = False
    injected_file: bool = False
    injected_url: bool = False
    blocked: bool = False


async def _fetch_messages_by_ids(
    event: AstrMessageEvent,
    msg_ids: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(event, AiocqhttpMessageEvent):
        return []
    if not msg_ids:
        return []
    try:
        client = event.bot
    except Exception:
        return []

    messages: list[dict[str, Any]] = []
    for msg_id in msg_ids:
        try:
            original_msg = await client.api.call_action("get_msg", message_id=msg_id)
        except Exception:
            continue
        if isinstance(original_msg, dict) and isinstance(original_msg.get("message"), list):
            messages.append(original_msg)
    return messages


def _build_chain(event: AstrMessageEvent, all_components: list[Any]) -> list[Any]:
    chain: list[Any] = []
    if (
        hasattr(event, "message_obj")
        and hasattr(event.message_obj, "message")
        and isinstance(event.message_obj.message, list)
    ):
        chain.extend(event.message_obj.message)
    if isinstance(all_components, list) and all_components:
        chain.extend(all_components)
    return chain


def _append_context_block(
    req: ProviderRequest,
    details: list[str],
    *,
    block_title: str,
    log_title: str,
) -> None:
    if not details:
        return
    normalized_details = [str(x).strip().rstrip("。；;，,") for x in details if str(x).strip()]
    if not normalized_details:
        return
    injected_text = (
        f"\n\n[{block_title}] "
        + "；".join(normalized_details)
        + "。请结合这些补充信息回答。"
    )
    req.prompt += injected_text
    logger.debug("[LLMEnhancement] %s: injected=%s", log_title, injected_text.strip())


def _append_prompt_context(req: ProviderRequest, details: list[str]) -> None:
    _append_context_block(
        req,
        details,
        block_title="引用内容补充",
        log_title="引用上下文注入完成",
    )


def _segment_is_emoji_image(seg_data: dict[str, Any]) -> bool:
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
        if (
            package.get("id")
            or package.get("package_id")
            or package.get("packageId")
        ):
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
        if isinstance(value, str) and value.strip().lower() in {"emoji", "sticker", "face", "meme"}:
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


def _normalize_emoji_summary(summary: str) -> str:
    text = str(summary or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text


def _extract_emoji_summary(seg_data: dict[str, Any]) -> str:
    return _normalize_emoji_summary(str(seg_data.get("summary") or ""))


def _append_emoji_summary_suffix(base_text: str, emoji_summary: str) -> str:
    base = str(base_text or "").strip()
    summary = _normalize_emoji_summary(emoji_summary)
    if not base or not summary:
        return base
    if summary in base:
        return base
    return f"{base}（表情：{summary}）"


def _describe_image_segment(seg_data: dict[str, Any]) -> str:
    label = "表情包" if _segment_is_emoji_image(seg_data) else "图片"
    if label == "表情包":
        return _append_emoji_summary_suffix(label, _extract_emoji_summary(seg_data))
    return label


def _extract_raw_message_segments(event: AstrMessageEvent) -> list[dict[str, Any]]:
    raw_message = getattr(getattr(event, "message_obj", None), "raw_message", None)
    if raw_message is None:
        return []
    try:
        segments = raw_message.get("message")
    except Exception:
        segments = getattr(raw_message, "message", None)
    if isinstance(segments, list):
        return [seg for seg in segments if isinstance(seg, dict)]
    return []


def _extract_chain_image_datas(event: AstrMessageEvent) -> list[dict[str, Any]]:
    chain = getattr(getattr(event, "message_obj", None), "message", None)
    if not isinstance(chain, list):
        return []

    image_datas: list[dict[str, Any]] = []
    for comp in chain:
        if not isinstance(comp, Comp.Image):
            continue
        seg_data: dict[str, Any] = {}
        for attr in ("subType", "summary", "file", "url"):
            value = getattr(comp, attr, None)
            if value not in (None, ""):
                key = "sub_type" if attr == "subType" else attr
                seg_data[key] = value
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
        if seg_data:
            image_datas.append(seg_data)
    return image_datas


async def inject_current_message_image_context(
    event: AstrMessageEvent,
    req: ProviderRequest,
) -> bool:
    raw_segments = _extract_raw_message_segments(event)
    chain_image_datas = _extract_chain_image_datas(event)
    image_labels: list[str] = []
    for segment in raw_segments:
        if str(segment.get("type") or "").lower() != "image":
            continue
        seg_data = segment.get("data", {}) or {}
        label = _describe_image_segment(seg_data)
        if label not in image_labels:
            image_labels.append(label)

    if not image_labels:
        for seg_data in chain_image_datas:
            label = _describe_image_segment(seg_data)
            if label not in image_labels:
                image_labels.append(label)

    if not image_labels:
        return False

    image_desc = "和".join(image_labels)
    _append_context_block(
        req,
        [f"当前用户发送了{image_desc}"],
        block_title="消息内容补充",
        log_title="当前消息媒体注入完成",
    )
    return True


def _format_forward_time(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(value)
    try:
        ts = int(value)
        if ts > 0:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return ""


def _extract_telegram_forward_origin_meta(event: AstrMessageEvent) -> dict[str, str]:
    """提取 Telegram 转发来源元信息（最小且稳定）。"""
    raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
    msg = getattr(raw, "message", None)
    if msg is None:
        return {}

    meta: dict[str, str] = {}

    # 优先使用 Telegram 新字段 forward_origin.sender_user
    origin = getattr(msg, "forward_origin", None)
    if origin is not None:
        origin_type = str(getattr(origin, "type", "") or "").strip().lower()
        type_map = {
            "user": "用户",
            "chat": "聊天",
            "channel": "频道",
            "hidden_user": "匿名用户",
        }
        if origin_type:
            meta["source_type"] = type_map.get(origin_type, origin_type)
        forward_time = _format_forward_time(getattr(origin, "date", None))
        if forward_time:
            meta["source_time"] = forward_time

        sender_user = getattr(origin, "sender_user", None)
        if sender_user is not None:
            username = str(getattr(sender_user, "username", "") or "").strip()
            first_name = str(getattr(sender_user, "first_name", "") or "").strip()
            if username or first_name:
                meta["source_name"] = username or first_name
            is_bot = getattr(sender_user, "is_bot", None)
            if isinstance(is_bot, bool):
                meta["source_is_bot"] = "是" if is_bot else "否"
            return meta

    api_kwargs = getattr(msg, "api_kwargs", None)
    if isinstance(api_kwargs, dict):
        # 兼容日志中常见旧形态 forward_from
        source_time = _format_forward_time(api_kwargs.get("forward_date"))
        if source_time and ("source_time" not in meta):
            meta["source_time"] = source_time

        ff = api_kwargs.get("forward_from")
        if isinstance(ff, dict):
            username = str(ff.get("username") or "").strip()
            first_name = str(ff.get("first_name") or "").strip()
            if username or first_name:
                meta["source_name"] = username or first_name
            is_bot = ff.get("is_bot")
            if isinstance(is_bot, bool):
                meta["source_is_bot"] = "是" if is_bot else "否"
            if "source_type" not in meta:
                meta["source_type"] = "用户"

    return meta


async def inject_current_message_forward_origin_context(
    event: AstrMessageEvent,
    req: ProviderRequest,
) -> bool:
    """默认注入 Telegram 转发来源信息。"""
    meta = _extract_telegram_forward_origin_meta(event)
    if not meta:
        return False

    details: list[str] = ["当前消息为转发消息"]
    source_type = str(meta.get("source_type") or "").strip()
    source_name = str(meta.get("source_name") or "").strip()
    source_time = str(meta.get("source_time") or "").strip()
    source_is_bot = str(meta.get("source_is_bot") or "").strip()
    if source_type:
        details.append(f"来源类型为{source_type}")
    if source_name:
        details.append(f"来源为{source_name}")
    if source_time:
        details.append(f"原始发送时间为{source_time}")
    if source_is_bot:
        details.append(f"来源是否机器人：{source_is_bot}")

    if len(details) <= 1:
        return False

    _append_context_block(
        req,
        details,
        block_title="消息来源补充",
        log_title="转发来源注入完成",
    )
    return True


async def check_self_reply_block(
    event: AstrMessageEvent,
    reply_seg: Comp.Reply,
    get_cfg: Callable[[str, Any], Any],
    original_msg: Optional[dict[str, Any]] = None,
) -> tuple[bool, str]:
    """检测引用是否为 Bot 自身消息并按概率拦截。返回 (blocked, reason)。"""
    if not isinstance(event, AiocqhttpMessageEvent):
        return False, ""
    if not isinstance(reply_seg, Comp.Reply):
        return False, ""

    reply_id = str(getattr(reply_seg, "id", "") or "").strip()
    if not reply_id:
        return False, ""
    reply_sender_id = str(getattr(reply_seg, "sender_id", "") or "").strip()
    self_id = str(event.get_self_id())
    if reply_sender_id and reply_sender_id != self_id:
        return False, ""

    try:
        if original_msg is None:
            client = event.bot
            fetched = await client.api.call_action("get_msg", message_id=reply_id)
            original_msg = fetched if isinstance(fetched, dict) else None
    except Exception as e:
        logger.debug(
            f"[LLMEnhancement] 引用 Bot 自身检测：获取引用消息失败，改为放行。msg_id={reply_id}, err={e}"
        )
        return False, "get_msg_failed"

    if not original_msg or "message" not in original_msg:
        return False, "invalid_original_msg"

    sender_info = original_msg.get("sender", {}) if isinstance(original_msg.get("sender", {}), dict) else {}
    original_sender = str(sender_info.get("user_id", "") or "").strip()
    if original_sender != self_id:
        return False, ""

    enable_video_parse = bool(get_cfg("video_parse_enable", True))
    enable_forward_parse = bool(get_cfg("forward_parse_enable", True))
    enable_file_parse = bool(get_cfg("file_parse_enable", True))
    enable_json_parse = bool(get_cfg("json_parse_enable", True))

    message_payload = original_msg.get("message")
    message_list = message_payload if isinstance(message_payload, list) else []

    has_video = any(
        isinstance(seg, dict) and seg.get("type") == "video" for seg in message_list
    )
    has_file = any(
        isinstance(seg, dict) and seg.get("type") == "file" for seg in message_list
    )
    has_forward = any(
        isinstance(seg, dict) and seg.get("type") == "forward" for seg in message_list
    )

    if (not has_forward) and enable_forward_parse and enable_json_parse:
        for seg in message_list:
            if not isinstance(seg, dict) or seg.get("type") != "json":
                continue
            inner_data = (seg.get("data") or {}).get("data")
            if not inner_data:
                continue
            raw_json = (
                json.dumps(inner_data, ensure_ascii=False)
                if isinstance(inner_data, dict)
                else str(inner_data)
            )
            is_forward_card, _ = parse_forward_card_info_from_json_segment_data(raw_json)
            if is_forward_card:
                has_forward = True
                break

    try:
        video_prob = float(get_cfg("quote_self_video_block_prob", 0) or 0)
    except Exception:
        video_prob = 0.0
    try:
        file_prob = float(get_cfg("quote_self_file_block_prob", 0) or 0)
    except Exception:
        file_prob = 0.0
    try:
        forward_prob = float(get_cfg("quote_self_forward_block_prob", 0) or 0)
    except Exception:
        forward_prob = 0.0

    if enable_video_parse and has_video and video_prob > 0 and random.random() < video_prob:
        reason = f"引用 Bot 自己的视频，按概率拦截（{video_prob}）"
        logger.debug(f"[LLMEnhancement] {reason}，msg_id={reply_id}")
        return True, reason

    if enable_file_parse and has_file and file_prob > 0 and random.random() < file_prob:
        reason = f"引用 Bot 自身的文件，按概率拦截（{file_prob}）"
        logger.debug(f"[LLMEnhancement] {reason}，msg_id={reply_id}")
        return True, reason

    if enable_forward_parse and has_forward and forward_prob > 0 and random.random() < forward_prob:
        reason = f"引用 Bot 自身的转发，按概率拦截（{forward_prob}）"
        logger.debug(f"[LLMEnhancement] {reason}，msg_id={reply_id}")
        return True, reason

    return False, ""


async def parse_reply_context(
    event: AstrMessageEvent,
    context: Any,
    req: ProviderRequest,
    reply_seg: Comp.Reply,
    forward_id: Optional[str],
    get_cfg: Callable[[str, Any], Any],
    download_media: Callable[[str, float], Any],
) -> ReplyParseResult:
    """解析引用消息，提取 forward_id、JSON 信息、图片与文件上下文。"""
    result = ReplyParseResult(forward_id=forward_id)
    try:
        client = event.bot
        original_msg = await client.api.call_action("get_msg", message_id=reply_seg.id)
        if not (original_msg and "message" in original_msg):
            return result

        sender_info = original_msg.get("sender", {})
        original_sender_name = sender_info.get("nickname", "未知用户")

        enable_forward_parse = bool(get_cfg("forward_parse_enable", True))
        enable_file_parse = bool(get_cfg("file_parse_enable", True))
        enable_json_parse = bool(get_cfg("json_parse_enable", True))
        enable_record_parse = bool(get_cfg("record_parse_enable", True))

        setattr(req, "_quoted_sender", original_sender_name)
        original_message_chain = original_msg["message"]
        if not isinstance(original_message_chain, list):
            return result

        quoted_image_labels: list[str] = []
        quoted_file_names: list[str] = []
        quoted_file_infos: list[str] = []
        quoted_file_parse_failed = False
        quoted_json_infos: list[str] = []
        quoted_record_texts: list[str] = []
        quoted_record_count = 0

        if not hasattr(req, "image_urls") or req.image_urls is None:
            req.image_urls = []
        if not hasattr(req, "_cleanup_paths"):
            req._cleanup_paths = []

        allow_file_name_inject = bool(get_cfg("inject_file_name", True))
        try:
            file_text_inject_len = max(0, int(get_cfg("inject_file_text_length", 0) or 0))
        except (TypeError, ValueError):
            file_text_inject_len = 0
        try:
            file_inject_max_size_mb = max(0, int(get_cfg("inject_file_max_size_mb", 20) or 0))
        except (TypeError, ValueError):
            file_inject_max_size_mb = 20

        for segment in original_message_chain:
            if not isinstance(segment, dict):
                continue
            seg_type = segment.get("type")
            seg_data = segment.get("data", {}) or {}

            if enable_forward_parse and seg_type == "forward":
                new_forward_id = seg_data.get("id")
                if new_forward_id:
                    result.forward_id = new_forward_id
                    break

            elif seg_type == "image":
                label = _describe_image_segment(seg_data)
                if label not in quoted_image_labels:
                    quoted_image_labels.append(label)

            elif seg_type == "record":
                quoted_record_count += 1
                if enable_record_parse:
                    asr_text, cleanup_paths = await transcribe_record_segment(
                        context=context,
                        get_cfg=get_cfg,
                        event=event,
                        segment=segment,
                    )
                    if cleanup_paths:
                        req._cleanup_paths.extend(cleanup_paths)
                    if asr_text:
                        quoted_record_texts.append(asr_text)

            elif enable_file_parse and seg_type == "file":
                file_name = seg_data.get("name") or seg_data.get("file_name") or seg_data.get("file")
                if allow_file_name_inject and file_name:
                    quoted_file_names.append(str(file_name))
                if file_text_inject_len > 0:
                    file_failure_details: list[str] = []
                    file_infos = await extract_file_infos_from_chain(
                        event=event,
                        chain=[segment],
                        max_chars=file_text_inject_len,
                        max_file_size_mb=file_inject_max_size_mb,
                        cleanup_paths=req._cleanup_paths,
                        failure_details=file_failure_details,
                    )
                    if not file_infos:
                        logger.debug(
                            "[LLMEnhancement] 引用文件未提取到文本摘要: "
                            f"file={file_name or 'unknown'}, max_chars={file_text_inject_len}, "
                            f"max_file_size_mb={file_inject_max_size_mb}"
                        )
                        if file_failure_details:
                            quoted_file_parse_failed = True
                    for fn, excerpt in file_infos[:2]:
                        quoted_file_infos.append(f"{fn}: {excerpt}")

            elif enable_json_parse and seg_type == "json":
                inner_data = seg_data.get("data")
                if not inner_data:
                    continue
                raw_json = json.dumps(inner_data, ensure_ascii=False) if isinstance(inner_data, dict) else str(inner_data)
                is_forward_card, forward_id_from_json = parse_forward_card_info_from_json_segment_data(raw_json)
                if enable_forward_parse and is_forward_card:
                    if forward_id_from_json:
                        result.forward_id = forward_id_from_json
                        logger.debug(
                            "[LLMEnhancement] 引用JSON识别为合并转发卡片，已提取 forward_id: "
                            f"{forward_id_from_json}"
                        )
                        break
                    logger.debug(
                        "[LLMEnhancement] 引用JSON识别为合并转发卡片，但未提取到 forward_id，"
                        "跳过 JSON 摘要注入。"
                    )
                    continue
                forward_news_texts, key_info = parse_json_segment_data(raw_json)
                if key_info:
                    quoted_json_infos.append(key_info)
                if forward_news_texts:
                    result.json_extracted_texts.extend(forward_news_texts)
                if key_info or forward_news_texts:
                    logger.debug(
                        "[LLMEnhancement] 引用JSON解析命中: "
                        f"has_key_info={bool(key_info)}, news_count={len(forward_news_texts)}"
                    )

        if quoted_image_labels or quoted_file_names or quoted_file_infos or quoted_json_infos or quoted_record_count:
            quoted_sender = getattr(req, "_quoted_sender", "未知用户")
            parts: list[str] = []
            if quoted_image_labels:
                image_desc = "和".join(quoted_image_labels)
                parts.append(f"当前用户引用了 {quoted_sender} 发送的{image_desc}。")
            if quoted_file_names:
                parts.append(f"当前用户引用了 {quoted_sender} 发送的文件：{'；'.join(quoted_file_names)}。")
            if quoted_file_infos:
                parts.append(f"被引用文件的内容摘要：{'；'.join(quoted_file_infos[:2])}。")
            elif quoted_file_parse_failed:
                parts.append("被引用文件未能成功解析，已跳过内容摘要注入。")
            if quoted_json_infos:
                parts.append(f"被引用卡片的关键信息：{'；'.join(quoted_json_infos[:2])}。")
            if quoted_record_count:
                if quoted_record_texts:
                    parts.append(f"当前用户引用了 {quoted_sender} 发送的语音，转写内容：{ ' / '.join(quoted_record_texts[:2]) }。")
                else:
                    parts.append(f"当前用户引用了 {quoted_sender} 发送的语音。")
            _append_prompt_context(req, parts)
            if quoted_json_infos:
                result.injected_json = True
            if quoted_file_names or quoted_file_infos:
                result.injected_file = True

    except Exception as e:
        logger.warning(f"解析引用上下文失败: {e}")

    return result


async def process_reference_context(
    event: AstrMessageEvent,
    context: Any,
    req: ProviderRequest,
    all_components: list[Any],
    get_cfg: Callable[[str, Any], Any],
    download_media: Callable[[str, float], Any],
) -> ReferenceContextResult:
    """处理引用、JSON、文件上下文注入，不处理转发聊天记录正文。"""
    result = ReferenceContextResult()
    is_aiocqhttp_event = isinstance(event, AiocqhttpMessageEvent)
    dynamic_batch_msg_ids = event.get_extra("_llme_dynamic_batch_msg_ids", default=[]) or []
    dynamic_batch_msg_ids = [str(mid).strip() for mid in dynamic_batch_msg_ids if str(mid).strip()]
    if not is_aiocqhttp_event:
        # 非 QQ 平台不支持按 message_id 回拉历史消息，关闭该分支。
        dynamic_batch_msg_ids = []

    for seg in all_components:
        if isinstance(seg, Comp.Forward) and not result.forward_id:
            result.forward_id = seg.id
            continue
        if isinstance(seg, Comp.Reply):
            result.reply_seg = seg

    if is_aiocqhttp_event and event.is_at_or_wake_command and (not result.forward_id) and result.reply_seg:
        parsed_reply = await parse_reply_context(
            event=event,
            context=context,
            req=req,
            reply_seg=result.reply_seg,
            forward_id=result.forward_id,
            get_cfg=get_cfg,
            download_media=download_media,
        )
        if parsed_reply.blocked:
            result.blocked = True
            return result
        result.forward_id = parsed_reply.forward_id
        result.injected_json = result.injected_json or parsed_reply.injected_json
        result.injected_file = result.injected_file or parsed_reply.injected_file
        if parsed_reply.json_extracted_texts:
            quoted_sender = getattr(req, "_quoted_sender", "未知用户")
            _append_prompt_context(
                req,
                [f"当前用户引用了 {quoted_sender} 的 JSON 卡片，提取到文本：{'；'.join(parsed_reply.json_extracted_texts[:5])}。"],
            )
            result.injected_json = True

    if (not dynamic_batch_msg_ids) and bool(get_cfg("json_parse_enable", True)):
        chain_for_json = _build_chain(event, all_components)
        direct_json_news_texts, direct_json_infos = extract_json_infos_from_chain(chain_for_json)
        direct_json_news_texts = list(dict.fromkeys(direct_json_news_texts))
        direct_json_infos = list(dict.fromkeys(direct_json_infos))
        if direct_json_infos or direct_json_news_texts:
            sender_name = event.get_sender_name() or "未知用户"
            parts: list[str] = []
            if direct_json_infos:
                parts.append(f"{sender_name} 发送的分享卡片信息：{'；'.join(direct_json_infos)}。")
            if direct_json_news_texts:
                parts.append(f"分享卡片正文摘录：{'；'.join(direct_json_news_texts[:5])}。")
            _append_prompt_context(req, parts)
            result.injected_json = True
        elif any(
            isinstance(seg, Comp.Json)
            or (isinstance(seg, dict) and str(seg.get("type") or "").lower() == "json")
            for seg in chain_for_json
        ):
            logger.debug("[LLMEnhancement] 检测到JSON组件，但未提取到可注入信息。")

    if (not dynamic_batch_msg_ids) and bool(get_cfg("file_parse_enable", True)):
        try:
            file_text_inject_len = max(0, int(get_cfg("inject_file_text_length", 0) or 0))
        except (TypeError, ValueError):
            file_text_inject_len = 0
        try:
            file_inject_max_size_mb = max(0, int(get_cfg("inject_file_max_size_mb", 20) or 0))
        except (TypeError, ValueError):
            file_inject_max_size_mb = 20
        if file_text_inject_len > 0:
            chain_for_file = _build_chain(event, all_components)
            if chain_for_file:
                if not hasattr(req, "_cleanup_paths"):
                    req._cleanup_paths = []
                direct_file_infos = await extract_file_infos_from_chain(
                    event=event,
                    chain=chain_for_file,
                    max_chars=file_text_inject_len,
                    max_file_size_mb=file_inject_max_size_mb,
                    cleanup_paths=req._cleanup_paths,
                )
                if direct_file_infos:
                    # 去重，避免 event.message 与 all_components 重叠导致重复注入。
                    direct_file_infos = list(dict.fromkeys(direct_file_infos))
                    sender_name = event.get_sender_name() or "未知用户"
                    _append_prompt_context(
                        req,
                        [
                            f"{sender_name} 发送的文件内容摘要："
                            + "；".join(f"{n}: {t}" for n, t in direct_file_infos)
                            + "。"
                        ],
                    )
                    result.injected_file = True
                elif any(isinstance(seg, Comp.File) or (isinstance(seg, dict) and str(seg.get('type') or '').lower() == 'file') for seg in chain_for_file):
                    logger.debug(
                        "[LLMEnhancement] 检测到文件组件，但未提取到可注入文本摘要: "
                        f"max_chars={file_text_inject_len}, max_file_size_mb={file_inject_max_size_mb}"
                    )


    if (not dynamic_batch_msg_ids) and bool(get_cfg("url_parse_enable", True)):
        chain_for_url = _build_chain(event, all_components)
        if chain_for_url or str(getattr(event, "message_str", "") or "").strip():
            try:
                timeout_sec = max(2, int(get_cfg("inject_url_timeout_sec", 8) or 8))
            except (TypeError, ValueError):
                timeout_sec = 8
            try:
                max_download_kb = max(32, int(get_cfg("inject_url_max_download_kb", 512) or 512))
            except (TypeError, ValueError):
                max_download_kb = 512
            block_private_network = bool(get_cfg("inject_url_block_private_network", True))
            blocked_domains_raw = get_cfg("inject_url_blocked_domains", [])
            blocked_domains = (
                [str(x or "").strip() for x in blocked_domains_raw if str(x or "").strip()]
                if isinstance(blocked_domains_raw, list)
                else []
            )
            try:
                cache_ttl_sec = max(0, int(get_cfg("inject_url_cache_ttl_sec", 600) or 600))
            except (TypeError, ValueError):
                cache_ttl_sec = 600

            url_result = await extract_url_infos_from_chain(
                event=event,
                chain=chain_for_url,
                timeout_sec=timeout_sec,
                max_download_kb=max_download_kb,
                block_private_network=block_private_network,
                blocked_domains=blocked_domains,
                cache_ttl_sec=cache_ttl_sec,
            )
            if url_result.injected and url_result.details:
                sender_name = event.get_sender_name() or "未知用户"
                _append_prompt_context(
                    req,
                    [
                        f"{sender_name} 发送的链接解析信息："
                        + "；".join(str(x).strip() for x in url_result.details[:2] if str(x).strip())
                        + "。"
                    ],
                )
                result.injected_url = True

    if dynamic_batch_msg_ids:
        original_msgs = await _fetch_messages_by_ids(event, dynamic_batch_msg_ids)
        if original_msgs:
            merged_raw_chain: list[Any] = []
            merged_sender_names: list[str] = []
            merged_forward_ids: list[str] = []
            merged_image_parts: list[str] = []

            for original_msg in original_msgs:
                sender_info = original_msg.get("sender", {}) or {}
                sender_name = str(
                    sender_info.get("nickname") or sender_info.get("card") or "未知用户"
                ).strip() or "未知用户"
                chain = original_msg.get("message") or []
                if not isinstance(chain, list):
                    continue
                merged_raw_chain.extend(chain)
                if sender_name not in merged_sender_names:
                    merged_sender_names.append(sender_name)
                for segment in chain:
                    if not isinstance(segment, dict):
                        continue
                    seg_type = str(segment.get("type") or "").lower()
                    seg_data = segment.get("data", {}) or {}
                    if seg_type == "forward":
                        forward_id = str(seg_data.get("id") or "").strip()
                        if forward_id and forward_id not in merged_forward_ids:
                            merged_forward_ids.append(forward_id)
                    elif seg_type == "json":
                        inner_data = seg_data.get("data")
                        if inner_data:
                            raw_json = json.dumps(inner_data, ensure_ascii=False) if isinstance(inner_data, dict) else str(inner_data)
                            is_forward_card, forward_id_from_json = parse_forward_card_info_from_json_segment_data(raw_json)
                            if is_forward_card and forward_id_from_json and forward_id_from_json not in merged_forward_ids:
                                merged_forward_ids.append(forward_id_from_json)
                    elif seg_type == "image":
                        label = _describe_image_segment(seg_data)
                        text = f"{sender_name} 发送的{label}"
                        if text not in merged_image_parts:
                            merged_image_parts.append(text)

            if merged_forward_ids and not result.forward_id:
                result.forward_id = merged_forward_ids[0]

            if merged_image_parts:
                _append_context_block(
                    req,
                    merged_image_parts[:3],
                    block_title="消息内容补充",
                    log_title="合并消息图片上下文注入完成",
                )

            if bool(get_cfg("json_parse_enable", True)) and merged_raw_chain:
                merged_json_news_texts, merged_json_infos = extract_json_infos_from_chain(merged_raw_chain)
                merged_json_news_texts = list(dict.fromkeys(merged_json_news_texts))
                merged_json_infos = list(dict.fromkeys(merged_json_infos))
                if merged_json_infos or merged_json_news_texts:
                    sender_text = "、".join(merged_sender_names[:3]) if merged_sender_names else (event.get_sender_name() or "未知用户")
                    parts: list[str] = []
                    if merged_json_infos:
                        parts.append(f"{sender_text} 发送的分享卡片信息：{'；'.join(merged_json_infos)}。")
                    if merged_json_news_texts:
                        parts.append(f"分享卡片正文摘录：{'；'.join(merged_json_news_texts[:5])}。")
                    _append_prompt_context(req, parts)
                    result.injected_json = True

            if bool(get_cfg("file_parse_enable", True)) and merged_raw_chain:
                try:
                    file_text_inject_len = max(0, int(get_cfg("inject_file_text_length", 0) or 0))
                except (TypeError, ValueError):
                    file_text_inject_len = 0
                try:
                    file_inject_max_size_mb = max(0, int(get_cfg("inject_file_max_size_mb", 20) or 0))
                except (TypeError, ValueError):
                    file_inject_max_size_mb = 20
                if file_text_inject_len > 0:
                    if not hasattr(req, "_cleanup_paths"):
                        req._cleanup_paths = []
                    merged_file_infos = await extract_file_infos_from_chain(
                        event=event,
                        chain=merged_raw_chain,
                        max_chars=file_text_inject_len,
                        max_file_size_mb=file_inject_max_size_mb,
                        cleanup_paths=req._cleanup_paths,
                    )
                    merged_file_infos = list(dict.fromkeys(merged_file_infos))
                    if merged_file_infos:
                        _append_prompt_context(
                            req,
                            ["合并消息中的文件内容摘要：" + "；".join(f"{n}: {t}" for n, t in merged_file_infos) + "。"],
                        )
                        result.injected_file = True

            if bool(get_cfg("url_parse_enable", True)) and merged_raw_chain:
                try:
                    timeout_sec = max(2, int(get_cfg("inject_url_timeout_sec", 8) or 8))
                except (TypeError, ValueError):
                    timeout_sec = 8
                try:
                    max_download_kb = max(32, int(get_cfg("inject_url_max_download_kb", 512) or 512))
                except (TypeError, ValueError):
                    max_download_kb = 512
                block_private_network = bool(get_cfg("inject_url_block_private_network", True))
                blocked_domains_raw = get_cfg("inject_url_blocked_domains", [])
                blocked_domains = (
                    [str(x or "").strip() for x in blocked_domains_raw if str(x or "").strip()]
                    if isinstance(blocked_domains_raw, list)
                    else []
                )
                try:
                    cache_ttl_sec = max(0, int(get_cfg("inject_url_cache_ttl_sec", 600) or 600))
                except (TypeError, ValueError):
                    cache_ttl_sec = 600
                merged_url_result = await extract_url_infos_from_chain(
                    event=event,
                    chain=merged_raw_chain,
                    timeout_sec=timeout_sec,
                    max_download_kb=max_download_kb,
                    block_private_network=block_private_network,
                    blocked_domains=blocked_domains,
                    cache_ttl_sec=cache_ttl_sec,
                )
                if merged_url_result.injected and merged_url_result.details:
                    _append_prompt_context(
                        req,
                        ["合并消息中的链接解析信息：" + "；".join(str(x).strip() for x in merged_url_result.details[:2] if str(x).strip()) + "。"],
                    )
                    result.injected_url = True
    return result


async def inject_record_asr_context(
    *,
    context: Any,
    event: AstrMessageEvent,
    req: ProviderRequest,
    all_components: list[Any],
    get_cfg: Callable[[str, Any], Any],
    cleanup_paths=cleanup_paths_later,
) -> bool:
    """在 LLM 请求阶段注入语音转写文本。"""
    record_asr_text, record_cleanup_paths = await transcribe_record_from_chain(
        context=context,
        get_cfg=get_cfg,
        event=event,
        chain=all_components,
    )
    if record_cleanup_paths:
        await cleanup_paths(record_cleanup_paths)
    if not record_asr_text:
        return False
    sender_name = event.get_sender_name() or ""
    user_question = (getattr(req, "prompt", "") or "").strip()
    sender_prefix = f"该语音消息由 {sender_name} 发送/提供。\n" if sender_name else ""
    context_prompt = (
        "\n\n以下是系统为你解析的语音转写，请结合此转写来响应用户的要求。信息如下：\n"
        "--- 注入内容开始 ---\n"
        f"{sender_prefix}[语音转写] {record_asr_text}\n"
        "--- 注入内容结束 ---"
    )
    req.prompt = (user_question + context_prompt) if user_question else context_prompt.strip()
    return True
