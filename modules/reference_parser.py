import json
import random
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

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent

    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


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
    if not (IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent)):
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

    for key in ("type", "image_type", "imageType"):
        value = seg_data.get(key)
        if isinstance(value, str) and value.strip().lower() in {"emoji", "sticker", "face", "meme"}:
            return True

    summary = seg_data.get("summary")
    if isinstance(summary, str):
        lowered = summary.lower()
        if ("表情" in summary) or ("emoji" in lowered) or ("sticker" in lowered):
            return True

    return False


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
        label = "表情包" if _segment_is_emoji_image(seg_data) else "图片"
        if label not in image_labels:
            image_labels.append(label)

    if not image_labels:
        for seg_data in chain_image_datas:
            label = "表情包" if _segment_is_emoji_image(seg_data) else "图片"
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


async def parse_reply_context(
    event: AstrMessageEvent,
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
        original_sender = str(sender_info.get("user_id", ""))
        original_sender_name = sender_info.get("nickname", "未知用户")
        self_id = str(event.get_self_id())

        enable_video_parse = bool(get_cfg("video_parse_enable", True))
        enable_forward_parse = bool(get_cfg("forward_parse_enable", True))
        enable_file_parse = bool(get_cfg("file_parse_enable", True))
        enable_json_parse = bool(get_cfg("json_parse_enable", True))

        if original_sender == self_id:
            message_list = original_msg["message"] if isinstance(original_msg["message"], list) else []
            has_video = any(s.get("type") == "video" for s in message_list if isinstance(s, dict))
            has_file = any(s.get("type") == "file" for s in message_list if isinstance(s, dict))
            has_forward = any(s.get("type") == "forward" for s in message_list if isinstance(s, dict))
            if (not has_forward) and enable_forward_parse and enable_json_parse:
                for s in message_list:
                    if not isinstance(s, dict) or s.get("type") != "json":
                        continue
                    inner_data = (s.get("data") or {}).get("data")
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

            video_prob = float(get_cfg("quote_self_video_block_prob", 0) or 0)
            file_prob = float(get_cfg("quote_self_file_block_prob", 0) or 0)
            forward_prob = float(get_cfg("quote_self_forward_block_prob", 0) or 0)

            if enable_video_parse and has_video and video_prob > 0 and random.random() < video_prob:
                logger.info(f"[LLMEnhancement] 引用 Bot 自己的视频，按概率拦截（{video_prob}）")
                result.blocked = True
                return result
            if enable_file_parse and has_file and file_prob > 0 and random.random() < file_prob:
                logger.info(f"[LLMEnhancement] 引用 Bot 自己的文件，按概率拦截（{file_prob}）")
                result.blocked = True
                return result
            if enable_forward_parse and has_forward and forward_prob > 0 and random.random() < forward_prob:
                logger.info(f"[LLMEnhancement] 引用 Bot 自己的转发，按概率拦截（{forward_prob}）")
                result.blocked = True
                return result

        setattr(req, "_quoted_sender", original_sender_name)
        original_message_chain = original_msg["message"]
        if not isinstance(original_message_chain, list):
            return result

        quoted_image_labels: list[str] = []
        quoted_file_names: list[str] = []
        quoted_file_infos: list[str] = []
        quoted_json_infos: list[str] = []

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
                label = "表情包" if _segment_is_emoji_image(seg_data) else "图片"
                if label not in quoted_image_labels:
                    quoted_image_labels.append(label)

            elif enable_file_parse and seg_type == "file":
                file_name = seg_data.get("name") or seg_data.get("file_name") or seg_data.get("file")
                if allow_file_name_inject and file_name:
                    quoted_file_names.append(str(file_name))
                if file_text_inject_len > 0:
                    file_infos = await extract_file_infos_from_chain(
                        event=event,
                        chain=[segment],
                        max_chars=file_text_inject_len,
                        max_file_size_mb=file_inject_max_size_mb,
                        cleanup_paths=req._cleanup_paths,
                    )
                    if not file_infos:
                        logger.debug(
                            "[LLMEnhancement] 引用文件未提取到文本摘要: "
                            f"file={file_name or 'unknown'}, max_chars={file_text_inject_len}, "
                            f"max_file_size_mb={file_inject_max_size_mb}"
                        )
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

        if quoted_image_labels or quoted_file_names or quoted_file_infos or quoted_json_infos:
            quoted_sender = getattr(req, "_quoted_sender", "未知用户")
            parts: list[str] = []
            if quoted_image_labels:
                image_desc = "和".join(quoted_image_labels)
                parts.append(f"当前用户引用了 {quoted_sender} 发送的{image_desc}。")
            if quoted_file_names:
                parts.append(f"当前用户引用了 {quoted_sender} 发送的文件：{'；'.join(quoted_file_names)}。")
            if quoted_file_infos:
                parts.append(f"被引用文件的内容摘要：{'；'.join(quoted_file_infos[:2])}。")
            if quoted_json_infos:
                parts.append(f"被引用卡片的关键信息：{'；'.join(quoted_json_infos[:2])}。")
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
    req: ProviderRequest,
    all_components: list[Any],
    get_cfg: Callable[[str, Any], Any],
    download_media: Callable[[str, float], Any],
) -> ReferenceContextResult:
    """处理引用、JSON、文件上下文注入，不处理转发聊天记录正文。"""
    result = ReferenceContextResult()
    if not (IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent)):
        return result
    dynamic_batch_msg_ids = event.get_extra("_llme_dynamic_batch_msg_ids", default=[]) or []
    dynamic_batch_msg_ids = [str(mid).strip() for mid in dynamic_batch_msg_ids if str(mid).strip()]

    for seg in all_components:
        if isinstance(seg, Comp.Forward) and not result.forward_id:
            result.forward_id = seg.id
            continue
        if isinstance(seg, Comp.Reply):
            result.reply_seg = seg

    if event.is_at_or_wake_command and (not result.forward_id) and result.reply_seg:
        parsed_reply = await parse_reply_context(
            event=event,
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
                parts.append(f"{sender_name} 发送的分享卡片信息：{'；'.join(direct_json_infos[:2])}。")
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
                            + "；".join(f"{n}: {t}" for n, t in direct_file_infos[:2])
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
                        label = "表情包" if _segment_is_emoji_image(seg_data) else "图片"
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
                        parts.append(f"{sender_text} 发送的分享卡片信息：{'；'.join(merged_json_infos[:2])}。")
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
                            ["合并消息中的文件内容摘要：" + "；".join(f"{n}: {t}" for n, t in merged_file_infos[:2]) + "。"],
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
