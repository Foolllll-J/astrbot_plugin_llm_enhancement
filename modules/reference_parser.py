import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

from .file_parser import extract_file_infos_from_chain
from .json_parser import extract_json_infos_from_chain, parse_json_segment_data

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
    blocked: bool = False


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


def _append_prompt_context(req: ProviderRequest, details: list[str]) -> None:
    if not details:
        return
    req.prompt += (
        "\n\n[引用内容补充] "
        + "；".join([str(x).strip() for x in details if str(x).strip()])
        + "。请结合这些补充信息回答。"
    )


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

        has_extracted_media = False
        quoted_file_names: list[str] = []
        quoted_file_infos: list[str] = []
        quoted_json_infos: list[str] = []
        max_size = get_cfg("video_max_size_mb", 50)

        if not hasattr(req, "image_urls") or req.image_urls is None:
            req.image_urls = []
        if not hasattr(req, "_cleanup_paths"):
            req._cleanup_paths = []

        allow_file_name_inject = bool(get_cfg("inject_file_name", True))
        try:
            file_text_inject_len = max(0, int(get_cfg("inject_file_text_length", 0) or 0))
        except (TypeError, ValueError):
            file_text_inject_len = 0

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
                url = seg_data.get("url")
                if not url:
                    continue
                has_extracted_media = True
                if url.startswith(("http://", "https://")):
                    try:
                        local_path = await download_media(url, max_size)
                        if local_path and local_path not in req.image_urls:
                            req.image_urls.append(local_path)
                            req._cleanup_paths.append(local_path)
                        if local_path:
                            continue
                    except Exception:
                        pass
                if url not in req.image_urls:
                    req.image_urls.append(url)

            elif enable_file_parse and seg_type == "file":
                file_name = seg_data.get("name") or seg_data.get("file_name") or seg_data.get("file")
                if allow_file_name_inject and file_name:
                    quoted_file_names.append(str(file_name))
                if file_text_inject_len > 0:
                    file_infos = await extract_file_infos_from_chain(
                        event=event,
                        chain=[segment],
                        max_chars=file_text_inject_len,
                        cleanup_paths=req._cleanup_paths,
                    )
                    if not file_infos:
                        logger.debug(
                            "[LLMEnhancement] 引用文件未提取到文本摘要: "
                            f"file={file_name or 'unknown'}, max_chars={file_text_inject_len}"
                        )
                    for fn, excerpt in file_infos[:2]:
                        quoted_file_infos.append(f"{fn}: {excerpt}")

            elif enable_json_parse and seg_type == "json":
                inner_data = seg_data.get("data")
                if not inner_data:
                    continue
                raw_json = json.dumps(inner_data, ensure_ascii=False) if isinstance(inner_data, dict) else str(inner_data)
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

        if has_extracted_media or quoted_file_names or quoted_file_infos or quoted_json_infos:
            quoted_sender = getattr(req, "_quoted_sender", "未知用户")
            parts: list[str] = []
            if has_extracted_media:
                parts.append(f"你引用了 {quoted_sender} 发送的图片。")
            if quoted_file_names:
                parts.append(f"你引用了 {quoted_sender} 发送的文件：{'；'.join(quoted_file_names)}。")
            if quoted_file_infos:
                parts.append(f"你引用文件的内容摘要：{'；'.join(quoted_file_infos[:2])}。")
            if quoted_json_infos:
                parts.append(f"你引用卡片的关键信息：{'；'.join(quoted_json_infos[:2])}。")
            _append_prompt_context(req, parts)
            if quoted_json_infos:
                result.injected_json = True
            if quoted_file_names or quoted_file_infos:
                result.injected_file = True
            if quoted_file_infos:
                logger.debug(
                    "[LLMEnhancement] 已注入引用文件文本摘要: "
                    f"count={len(quoted_file_infos[:2])}, sender={quoted_sender}"
                )
            if quoted_json_infos:
                logger.debug(
                    "[LLMEnhancement] 已注入引用JSON卡片摘要: "
                    f"count={len(quoted_json_infos[:2])}, sender={quoted_sender}"
                )

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
                [f"你引用了 {quoted_sender} 的 JSON 卡片，提取到文本：{'；'.join(parsed_reply.json_extracted_texts[:5])}。"],
            )
            result.injected_json = True
            logger.debug(
                "[LLMEnhancement] 已注入引用JSON正文摘录: "
                f"count={len(parsed_reply.json_extracted_texts[:5])}, sender={quoted_sender}"
            )

    if bool(get_cfg("json_parse_enable", True)):
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
            logger.debug(
                "[LLMEnhancement] 已注入当前消息JSON摘要: "
                f"info_count={len(direct_json_infos[:2])}, news_count={len(direct_json_news_texts[:5])}, "
                f"sender={sender_name}"
            )
        elif any(
            isinstance(seg, Comp.Json)
            or (isinstance(seg, dict) and str(seg.get("type") or "").lower() == "json")
            for seg in chain_for_json
        ):
            logger.debug("[LLMEnhancement] 检测到JSON组件，但未提取到可注入信息。")
    else:
        logger.debug("[LLMEnhancement] JSON注入关闭: json_parse_enable=false")

    if bool(get_cfg("file_parse_enable", True)):
        try:
            file_text_inject_len = max(0, int(get_cfg("inject_file_text_length", 0) or 0))
        except (TypeError, ValueError):
            file_text_inject_len = 0
        if file_text_inject_len <= 0:
            logger.debug("[LLMEnhancement] 文件文本注入关闭: inject_file_text_length<=0")
        if file_text_inject_len > 0:
            chain_for_file = _build_chain(event, all_components)
            if chain_for_file:
                if not hasattr(req, "_cleanup_paths"):
                    req._cleanup_paths = []
                direct_file_infos = await extract_file_infos_from_chain(
                    event=event,
                    chain=chain_for_file,
                    max_chars=file_text_inject_len,
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
                    logger.debug(
                        "[LLMEnhancement] 已注入当前消息文件文本摘要: "
                        f"count={len(direct_file_infos[:2])}, sender={sender_name}"
                    )
                elif any(isinstance(seg, Comp.File) or (isinstance(seg, dict) and str(seg.get('type') or '').lower() == 'file') for seg in chain_for_file):
                    logger.debug(
                        "[LLMEnhancement] 检测到文件组件，但未提取到可注入文本摘要: "
                        f"max_chars={file_text_inject_len}"
                    )

    return result
