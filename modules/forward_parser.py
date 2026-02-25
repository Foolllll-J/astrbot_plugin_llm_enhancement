import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

from .json_parser import parse_json_segment_data
from .video_parser import extract_audio_wav, extract_forward_video_keyframes

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent

    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


MAX_RECURSION_DEPTH = 5
FORWARD_PARSE_CACHE_TTL_SEC = 3600
FORWARD_PARSE_CACHE_MAX_SIZE = 100

_forward_result_cache: Dict[str, Dict[str, Any]] = {}
_forward_result_cache_lock = asyncio.Lock()


def _build_forward_result_cache_key(
    forward_id: str,
    max_message_count: int,
    nested_parse_depth: int,
    json_parse_enable: bool,
    max_forward_video_count: int,
    max_forward_video_frame_count: int,
    frame_interval_sec: int,
    max_mb: int,
    asr_enabled: bool,
    max_forward_images: int,
    core_quote_image_caption_sig: str,
) -> str:
    return (
        f"v5|{forward_id}|{max_message_count}|{nested_parse_depth}|{int(json_parse_enable)}|{max_forward_video_count}|"
        f"{max_forward_video_frame_count}|{frame_interval_sec}|{max_mb}|{int(asr_enabled)}|{max_forward_images}|"
        f"{core_quote_image_caption_sig}"
    )


async def _get_cached_forward_context(cache_key: str) -> Optional[str]:
    now_ts = time.time()
    async with _forward_result_cache_lock:
        item = _forward_result_cache.get(cache_key)
        if not item:
            return None
        expire_ts = float(item.get("expire", 0.0) or 0.0)
        if expire_ts <= now_ts:
            _forward_result_cache.pop(cache_key, None)
            return None
        context_text = str(item.get("context") or "").strip()
        return context_text or None


async def _set_cached_forward_context(cache_key: str, context_text: str) -> None:
    context_text = str(context_text or "").strip()
    if not context_text:
        return
    now_ts = time.time()
    async with _forward_result_cache_lock:
        expired_keys = [
            key for key, val in _forward_result_cache.items() if float(val.get("expire", 0.0) or 0.0) <= now_ts
        ]
        for key in expired_keys:
            _forward_result_cache.pop(key, None)

        if len(_forward_result_cache) >= FORWARD_PARSE_CACHE_MAX_SIZE:
            _forward_result_cache.clear()

        _forward_result_cache[cache_key] = {
            "expire": now_ts + FORWARD_PARSE_CACHE_TTL_SEC,
            "context": context_text,
        }


def _extract_completion_text(response: Any) -> str:
    if hasattr(response, "completion_text"):
        return str(getattr(response, "completion_text") or "").strip()
    if isinstance(response, dict):
        return str(response.get("completion_text") or response.get("text") or "").strip()
    return str(response or "").strip()


def _normalize_path_list(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    for p in paths:
        sp = str(p or "").strip()
        if not sp:
            continue
        if sp not in normalized:
            normalized.append(sp)
    return normalized


def _cleanup_cache_paths(paths: list[str]) -> None:
    for p in _normalize_path_list(paths):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)
        except Exception as e:
            logger.debug(f"[LLMEnhancement] 转发临时文件清理失败: path={p}, err={e}")


async def _schedule_delayed_cleanup(paths: list[str], delay_sec: int = 60) -> None:
    normalized = _normalize_path_list(paths)
    if not normalized:
        return

    async def _cleanup_later() -> None:
        await asyncio.sleep(delay_sec)
        _cleanup_cache_paths(normalized)

    asyncio.create_task(_cleanup_later())


def _inject_forward_context(req: ProviderRequest, context_text: str) -> None:
    user_question = req.prompt.strip() or "请根据以下聊天记录解析内容进行回复"
    quoted_sender = getattr(req, "_quoted_sender", "未知用户")
    context_prompt = (
        f"\n\n用户提供了由 {quoted_sender} 发送/引用的聊天记录。"
        "以下是系统完成多组件解析后的完整结构化内容，请基于此回复：\n"
        f"--- 聊天记录解析开始 ---\n{context_text}\n--- 聊天记录解析结束 ---"
    )
    req.prompt = user_question + context_prompt


def _extract_core_quoted_image_caption(req: ProviderRequest, max_len: int = 280) -> str:
    """提取 Core 的引用图片转述文本，用于转发注入补充说明。"""
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        return ""

    marker = "[Image Caption in quoted message]:"
    xml_tag_pattern = re.compile(r"<image_caption>([\s\S]*?)</image_caption>", re.IGNORECASE)
    captions: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if not isinstance(text, str):
            continue

        if marker in text:
            for line in text.splitlines():
                if marker in line:
                    cap = line.split(marker, 1)[1].strip()
                    if cap:
                        captions.append(cap)
            if marker in text:
                cap = text.split(marker, 1)[1].strip()
                if cap:
                    captions.append(cap)

        for m in xml_tag_pattern.findall(text):
            cap = str(m or "").strip()
            if cap:
                captions.append(cap)

    if not captions:
        return ""

    merged = "；".join(list(dict.fromkeys(captions)))
    merged = re.sub(r"\s+", " ", merged).strip()
    if len(merged) > max_len:
        merged = merged[:max_len].rstrip() + "……"
    return merged


def _build_caption_sig(text: str) -> str:
    if not text:
        return "none"
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


async def extract_content_recursively(
    message_nodes: List[Dict[str, Any]],
    extracted_texts: list[str],
    image_urls: list[str],
    video_sources: list[str],
    enable_json_parse: bool = True,
    depth: int = 0,
    max_depth: int = MAX_RECURSION_DEPTH,
) -> None:
    if depth > max_depth:
        logger.warning(f"forward_parser: 达到最大递归深度 ({max_depth})，停止解析。")
        extracted_texts.append("  " * depth + "[已达到最大转发嵌套深度，后续内容省略]")
        return

    indent = "  " * depth
    for message_node in message_nodes:
        sender_name = message_node.get("sender", {}).get("nickname", "未知用户")
        raw_content = message_node.get("message") or message_node.get("content", [])

        content_chain: list[Any] = []
        if isinstance(raw_content, str):
            try:
                parsed_content = json.loads(raw_content)
                if isinstance(parsed_content, list):
                    content_chain = parsed_content
            except (json.JSONDecodeError, TypeError):
                content_chain = [{"type": "text", "data": {"text": raw_content}}]
        elif isinstance(raw_content, list):
            content_chain = raw_content

        node_text_parts: list[str] = []
        has_only_forward = False
        if isinstance(content_chain, list):
            if len(content_chain) == 1 and isinstance(content_chain[0], dict) and content_chain[0].get("type") == "forward":
                has_only_forward = True

            for segment in content_chain:
                if not isinstance(segment, dict):
                    continue
                seg_type = segment.get("type")
                seg_data = segment.get("data", {}) or {}

                if seg_type == "text":
                    text = seg_data.get("text", "")
                    if text:
                        node_text_parts.append(text)
                elif seg_type == "image":
                    url = seg_data.get("url")
                    if url:
                        image_urls.append(str(url))
                        node_text_parts.append(f"[图片(来自:{sender_name})]")
                elif seg_type == "video":
                    url = seg_data.get("url")
                    file_val = seg_data.get("file")
                    if url:
                        video_sources.append(str(url))
                        node_text_parts.append(f"[视频(来自:{sender_name})]")
                    elif file_val:
                        video_sources.append(str(file_val))
                        node_text_parts.append(f"[视频(来自:{sender_name})]")
                elif seg_type == "file":
                    file_name = seg_data.get("name") or seg_data.get("file_name") or seg_data.get("file") or "未知文件"
                    node_text_parts.append(f"[文件(来自:{sender_name}): {file_name}]")
                elif seg_type == "json":
                    if not enable_json_parse:
                        node_text_parts.append(f"[分享卡片(来自:{sender_name})]")
                    else:
                        raw_json = seg_data.get("data")
                        if raw_json:
                            raw_json_str = json.dumps(raw_json, ensure_ascii=False) if isinstance(raw_json, dict) else str(raw_json)
                            news_texts, key_info = parse_json_segment_data(raw_json_str)
                            if key_info:
                                node_text_parts.append(f"[分享卡片(来自:{sender_name}): {key_info}]")
                            elif news_texts:
                                node_text_parts.append(f"[分享卡片摘要(来自:{sender_name}): {'；'.join(news_texts[:3])}]")
                            else:
                                node_text_parts.append(f"[分享卡片(来自:{sender_name}): 未提取到关键信息]")
                        else:
                            node_text_parts.append(f"[分享卡片(来自:{sender_name}): 未提取到关键信息]")
                elif seg_type == "forward":
                    nested_content = seg_data.get("content")
                    if isinstance(nested_content, list):
                        await extract_content_recursively(
                            nested_content,
                            extracted_texts,
                            image_urls,
                            video_sources,
                            enable_json_parse=enable_json_parse,
                            depth=depth + 1,
                            max_depth=max_depth,
                        )
                    else:
                        node_text_parts.append("[转发消息内容缺失或格式错误]")

        full_node_text = "".join(node_text_parts).strip()
        if full_node_text and not has_only_forward:
            extracted_texts.append(f"{indent}{sender_name}: {full_node_text}")


async def extract_forward_content(
    client: Any,
    forward_id: str,
    max_message_count: int,
    nested_parse_depth: int,
    enable_json_parse: bool = True,
) -> Tuple[list[str], list[str], list[str]]:
    extracted_texts: list[str] = []
    image_urls: list[str] = []
    video_sources: list[str] = []
    try:
        forward_data = await client.api.call_action("get_forward_msg", id=forward_id)
    except Exception as e:
        logger.warning(f"调用 get_forward_msg API 失败 (ID: {forward_id}): {e}")
        return [], [], []

    if not forward_data or "messages" not in forward_data:
        return [], [], []

    messages = forward_data["messages"]
    if max_message_count > 0 and len(messages) > max_message_count:
        logger.info(f"[Forward解析] 消息数量超出上限，已截断: before={len(messages)}, limit={max_message_count}")
        messages = messages[:max_message_count]

    await extract_content_recursively(
        messages,
        extracted_texts,
        image_urls,
        video_sources,
        enable_json_parse=enable_json_parse,
        depth=0,
        max_depth=max(1, nested_parse_depth),
    )
    return extracted_texts, image_urls, video_sources


async def _describe_forward_images(
    image_urls: list[str],
    vision_provider: Any,
    max_forward_images: int,
) -> list[str]:
    if not image_urls:
        return []

    deduped: list[str] = []
    for img in image_urls:
        s = str(img or "").strip()
        if s and s not in deduped:
            deduped.append(s)

    if max_forward_images <= 0:
        return []

    selected = deduped[:max_forward_images]
    if not vision_provider:
        return ["无可用视觉模型，无法提取细节。"] * len(selected)

    lines: list[str] = []
    for idx, url in enumerate(selected, 1):
        try:
            response = await vision_provider.text_chat(
                prompt="请用一句中文描述这张图片的关键内容，不超过30字。若无法判断，回复“未识别”。",
                system_prompt="你是图片内容理解助手，请只输出描述结果，不要额外解释。",
                image_urls=[url],
                context=[],
            )
            desc = _extract_completion_text(response) or "未识别"
        except Exception as e:
            logger.debug(f"[Forward解析] 图片转述失败: idx={idx}, err={e}")
            desc = "未识别"
        lines.append(desc)
    return lines


async def _transcribe_forward_video_audio(
    event: AstrMessageEvent,
    local_videos: list[str],
    get_stt_provider: Callable[[AstrMessageEvent], Any],
    get_cfg: Callable[[str, Any], Any],
) -> list[str]:
    if not local_videos:
        return []

    stt = get_stt_provider(event)
    if not stt:
        return []

    lines: list[str] = []
    for idx, lv_path in enumerate(local_videos):
        try:
            wav_path = await extract_audio_wav(get_cfg("ffmpeg_path", ""), lv_path)
            if not wav_path or not os.path.exists(wav_path):
                continue
            try:
                asr_text = None
                if hasattr(stt, "get_text"):
                    asr_text = await stt.get_text(wav_path)
                elif hasattr(stt, "speech_to_text"):
                    asr_res = await stt.speech_to_text(wav_path)
                    asr_text = asr_res.get("text", "") if isinstance(asr_res, dict) else str(asr_res)
                asr_text = str(asr_text or "").strip()
                if asr_text:
                    lines.append(f"[视频语音转写 {idx + 1}] {asr_text}")
            except Exception as e:
                logger.debug(f"[Forward解析] 转发视频 ASR 失败: idx={idx + 1}, err={e}")
            finally:
                try:
                    os.remove(wav_path)
                except Exception as e:
                    logger.debug(f"[Forward解析] 转发视频 WAV 清理失败: path={wav_path}, err={e}")
        except Exception as e:
            logger.debug(f"[Forward解析] 转发视频音频提取失败: idx={idx + 1}, path={lv_path}, err={e}")
    return lines


async def _build_forward_video_lines(
    frame_paths: list[str],
    asr_lines: list[str],
    vision_provider: Any,
) -> list[str]:
    if not frame_paths and not asr_lines:
        return []

    lines: list[str] = []
    if frame_paths:
        if not vision_provider:
            lines.append("[视频转述] 检测到视频关键帧，但当前无可用视觉模型。")
        else:
            for idx, frame in enumerate(frame_paths, 1):
                try:
                    response = await vision_provider.text_chat(
                        prompt=f"请描述这帧画面内容（第{idx}帧），不超过25字。",
                        system_prompt="你是视频帧分析助手，请直接输出结果。",
                        image_urls=[frame],
                        context=[],
                    )
                    desc = _extract_completion_text(response) or "未识别"
                except Exception as e:
                    logger.debug(f"[Forward解析] 转发视频关键帧识别失败: idx={idx}, err={e}")
                    desc = "未识别"
                lines.append(f"[视频关键帧 {idx}] {desc}")

    lines.extend(asr_lines)
    return lines


_IMAGE_PH_RE = re.compile(r"\[图片\(来自:[^\]]+\)\]")
_VIDEO_PH_RE = re.compile(r"\[视频\(来自:[^\]]+\)\]")


def _embed_media_into_records(
    base_records: list[str],
    image_lines: list[str],
    video_lines: list[str],
) -> list[str]:
    records: list[str] = []
    image_idx = 0
    has_video_placeholder = False
    video_inline = "；".join([str(x).strip() for x in video_lines if str(x).strip()][:4]).strip()

    for line in base_records:
        curr_line = str(line or "")

        def _img_repl(_: re.Match[str]) -> str:
            nonlocal image_idx
            if image_idx < len(image_lines):
                desc = str(image_lines[image_idx] or "").strip() or "未识别"
                image_idx += 1
                return f"[图片解析:{desc}]"
            image_idx += 1
            return "[图片解析:未解析]"

        def _video_repl(_: re.Match[str]) -> str:
            nonlocal has_video_placeholder
            has_video_placeholder = True
            return f"[视频解析:{video_inline or '未解析'}]"

        if image_lines:
            curr_line = _IMAGE_PH_RE.sub(_img_repl, curr_line)
        curr_line = _VIDEO_PH_RE.sub(_video_repl, curr_line)
        records.append(curr_line)

    if image_idx < len(image_lines):
        for extra in image_lines[image_idx:]:
            extra_text = str(extra or "").strip()
            if extra_text:
                records.append(f"[图片解析补充] {extra_text}")

    if video_inline and not has_video_placeholder:
        records.append(f"[视频解析补充] {video_inline}")

    return records


def _build_forward_context_text(
    base_records: list[str],
    image_lines: list[str],
    video_lines: list[str],
    core_quote_image_caption: str = "",
) -> str:
    sections: list[str] = []
    merged_records = _embed_media_into_records(base_records, image_lines, video_lines)
    if merged_records:
        sections.append("原始聊天记录:\n" + "\n".join(merged_records))
    if core_quote_image_caption:
        sections.append(f"引用图片补充说明（来自框架转述）:\n{core_quote_image_caption}")
    return "\n\n".join(sections).strip()


async def process_forward_record_content(
    event: AstrMessageEvent,
    req: ProviderRequest,
    forward_id: Optional[str],
    get_cfg: Callable[[str, Any], Any],
    get_stt_provider: Callable[[AstrMessageEvent], Any],
    get_vision_provider: Callable[[AstrMessageEvent], Any],
    cleanup_paths: Callable[[list[str]], Awaitable[None]],
) -> bool:
    """处理合并转发聊天记录，并注入完整结构化解析内容。"""
    if not forward_id:
        return False
    if not bool(get_cfg("forward_parse_enable", True)):
        return False
    if not (IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent)):
        return False

    try:
        max_forward_messages = int(get_cfg("forward_message_max_count", 50) or 50)
        nested_parse_depth = int(get_cfg("nested_parse_depth", MAX_RECURSION_DEPTH) or MAX_RECURSION_DEPTH)
        json_parse_enable = bool(get_cfg("json_parse_enable", True))
        max_forward_video_count = int(get_cfg("forward_video_max_count", 2) or 2)
        max_forward_video_frame_count = int(get_cfg("forward_video_max_frame_count", 1) or 1)
        frame_interval_sec = int(get_cfg("video_frame_interval_sec", 12) or 12)
        max_mb = int(get_cfg("video_max_size_mb", 50) or 50)
        asr_enabled = bool(get_cfg("video_asr_enable", True))
        max_forward_images = int(get_cfg("forward_image_max_count", 8) or 8)
        core_quote_image_caption = _extract_core_quoted_image_caption(req)
        core_quote_image_caption_exists = bool(core_quote_image_caption)
        core_quote_image_caption_sig = _build_caption_sig(core_quote_image_caption)
        logger.debug(
            f"[Forward解析] Core图片转述检测: exists={core_quote_image_caption_exists}, "
            f"caption_len={len(core_quote_image_caption)}, sig={core_quote_image_caption_sig}"
        )
        result_cache_key = _build_forward_result_cache_key(
            forward_id=forward_id,
            max_message_count=max_forward_messages,
            nested_parse_depth=nested_parse_depth,
            json_parse_enable=json_parse_enable,
            max_forward_video_count=max_forward_video_count,
            max_forward_video_frame_count=max_forward_video_frame_count,
            frame_interval_sec=frame_interval_sec,
            max_mb=max_mb,
            asr_enabled=asr_enabled,
            max_forward_images=max_forward_images,
            core_quote_image_caption_sig=core_quote_image_caption_sig,
        )

        cached_context = await _get_cached_forward_context(result_cache_key)
        if cached_context:
            _inject_forward_context(req, cached_context)
            logger.debug(
                f"[Forward解析] 最终上下文缓存命中: forward_id={forward_id}, context_len={len(cached_context)}"
            )
            await cleanup_paths(getattr(req, "_cleanup_paths", []))
            return True

        extracted_texts, image_urls, forward_video_sources = await extract_forward_content(
            event.bot,
            forward_id,
            max_message_count=max_forward_messages,
            nested_parse_depth=nested_parse_depth,
            enable_json_parse=json_parse_enable,
        )
        if not (extracted_texts or image_urls or forward_video_sources):
            return False

        vision_provider = get_vision_provider(event)
        if core_quote_image_caption_exists and image_urls:
            logger.debug("[Forward解析] 检测到 Core 已完成引用图片转述，跳过转发图片二次转述。")
            image_lines = []
        else:
            image_lines = await _describe_forward_images(image_urls, vision_provider, max_forward_images)

        video_lines: list[str] = []
        if forward_video_sources and max_forward_video_count > 0 and max_forward_video_frame_count > 0:
            f_frames, f_cleanup, f_local_videos = await extract_forward_video_keyframes(
                event,
                forward_video_sources,
                max_count=max_forward_video_count,
                max_frame_count=max_forward_video_frame_count,
                frame_interval_sec=frame_interval_sec,
                ffmpeg_path=get_cfg("ffmpeg_path", ""),
                max_mb=max_mb,
                max_duration=7200,
                timeout_sec=10,
            )
            if f_cleanup:
                await _schedule_delayed_cleanup(f_cleanup, delay_sec=60)

            asr_lines: list[str] = []
            if asr_enabled and f_local_videos:
                asr_lines = await _transcribe_forward_video_audio(
                    event=event,
                    local_videos=f_local_videos,
                    get_stt_provider=get_stt_provider,
                    get_cfg=get_cfg,
                )
            video_lines = await _build_forward_video_lines(
                frame_paths=f_frames,
                asr_lines=asr_lines,
                vision_provider=vision_provider,
            )
        elif forward_video_sources:
            video_lines = ["[视频转述] 聊天记录包含视频，但已在配置中关闭视频解析。"]

        context_text = _build_forward_context_text(
            base_records=extracted_texts,
            image_lines=image_lines,
            video_lines=video_lines,
            core_quote_image_caption=core_quote_image_caption,
        )
        if not context_text:
            return False

        await _set_cached_forward_context(result_cache_key, context_text)
        _inject_forward_context(req, context_text)
        logger.info(
            f"[转发消息] 成功注入完整解析内容: text={len(extracted_texts)}, image={len(image_lines)}, "
            f"video={len(video_lines)}, context_len={len(context_text)}"
        )

        await cleanup_paths(getattr(req, "_cleanup_paths", []))
        return True
    except Exception as e:
        logger.warning(f"转发内容提取失败: {e}")
        return False
