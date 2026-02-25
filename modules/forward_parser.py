import asyncio
import json
import os
import random
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable, Awaitable

import astrbot.api.message_components as Comp
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger
from .video_parser import extract_forward_video_keyframes, extract_audio_wav

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False

MAX_RECURSION_DEPTH = 5  # 最大递归深度，防止恶意嵌套


@dataclass
class ReferenceProcessResult:
    reply_seg: Optional[Comp.Reply] = None
    handled_forward: bool = False
    blocked: bool = False


@dataclass
class ReplyParseResult:
    forward_id: Optional[str]
    json_extracted_texts: list[str] = field(default_factory=list)
    blocked: bool = False


async def parse_reply_context(
    event: AstrMessageEvent,
    req: ProviderRequest,
    reply_seg: Comp.Reply,
    forward_id: Optional[str],
    get_cfg: Callable[[str, Any], Any],
    download_media: Callable[[str, float], Any],
) -> ReplyParseResult:
    """解析引用消息内容，提取转发/媒体/文本上下文并注入到请求。"""
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

        if original_sender == self_id:
            has_video = any(s.get("type") == "video" for s in original_msg["message"])
            has_file = any(s.get("type") == "file" for s in original_msg["message"])
            has_forward = any(s.get("type") == "forward" for s in original_msg["message"])

            video_prob = float(get_cfg("quote_self_video_block_prob", 0) or 0)
            file_prob = float(get_cfg("quote_self_file_block_prob", 0) or 0)
            forward_prob = float(get_cfg("quote_self_forward_block_prob", 0) or 0)

            if enable_video_parse and has_video and video_prob > 0 and random.random() < video_prob:
                logger.info(f"[LLMEnhancement] 检测到引用了 Bot 自身视频，命中屏蔽概率 ({video_prob})，已拦截请求。")
                result.blocked = True
                return result
            if enable_file_parse and has_file and file_prob > 0 and random.random() < file_prob:
                logger.info(f"[LLMEnhancement] 检测到引用了 Bot 自身文件，命中屏蔽概率 ({file_prob})，已拦截请求。")
                result.blocked = True
                return result
            if enable_forward_parse and has_forward and forward_prob > 0 and random.random() < forward_prob:
                logger.info(f"[LLMEnhancement] 检测到引用了 Bot 自身 Forward，命中屏蔽概率 ({forward_prob})，已拦截请求。")
                result.blocked = True
                return result

        setattr(req, "_quoted_sender", original_sender_name)

        original_message_chain = original_msg["message"]
        if not isinstance(original_message_chain, list):
            return result

        has_extracted_media = False
        quoted_file_names: list[str] = []
        max_size = get_cfg("video_max_size_mb", 50)
        if not hasattr(req, "image_urls") or req.image_urls is None:
            req.image_urls = []
        if not hasattr(req, "_cleanup_paths"):
            req._cleanup_paths = []

        allow_file_name_inject = bool(get_cfg("inject_file_name", True))
        for segment in original_message_chain:
            seg_type = segment.get("type")

            if enable_forward_parse and seg_type == "forward":
                new_forward_id = segment.get("data", {}).get("id")
                if new_forward_id:
                    result.forward_id = new_forward_id
                    break

            elif seg_type == "image":
                url = segment.get("data", {}).get("url")
                if not url:
                    continue
                has_extracted_media = True

                if url.startswith(("http://", "https://")):
                    try:
                        local_path = await download_media(url, max_size)
                        if local_path and local_path not in req.image_urls:
                            req.image_urls.append(local_path)
                            logger.debug(
                                f"[LLMEnhancement] 从引用消息中提取并下载媒体: {url[:50]}... -> {local_path}"
                            )
                            req._cleanup_paths.append(local_path)
                        if local_path:
                            continue
                    except Exception as e:
                        logger.warning(f"下载引用消息图片失败: {e}")

                if url not in req.image_urls:
                    req.image_urls.append(url)
                    logger.debug(f"[LLMEnhancement] 从引用消息中提取到媒体 URL: {url}")

            elif enable_file_parse and allow_file_name_inject and seg_type == "file":
                file_name = segment.get("data", {}).get("file")
                if file_name:
                    quoted_file_names.append(file_name)
                    logger.debug(f"[LLMEnhancement] 从引用消息中提取到文件名: {file_name}")

            elif seg_type == "json":
                # 兼容保留：QQ 合并转发卡片(JSON)中的摘要文本提取
                try:
                    inner_data_str = segment.get("data", {}).get("data")
                    if not inner_data_str:
                        continue
                    inner_data_str = inner_data_str.replace("&#44;", ",")
                    inner_json = json.loads(inner_data_str)
                    if (
                        inner_json.get("app") == "com.tencent.multimsg"
                        and inner_json.get("config", {}).get("forward") == 1
                    ):
                        news_items = inner_json.get("meta", {}).get("detail", {}).get("news", [])
                        for item in news_items:
                            text_content = item.get("text")
                            if text_content:
                                clean_text = text_content.strip().replace("[图片]", "").strip()
                                if clean_text:
                                    result.json_extracted_texts.append(clean_text)
                        if result.json_extracted_texts:
                            break
                except Exception as e:
                    logger.debug(f"[LLMEnhancement] 解析转发 JSON 失败，已跳过该片段: err={e}")
                    continue

        if has_extracted_media or quoted_file_names:
            quoted_sender = getattr(req, "_quoted_sender", "未知用户")
            context_parts = []
            if has_extracted_media:
                context_parts.append(f"以上图片提取自用户 {quoted_sender} 的引用消息")
            if quoted_file_names:
                file_list_str = "、".join(quoted_file_names)
                context_parts.append(f"用户 {quoted_sender} 的引用消息中包含文件: {file_list_str}")

            context_desc = f"\n\n[补充信息] {'；'.join(context_parts)}，请在回复时参考该上下文。"
            req.prompt += context_desc
            logger.debug(f"[LLMEnhancement] 已为引用内容注入发送者/文件信息: {quoted_sender}")

        has_video_in_quote = any(s.get("type") == "video" for s in original_message_chain)
        if has_video_in_quote:
            quoted_sender = getattr(req, "_quoted_sender", "未知用户")
            logger.debug(f"[LLMEnhancement] 检测到引用视频，已记录发送者信息: {quoted_sender}")
    except Exception as e:
        logger.warning(f"获取被回复消息详情失败: {e}")

    return result


async def extract_content_recursively(
    message_nodes: List[Dict[str, Any]],
    extracted_texts: list[str],
    image_urls: list[str],
    video_sources: list[str],
    depth: int = 0,
    max_depth: int = MAX_RECURSION_DEPTH,
):
    """
    核心递归解析器。遍历消息节点列表，提取文本、图片、视频，并处理嵌套的 forward 结构。
    """
    if depth > max_depth:
        logger.warning(f"forward_parser: 达到最大递归深度 ({max_depth})，停止解析。")
        extracted_texts.append("  " * depth + "[已达到最大转发嵌套深度，后续内容略]")
        return

    indent = "  " * depth
    for message_node in message_nodes: 
        sender_name = message_node.get("sender", {}).get("nickname", "未知用户")
        raw_content = message_node.get("message") or message_node.get("content", [])
        
        logger.debug(f"forward_parser: 正在解析消息节点 (深度: {depth}, 发送者: {sender_name})")
        
        content_chain = [] 
        if isinstance(raw_content, str): 
            try: 
                parsed_content = json.loads(raw_content) 
                if isinstance(parsed_content, list): 
                    content_chain = parsed_content 
            except (json.JSONDecodeError, TypeError): 
                content_chain = [{"type": "text", "data": {"text": raw_content}}] 
        elif isinstance(raw_content, list): 
            content_chain = raw_content 

        node_text_parts = [] 
        has_only_forward = False
        
        if isinstance(content_chain, list):
            if len(content_chain) == 1 and content_chain[0].get("type") == "forward":
                has_only_forward = True
                
            for segment in content_chain: 
                if isinstance(segment, dict): 
                    seg_type = segment.get("type") 
                    seg_data = segment.get("data", {}) 
                    
                    if seg_type == "text": 
                        text = seg_data.get("text", "") 
                        if text: node_text_parts.append(text) 
                    elif seg_type == "image": 
                        url = seg_data.get("url") 
                        if url: 
                            image_urls.append(url) 
                            node_text_parts.append(f"[图片(来自:{sender_name})]") 
                    elif seg_type == "video":
                        url = seg_data.get("url")
                        file = seg_data.get("file")
                        if url:
                            video_sources.append(url)
                            node_text_parts.append(f"[视频(来自:{sender_name})]")
                        elif file:
                            video_sources.append(file)
                            node_text_parts.append(f"[视频(来自:{sender_name})]")
                    elif seg_type == "forward":
                        nested_content = seg_data.get("content")
                        if isinstance(nested_content, list):
                            logger.debug(f"forward_parser: 发现嵌套转发内容 (深度: {depth + 1}, 节点数: {len(nested_content)})")
                            await extract_content_recursively(
                                nested_content,
                                extracted_texts,
                                image_urls,
                                video_sources,
                                depth + 1,
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
) -> Tuple[list[str], list[str], list[str]]:
    """
    从合并转发消息中提取内容（文本、图片、视频源）。
    """
    extracted_texts = [] 
    image_urls = []
    video_sources = []
    try: 
        forward_data = await client.api.call_action('get_forward_msg', id=forward_id) 
    except Exception as e: 
        logger.warning(f"调用 get_forward_msg API 失败 (ID: {forward_id}): {e}") 
        return [], [], [] 

    if not forward_data or "messages" not in forward_data: 
        return [], [], [] 

    messages = forward_data["messages"]
    if max_message_count > 0 and len(messages) > max_message_count:
        logger.info(
            f"[Forward解析] 消息数量超出上限，已截断: before={len(messages)}, limit={max_message_count}"
        )
        messages = messages[:max_message_count]

    await extract_content_recursively(
        messages,
        extracted_texts,
        image_urls,
        video_sources,
        depth=0,
        max_depth=max(1, nested_parse_depth),
    )
    return extracted_texts, image_urls, video_sources


async def process_reference_and_forward_content(
    event: AstrMessageEvent,
    req: ProviderRequest,
    all_components: list[Any],
    get_cfg: Callable[[str, Any], Any],
    get_stt_provider: Callable[[AstrMessageEvent], Any],
    cleanup_paths: Callable[[list[str]], Awaitable[None]],
    download_media: Callable[[str, float], Any],
) -> ReferenceProcessResult:
    """处理引用消息与转发消息内容提取。若转发内容已完整注入，会返回 handled_forward=True。"""
    result = ReferenceProcessResult()
    forward_parse_enabled = bool(get_cfg("forward_parse_enable", True))
    forward_id: Optional[str] = None
    json_extracted_texts: list[str] = []

    if not (IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent)):
        return result

    for seg in all_components:
        if isinstance(seg, Comp.Forward):
            forward_id = seg.id
            break
        if isinstance(seg, Comp.Reply):
            result.reply_seg = seg

    if event.is_at_or_wake_command and (not forward_id) and result.reply_seg:
        parsed_reply = await parse_reply_context(
            event=event,
            req=req,
            reply_seg=result.reply_seg,
            forward_id=forward_id,
            get_cfg=get_cfg,
            download_media=download_media,
        )
        if parsed_reply.blocked:
            result.blocked = True
            return result
        forward_id = parsed_reply.forward_id
        json_extracted_texts = parsed_reply.json_extracted_texts

    if not forward_parse_enabled:
        return result

    if not (forward_id or json_extracted_texts):
        return result

    image_urls: list[str] = []
    forward_video_sources: list[str] = []
    try:
        if forward_id:
            max_forward_messages = int(get_cfg("forward_message_max_count", 50) or 50)
            nested_parse_depth = int(get_cfg("nested_parse_depth", MAX_RECURSION_DEPTH) or MAX_RECURSION_DEPTH)
            extracted_texts, image_urls, forward_video_sources = await extract_forward_content(
                event.bot,
                forward_id,
                max_message_count=max_forward_messages,
                nested_parse_depth=nested_parse_depth,
            )
        else:
            extracted_texts = json_extracted_texts

        if not (extracted_texts or image_urls or forward_video_sources):
            return result

        max_forward_video_count = int(get_cfg("forward_video_max_count", 2) or 2)
        max_forward_video_frame_count = int(get_cfg("forward_video_max_frame_count", 1) or 1)
        if forward_video_sources and max_forward_video_count > 0 and max_forward_video_frame_count > 0:
            f_frames, f_cleanup, f_local_videos = await extract_forward_video_keyframes(
                event,
                forward_video_sources,
                max_count=max_forward_video_count,
                max_frame_count=max_forward_video_frame_count,
                frame_interval_sec=int(get_cfg("video_frame_interval_sec", 12) or 12),
                ffmpeg_path=get_cfg("ffmpeg_path", ""),
                max_mb=get_cfg("video_max_size_mb", 50),
                max_duration=7200,
                timeout_sec=10,
            )
            if f_frames:
                image_urls.extend(f_frames)

                if get_cfg("video_asr_enable", True) and f_local_videos:
                    for idx, lv_path in enumerate(f_local_videos):
                        try:
                            wav_path = await extract_audio_wav(get_cfg("ffmpeg_path", ""), lv_path)
                            if wav_path and os.path.exists(wav_path):
                                stt = get_stt_provider(event)
                                if stt:
                                    asr_text = None
                                    try:
                                        if hasattr(stt, "get_text"):
                                            asr_text = await stt.get_text(wav_path)
                                        elif hasattr(stt, "speech_to_text"):
                                            asr_res = await stt.speech_to_text(wav_path)
                                            asr_text = asr_res.get("text", "") if isinstance(asr_res, dict) else str(asr_res)

                                        if asr_text:
                                            extracted_texts.append(f" [视频语音转写 {idx + 1}] {asr_text}")
                                            logger.debug(f"转发视频 {idx + 1} ASR 成功")
                                    except Exception as e:
                                        logger.debug(f"[LLMEnhancement] 转发视频 ASR 失败: idx={idx + 1}, err={e}")

                                try:
                                    os.remove(wav_path)
                                except Exception as e:
                                    logger.debug(f"[LLMEnhancement] 转发视频 WAV 清理失败: path={wav_path}, err={e}")
                        except Exception as e:
                            logger.debug(f"[LLMEnhancement] 转发视频处理失败: idx={idx + 1}, path={lv_path}, err={e}")

                if f_cleanup:
                    async def cleanup_f():
                        await asyncio.sleep(60)
                        for p in f_cleanup:
                            try:
                                if os.path.isdir(p):
                                    shutil.rmtree(p)
                                elif os.path.isfile(p):
                                    os.remove(p)
                            except Exception as e:
                                logger.debug(f"[LLMEnhancement] 转发临时目录清理失败: path={p}, err={e}")

                    asyncio.create_task(cleanup_f())
        elif forward_video_sources:
            logger.info(
                "[聊天记录解析] 已关闭聊天记录视频抽帧，保留视频占位符文本。"
            )

        max_forward_images = int(get_cfg("forward_image_max_count", 8) or 8)
        if max_forward_images > 0 and len(image_urls) > max_forward_images:
            removed = len(image_urls) - max_forward_images
            image_urls = image_urls[:max_forward_images]
            logger.info(
                f"[Forward解析] 图片数量超出上限，已截断: removed={removed}, remain={len(image_urls)}"
            )
        elif max_forward_images <= 0:
            image_urls = []

        chat_records = "\n".join(extracted_texts)
        user_question = req.prompt.strip() or "请总结一下这个聊天记录"
        quoted_sender = getattr(req, "_quoted_sender", "未知用户")

        context_prompt = (
            f"\n\n用户提供了由 {quoted_sender} 发送/转发的聊天记录，请根据这些记录内容来响应。聊天记录如下：\n"
            f"--- 聊天记录开始 ---\n"
            f"{chat_records}\n"
            f"--- 聊天记录结束 ---"
        )
        req.prompt = user_question + context_prompt
        req.image_urls.extend(image_urls)
        logger.info(f"[转发消息] 成功注入转发内容 ({len(extracted_texts)} 条文本, {len(image_urls)} 张图片)")

        await cleanup_paths(req._cleanup_paths)
        result.handled_forward = True
        return result
    except Exception as e:
        logger.warning(f"内容提取失败: {e}")
        return result
