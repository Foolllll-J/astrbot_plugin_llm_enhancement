import asyncio
from pathlib import Path
from typing import Any, Callable, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

from .video_parser import (
    extract_videos_from_chain,
    probe_duration_sec,
    download_video_to_temp,
    MediaScenario,
    MediaContext,
    VideoFrameProcessor,
    is_gif_file,
)


async def _probe_duration_helper(get_cfg: Callable[[str, Any], Any], media_path: str) -> float:
    try:
        duration = await asyncio.to_thread(
            probe_duration_sec,
            get_cfg("ffmpeg_path", ""),
            media_path,
        ) or 0
        return duration
    except Exception as e:
        logger.warning(f"[探测时长] 异常:  {e}")
        return 0


async def detect_media_scenario(
    req: ProviderRequest,
    get_cfg: Callable[[str, Any], Any],
    video_sources: Optional[list[str]] = None,
) -> MediaContext:
    """检测当前媒体场景并返回上下文。"""
    ctx = MediaContext()
    image_urls: Any = getattr(req, "image_urls", None)

    if video_sources and len(video_sources) > 0:
        ctx.media_path = video_sources[0]
    elif image_urls and isinstance(image_urls, list) and len(image_urls) > 0:
        ctx.media_path = image_urls[0]
    else:
        ctx.scenario = MediaScenario.NONE
        return ctx

    first_path = ctx.media_path
    if not first_path or not isinstance(first_path, str):
        ctx.scenario = MediaScenario.NONE
        logger.warning("[场景判断] → 无效媒体（非字符串）")
        return ctx

    if first_path.startswith(("http://", "https://")):
        logger.debug(f"[场景判断] 检测到远程 URL，尝试下载: {first_path}")
        try:
            max_size = get_cfg("video_max_size_mb", 50)
            local_path = await download_video_to_temp(first_path, max_size)
            if local_path:
                ctx.media_path = local_path
                ctx.cleanup_paths.append(local_path)
                first_path = local_path
                logger.info(f"[场景判断] 下载成功: {local_path}")
            else:
                ctx.scenario = MediaScenario.NONE
                return ctx
        except Exception as e:
            logger.warning(f"[场景判断] 下载过程抛出异常: {e}")
            ctx.scenario = MediaScenario.NONE
            return ctx

    ctx.media_path = first_path

    if is_gif_file(first_path):
        ctx.duration = await _probe_duration_helper(get_cfg, first_path)
        if ctx.duration <= 0:
            ctx.scenario = MediaScenario.NONE
            logger.info("[场景判断] → GIF 探测时长为 0s，视为静态图片，跳过增强处理")
            return ctx
        ctx.scenario = MediaScenario.GIF_ANIMATED
        logger.info(f"[场景判断] → GIF 动图 (时长: {ctx.duration:.2f}s)")
        return ctx

    suffix = Path(first_path).suffix.lower()
    is_from_video_source = bool(video_sources and len(video_sources) > 0 and ctx.media_path == video_sources[0])
    is_video_format = suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"] or is_from_video_source
    if not is_video_format:
        try:
            with open(first_path, "rb") as f:
                header = f.read(32)
                if b"ftyp" in header or b"matroska" in header or b"fLaC" in header:
                    is_video_format = True
        except Exception as e:
            logger.debug(f"[场景判断] 媒体头部探测失败: path={first_path}, err={e}")

    if is_video_format:
        ctx.duration = await _probe_duration_helper(get_cfg, first_path)
        if ctx.duration <= 0:
            ctx.scenario = MediaScenario.NONE
            logger.info("[场景判断] → 探测时长为 0s，判定为普通图片，跳过增强处理")
            return ctx
        ctx.scenario = MediaScenario.VIDEO
        logger.info(f"[场景判断] → 视频 (统一抽帧流程, 时长: {ctx.duration:.2f}s)")
        return ctx

    ctx.scenario = MediaScenario.NONE
    logger.info(f"[场景判断] → 静态图片或未知格式 (后缀: {suffix})，跳过增强处理")
    return ctx


async def process_media_content(
    context: Any,
    event: AstrMessageEvent,
    req: ProviderRequest,
    all_components: list[Any],
    reply_seg: Optional[Comp.Reply],
    get_cfg: Callable[[str, Any], Any],
) -> None:
    """处理视频/GIF/文件相关媒体分支。"""
    if not get_cfg("video_detect_enable", True):
        return

    video_sources = extract_videos_from_chain(all_components)

    if not video_sources and reply_seg:
        try:
            client = event.bot
            original_msg = await client.api.call_action("get_msg", message_id=reply_seg.id)
            if original_msg and "message" in original_msg:
                video_sources = extract_videos_from_chain(original_msg["message"])
        except Exception as e:
            logger.warning(f"从引用消息提取视频失败: {e}")

    media_ctx = await detect_media_scenario(req, get_cfg, video_sources)
    req._cleanup_paths.extend(media_ctx.cleanup_paths)

    logger.info(f"[LLMEnhancement] 媒体场景: {media_ctx.scenario.value}")

    processor = VideoFrameProcessor(context, event, get_cfg)

    if media_ctx.scenario in [MediaScenario.VIDEO, MediaScenario.GIF_ANIMATED]:
        if media_ctx.media_path in req.image_urls:
            req.image_urls.remove(media_ctx.media_path)
        if len(req.image_urls) > 0:
            req.image_urls = [
                url for url in req.image_urls
                if not any(url.lower().endswith(s) for s in [".mp4", ".mov", ".avi", ".wmv", ".flv", ".m4v"])
            ]

    if media_ctx.scenario == MediaScenario.VIDEO:
        logger.debug("[分支] 执行视频处理")
        quoted_sender = getattr(req, "_quoted_sender", None) if reply_seg else None
        current_msg_id = getattr(getattr(event, "message_obj", None), "message_id", None)
        msg_id = str(reply_seg.id) if reply_seg else str(current_msg_id)
        logger.debug(f"[LLMEnhancement] 视频处理最终 msg_id: {msg_id}")
        success = await processor.process_long_video(
            req,
            media_ctx.media_path,
            media_ctx.duration,
            sender_name=quoted_sender,
            msg_id=msg_id,
        )
        if not success:
            logger.warning("[分支] 视频处理失败")

    elif media_ctx.scenario == MediaScenario.GIF_ANIMATED:
        logger.debug("[分支] 执行 GIF 处理")
        quoted_sender = getattr(req, "_quoted_sender", None) if reply_seg else None
        current_msg_id = getattr(getattr(event, "message_obj", None), "message_id", None)
        msg_id = str(reply_seg.id) if reply_seg else str(current_msg_id)
        logger.debug(f"[LLMEnhancement] GIF 处理最终 msg_id: {msg_id}")
        success = await processor.process_gif(
            req,
            media_ctx.media_path,
            sender_name=quoted_sender,
            msg_id=msg_id,
        )
        if not success:
            logger.warning("[分支] GIF 处理失败")

    elif media_ctx.scenario == MediaScenario.NONE:
        logger.debug("[分支] 无媒体内容")
