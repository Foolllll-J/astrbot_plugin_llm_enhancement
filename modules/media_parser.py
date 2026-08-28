from __future__ import annotations
import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import math
import json
import aiohttp
from pathlib import Path
from enum import Enum
from typing import List, Optional, Any, Dict, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
import astrbot.api.message_components as Comp
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from .runtime_helpers import (
    _is_unavailable_get_msg_payload,
    append_text_part_to_request,
    find_provider,
    transcribe_audio_with_fallback,
    _provider_supports_audio_input,
)

VIDEO_SUMMARY_CACHE_TTL_SEC = 86400
VIDEO_SUMMARY_CACHE_MAX_SIZE = 100


class MediaScenario(Enum):
    """媒体处理场景"""

    NONE = "none"  # 无媒体
    FORWARD_MESSAGE = "forward_message"  # 转发消息（包含文本/图片/视频）
    VIDEO = "video"  # 视频（统一抽帧流程）
    GIF_DIRECT = "gif_direct"  # GIF 直送 Vision Provider 单次描述
    GIF_ANIMATED = "gif_animated"  # GIF 动图（抽帧模式）


class MediaContext:
    """媒体处理上下文"""

    def __init__(self):
        self.scenario: MediaScenario = MediaScenario.NONE
        self.media_path: Optional[str] = None
        self.duration: float = 0
        self.extracted_texts: List[str] = []
        self.extracted_images: List[str] = []
        self.cleanup_paths: List[str] = []


def ob_data(obj: Any) -> Dict[str, Any]:
    """OneBot 风格响应可能包裹在 data 字段中，展开后返回字典。"""
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, dict):
            return data
        return obj
    return {}


def _safe_subprocess_run(cmd: List[str]) -> subprocess.CompletedProcess:
    """安全执行子进程调用。"""
    if not isinstance(cmd, list) or not cmd:
        raise ValueError("cmd must be a non-empty list")
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        check=False,
        shell=False,
    )


def is_gif_file(path: str) -> bool:
    """
    通过魔数判断是否为 GIF。
    """
    try:
        p = Path(path)
        if not p.exists():
            return False
        with p.open("rb") as f:
            header = f.read(6)
    except OSError:
        return False

    # GIF87a、GIF89a 格式
    if header in (b"GIF87a", b"GIF89a"):
        return True

    if p.suffix.lower() == ".gif":
        return True

    return False


def extract_videos_from_chain(chain: List[object]) -> List[str]:
    """从消息链中递归提取视频相关 URL / 路径。"""
    videos: List[str] = []
    if not isinstance(chain, list):
        return videos

    video_exts = (
        ".mp4",
        ".mov",
        ".m4v",
        ".avi",
        ".webm",
        ".mkv",
        ".flv",
        ".wmv",
        ".ts",
        ".mpeg",
        ".mpg",
        ".3gp",
        ".gif",
    )

    def _looks_like_video(name_or_url: str) -> bool:
        if not isinstance(name_or_url, str) or not name_or_url:
            return False
        s = name_or_url.lower()
        return any(s.endswith(ext) for ext in video_exts)

    for seg in chain:
        try:
            if isinstance(seg, dict):
                # 处理 OneBot v11 原始字典格式。
                stype = seg.get("type")
                sdata = seg.get("data", {})
                if stype == "video":
                    u = sdata.get("url")
                    f = sdata.get("file")
                    if isinstance(u, str) and u:
                        videos.append(u)
                    elif isinstance(f, str) and f:
                        videos.append(f)
                elif stype == "file":
                    u = sdata.get("url")
                    f = sdata.get("file")
                    n = sdata.get("name")
                    cand = None
                    if isinstance(u, str) and u and _looks_like_video(u):
                        cand = u
                    elif (
                        isinstance(f, str)
                        and f
                        and (_looks_like_video(f) or os.path.isabs(f))
                    ):
                        cand = f
                    elif isinstance(n, str) and n and _looks_like_video(n):
                        if isinstance(u, str) and u:
                            cand = u
                        elif isinstance(f, str) and f:
                            cand = f
                    if cand:
                        videos.append(cand)
            elif isinstance(seg, Comp.Video):
                f = getattr(seg, "file", None)
                u = getattr(seg, "url", None)
                if isinstance(u, str) and u:
                    videos.append(u)
                elif isinstance(f, str) and f:
                    videos.append(f)
            elif isinstance(seg, Comp.File):
                u = getattr(seg, "url", None)
                f = getattr(seg, "file", None)
                n = getattr(seg, "name", None)
                cand = None
                if isinstance(u, str) and u and _looks_like_video(u):
                    cand = u
                elif (
                    isinstance(f, str)
                    and f
                    and (_looks_like_video(f) or os.path.isabs(f))
                ):
                    cand = f
                elif isinstance(n, str) and n and _looks_like_video(n):
                    if isinstance(u, str) and u:
                        cand = u
                    elif isinstance(f, str) and f:
                        cand = f
                if isinstance(cand, str) and cand:
                    videos.append(cand)
            elif hasattr(Comp, "Node") and isinstance(seg, getattr(Comp, "Node")):
                content = getattr(seg, "content", None)
                if isinstance(content, list):
                    videos.extend(extract_videos_from_chain(content))
            elif hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            videos.extend(extract_videos_from_chain(c))
            elif hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                # Forward 组件可能包含 nodes。
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            videos.extend(extract_videos_from_chain(c))
        except Exception:
            continue
    return videos


def extract_audios_from_chain(chain: List[object]) -> List[str]:
    """从消息链中提取音频文件 URL / 路径（仅 file 类型段）。"""
    audios: List[str] = []
    if not isinstance(chain, list):
        return audios

    audio_exts = (
        ".amr",
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".flac",
        ".wma",
    )

    def _looks_like_audio(name_or_url: str) -> bool:
        if not isinstance(name_or_url, str) or not name_or_url:
            return False
        s = name_or_url.lower()
        return any(s.endswith(ext) for ext in audio_exts)

    for seg in chain:
        try:
            if isinstance(seg, dict):
                stype = seg.get("type")
                sdata = seg.get("data", {})
                if stype == "file":
                    u = sdata.get("url")
                    f = sdata.get("file")
                    n = sdata.get("name")
                    cand = None
                    if isinstance(u, str) and u and _looks_like_audio(u):
                        cand = u
                    elif (
                        isinstance(f, str)
                        and f
                        and (_looks_like_audio(f) or os.path.isabs(f))
                    ):
                        cand = f
                    elif isinstance(n, str) and n and _looks_like_audio(n):
                        if isinstance(u, str) and u:
                            cand = u
                        elif isinstance(f, str) and f:
                            cand = f
                    if cand:
                        audios.append(cand)
            elif isinstance(seg, Comp.File):
                u = getattr(seg, "url", None)
                f = getattr(seg, "file", None)
                n = getattr(seg, "name", None)
                cand = None
                if isinstance(u, str) and u and _looks_like_audio(u):
                    cand = u
                elif (
                    isinstance(f, str)
                    and f
                    and (_looks_like_audio(f) or os.path.isabs(f))
                ):
                    cand = f
                elif isinstance(n, str) and n and _looks_like_audio(n):
                    if isinstance(u, str) and u:
                        cand = u
                    elif isinstance(f, str) and f:
                        cand = f
                if isinstance(cand, str) and cand:
                    audios.append(cand)
            elif hasattr(Comp, "Node") and isinstance(seg, getattr(Comp, "Node")):
                content = getattr(seg, "content", None)
                if isinstance(content, list):
                    audios.extend(extract_audios_from_chain(content))
            elif hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            audios.extend(extract_audios_from_chain(c))
            elif hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            audios.extend(extract_audios_from_chain(c))
        except Exception:
            continue
    return audios


async def napcat_resolve_file_url(
    event: AstrMessageEvent, file_id: str
) -> Optional[str]:
    """使用 Napcat 接口将文件/视频的 file_id 解析为可下载 URL 或本地路径。"""
    if not (isinstance(file_id, str) and file_id):
        return None
    if not isinstance(event, AiocqhttpMessageEvent):
        return None
    if not (
        hasattr(event, "bot")
        and hasattr(event.bot, "api")
        and hasattr(event.bot.api, "call_action")
    ):
        return None

    try:
        gid = event.get_group_id()
    except Exception:
        gid = None

    def _build_candidates(raw_id: str) -> list[str]:
        rid = str(raw_id or "").strip()
        if not rid:
            return []
        cands = [rid]
        base = os.path.basename(rid)
        if base and base not in cands:
            cands.append(base)
        stem, ext = os.path.splitext(base)
        if stem and ext and stem not in cands:
            cands.append(stem)
        return cands

    actions = []
    for cand in _build_candidates(file_id):
        actions.extend(
            [
                {"action": "get_file", "params": {"file_id": cand}},
                {"action": "get_file", "params": {"file": cand}},
                {"action": "get_image", "params": {"file": cand}},
                {"action": "get_image", "params": {"file_id": cand}},
                {"action": "get_private_file_url", "params": {"file_id": cand}},
            ],
        )
        if gid:
            actions.append(
                {
                    "action": "get_group_file_url",
                    "params": {"group_id": gid, "file_id": cand},
                }
            )

    for item in actions:
        try:
            ret = await event.bot.api.call_action(item["action"], **item["params"])
            data = ob_data(ret)
            url = data.get("url")
            if isinstance(url, str) and url:
                return url
            f = data.get("file")
            if isinstance(f, str) and f:
                if f.startswith("file://"):
                    fp = f[7:]
                    if fp.startswith("/") and len(fp) > 3 and fp[2] == ":":
                        fp = fp[1:]
                    if os.path.exists(fp):
                        return os.path.abspath(fp)
                if os.path.exists(f):
                    return os.path.abspath(f)
        except Exception:
            continue
    return None


async def download_media_to_temp(url: str, size_mb_limit: int) -> Optional[str]:
    """下载媒体到临时文件。若已是本地文件则直接返回。"""
    source = str(url or "").strip()
    if not source:
        return None
    if os.path.isfile(source):
        return source
    if source.lower().startswith("file://"):
        local_path = source[7:]
        if local_path.startswith("/") and len(local_path) > 3 and local_path[2] == ":":
            local_path = local_path[1:]
        if os.path.isfile(local_path):
            return os.path.abspath(local_path)
        return None
    # 绝对路径/文件 ID 不是可下载的 URL。调用方应先通过 OneBot 解析文件 ID。
    if not source.lower().startswith(("http://", "https://")):
        return None
    max_bytes = size_mb_limit * 1024 * 1024

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(source, timeout=60) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"[媒体处理] 下载失败: HTTP {resp.status} (URL: {source})"
                    )
                    return None

                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit() and int(cl) > max_bytes:
                    logger.warning(
                        f"[媒体处理] 下载终止: 文件过大 ({int(cl) / (1024 * 1024):.1f}MB > {size_mb_limit}MB)"
                    )
                    return None

                # 根据 Content-Type 决定后缀。
                content_type = resp.headers.get("Content-Type", "").lower()
                if "image/gif" in content_type:
                    ext = ".gif"
                elif "image/" in content_type:
                    ext = ".jpg"
                elif "video/" in content_type:
                    ext = ".mp4"
                else:
                    ext = ".mp4"  # 默认

                tmp = tempfile.NamedTemporaryFile(
                    prefix="llm_media_", suffix=ext, delete=False
                )
                tmp_path = tmp.name
                tmp.close()

                total = 0
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        total += len(chunk)
                        if total > max_bytes:
                            os.remove(tmp_path)
                            logger.warning("[媒体处理] 下载终止: 实际下载数据超过限制")
                            return None
                        f.write(chunk)
                return tmp_path
    except Exception as e:
        logger.error(f"[媒体处理] 下载异常: {e} (URL: {url})")
    return None


def probe_duration_sec(ffmpeg_path: str, video_path: str) -> Optional[float]:
    """探测视频时长。"""
    # 优先使用与 ffmpeg 同目录的 ffprobe。
    ffprobe_path = None
    if ffmpeg_path:
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if ffmpeg_dir:
            cand = os.path.join(
                ffmpeg_dir, "ffprobe.exe" if os.name == "nt" else "ffprobe"
            )
            if os.path.exists(cand):
                ffprobe_path = cand

    if not ffprobe_path:
        ffprobe_path = shutil.which("ffprobe")

    if not ffprobe_path:
        return None

    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    try:
        res = _safe_subprocess_run(cmd)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return None


async def sample_frames_equidistant(
    ffmpeg_path: str, video_path: str, duration_sec: float, count: int
) -> List[str]:
    """等距抽帧。"""
    if not ffmpeg_path or not shutil.which(ffmpeg_path):
        ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return []

    out_dir = tempfile.mkdtemp(prefix="llm_frames_")
    frames = []
    loop = asyncio.get_running_loop()

    try:
        for i in range(1, count + 1):
            t = (i / (count + 1.0)) * duration_sec
            out_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            cmd = [
                ffmpeg_path,
                "-y",
                "-ss",
                f"{t:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-qscale:v",
                "2",
                out_path,
            ]
            res = await loop.run_in_executor(None, lambda: _safe_subprocess_run(cmd))
            if res.returncode == 0 and os.path.exists(out_path):
                frames.append(out_path)
    except Exception:
        pass
    return frames


async def extract_forward_media_keyframes(
    event: AstrMessageEvent,
    video_sources: List[str],
    max_count: int,
    max_frame_count: int,
    frame_interval_sec: int,
    ffmpeg_path: str,
    max_mb: int,
    max_duration: int,
    timeout_sec: int = 60,
) -> Tuple[List[str], List[str], List[str]]:
    """
    将聊天记录中的视频源转换为关键帧图片。
    抽帧策略：优先按 frame_interval_sec 估算抽帧数，再受 max_frame_count 上限约束。
    返回: (帧路径列表, 待清理路径列表, 本地视频路径列表)
    """
    frames = []
    cleanup_paths = []
    local_video_paths = []

    if max_count <= 0 or max_frame_count <= 0:
        return [], [], []

    for src in video_sources[:max_count]:
        src_str = str(src or "").strip()
        if not src_str:
            continue
        video_path = src_str
        is_temp_video = False

        # 兼容 file:// 本地路径。
        if video_path.startswith("file://"):
            fp = video_path[7:]
            if fp.startswith("/") and len(fp) > 3 and fp[2] == ":":
                fp = fp[1:]
            video_path = fp

        # 1. 解析 NapCat file_id（仅对非 URL 且非绝对路径值尝试）
        if not os.path.exists(video_path) and not video_path.startswith(
            ("http://", "https://")
        ):
            if os.path.isabs(video_path):
                # 一些平台会回传不可访问的容器内绝对路径，尝试用文件名回退解析。
                fallback_id = os.path.basename(video_path)
                if not fallback_id:
                    logger.debug(
                        f"video_parser: 本地绝对路径不可访问，跳过该视频: {video_path}"
                    )
                    continue
                logger.debug(
                    f"video_parser: 绝对路径不可访问，尝试按文件名回退解析: {fallback_id}"
                )
                resolved = await napcat_resolve_file_url(event, fallback_id)
                if resolved:
                    logger.debug(f"video_parser: 回退解析成功 -> {resolved}")
                    video_path = resolved
                else:
                    logger.debug(
                        f"video_parser: 回退解析失败，跳过该视频: {video_path}"
                    )
                    continue
            else:
                logger.debug(f"video_parser: 尝试解析 NapCat 文件 ID: {video_path}")
                resolved = await napcat_resolve_file_url(event, video_path)
                if resolved:
                    logger.debug(f"video_parser: 文件 ID 解析成功 -> {resolved}")
                    video_path = resolved
                else:
                    logger.debug(
                        f"video_parser: 文件 ID 解析失败，跳过该视频: {video_path}"
                    )
                    continue

        # 2. 下载远程视频
        if video_path.startswith(("http://", "https://")):
            logger.debug(f"video_parser: 正在下载并解析合并转发中的视频: {video_path}")
            try:
                downloaded = await asyncio.wait_for(
                    download_media_to_temp(video_path, max_mb),
                    timeout=timeout_sec,
                )
                if downloaded:
                    video_path = downloaded
                    is_temp_video = True
                    cleanup_paths.append(video_path)
                else:
                    continue
            except asyncio.TimeoutError:
                continue

        if not os.path.exists(video_path):
            continue

        # 3. 探测时长
        duration = await asyncio.to_thread(probe_duration_sec, ffmpeg_path, video_path)

        # 安全限制：硬编码 120 分钟 (7200秒)。
        safety_max_duration = 7200

        if duration is None or duration > safety_max_duration or duration <= 0:
            if is_temp_video and video_path in cleanup_paths:
                os.remove(video_path)
                cleanup_paths.remove(video_path)
            continue

        # 4. 抽帧数计算：优先按抽帧间隔估算，再受上限约束
        interval_sec = int(frame_interval_sec or 0)
        if interval_sec > 0:
            ideal_count = math.ceil(duration / interval_sec)
            sample_count = max(1, min(ideal_count, max_frame_count))
            if ideal_count > max_frame_count:
                actual_interval = duration / sample_count
                logger.debug(
                    f"[聊天记录解析] 视频时长 {duration:.1f}s 超过间隔覆盖范围，调整抽帧间隔: "
                    f"{interval_sec}s -> {actual_interval:.1f}s (上限 {max_frame_count} 帧)",
                )
        else:
            sample_count = max(1, max_frame_count)

        sampled = await sample_frames_equidistant(
            ffmpeg_path, video_path, duration, sample_count
        )
        if sampled:
            frames.extend(sampled)
            cleanup_paths.extend(sampled)
            local_video_paths.append(video_path)

    return frames, cleanup_paths, local_video_paths


async def extract_audio_wav(ffmpeg_path: str, video_path: str) -> Optional[str]:
    """从视频提取音频保存为 WAV 格式。"""
    if not os.path.exists(video_path):
        return None
    tmp = tempfile.NamedTemporaryFile(
        prefix="video_audio_", suffix=".wav", delete=False
    )
    out_path = tmp.name
    tmp.close()

    # ffmpeg 命令参考：-i input.mp4 -vn -ac 1 -ar 16000 -f wav output.wav
    cmd = [
        ffmpeg_path or "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        out_path,
    ]
    loop = asyncio.get_running_loop()

    def _run():
        return _safe_subprocess_run(cmd)

    try:
        res = await loop.run_in_executor(None, _run)
        if res.returncode != 0:
            try:
                os.remove(out_path)
            except Exception:
                pass
            return None
        return out_path if os.path.exists(out_path) else None
    except Exception:
        try:
            os.remove(out_path)
        except Exception:
            pass
        return None


async def prepare_video_context(
    event: AstrMessageEvent,
    video_sources: List[str],
    max_mb: int,
    max_duration: int,
    sample_count: int,
    ffmpeg_path: str,
    process_timeout: int = 30,
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """
    处理视频源，返回抽取的帧路径列表、待清理的路径列表、最终使用的视频本地路径，以及处理状态（失败原因）。
    """
    frames = []
    cleanup_paths = []
    final_video_path = None
    status = None  # 记录失败原因。

    start_time = time.time()

    for src in video_sources:
        if time.time() - start_time > process_timeout:
            logger.warning(f"视频解析超时（{process_timeout}s），来源: {src}")
            status = "timeout"
            break

        video_path = src
        is_temp_video = False

        # 1. 解析 Napcat file_id
        if not os.path.exists(src) and not src.startswith(("http://", "https://")):
            resolved = await napcat_resolve_file_url(event, src)
            if resolved:
                video_path = resolved
            else:
                status = "file_not_found"
                continue

        # 2. 下载远程视频
        if video_path.startswith(("http://", "https://")):
            try:
                # 注意：download_media_to_temp 内部已经处理了大小限制。
                downloaded = await asyncio.wait_for(
                    download_media_to_temp(video_path, max_mb),
                    timeout=process_timeout - (time.time() - start_time),
                )
                if downloaded:
                    video_path = downloaded
                    is_temp_video = True
                    cleanup_paths.append(video_path)
                else:
                    # 如果返回 None，很可能是因为文件过大。
                    status = "too_large"
                    continue
            except Exception:
                status = "download_failed"
                continue

        if not os.path.exists(video_path):
            status = "file_not_found"
            continue

        # 3. 探测时长
        duration = await asyncio.to_thread(probe_duration_sec, ffmpeg_path, video_path)

        # 安全限制：硬编码 120 分钟 (7200秒)。
        safety_max_duration = 7200

        if duration is None or duration > safety_max_duration or duration <= 0:
            if is_temp_video and video_path in cleanup_paths:
                try:
                    os.remove(video_path)
                except Exception as e:
                    logger.debug(
                        f"[媒体处理] 临时视频文件清理失败: {video_path}, err={e}"
                    )
                cleanup_paths.remove(video_path)
            status = "too_long"
            continue

        # 4. 抽帧
        sampled = await sample_frames_equidistant(
            ffmpeg_path, video_path, duration, sample_count
        )
        if sampled:
            frames.extend(sampled)
            cleanup_paths.extend(sampled)
            final_video_path = video_path
            status = "success"
            break
        else:
            status = "sample_failed"

    return frames, cleanup_paths, final_video_path, status


class MediaFrameProcessor:
    """统一的视频/GIF 帧处理器"""

    # 视频总结缓存: {message_id: {"summary": str, "expire": float}}
    _summary_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = asyncio.Lock()

    def __init__(self, context, event, config_getter):
        self.context = context
        self.event = event
        self._get_cfg = config_getter

    @classmethod
    async def get_cached_summary(cls, video_key: str) -> Optional[str]:
        """获取缓存的视频总结"""
        async with cls._cache_lock:
            if video_key in cls._summary_cache:
                item = cls._summary_cache[video_key]
                if time.time() < item["expire"]:
                    logger.debug(f"[媒体处理] 视频总结缓存命中: {video_key[:50]}...")
                    return item["summary"]
                else:
                    del cls._summary_cache[video_key]
        return None

    @classmethod
    async def set_cached_summary(
        cls, video_key: str, summary: str, ttl: int = VIDEO_SUMMARY_CACHE_TTL_SEC
    ):
        """设置视频总结缓存，默认有效期 24 小时"""
        async with cls._cache_lock:
            # FIFO：超限时移除最早插入的键。
            if len(cls._summary_cache) >= VIDEO_SUMMARY_CACHE_MAX_SIZE:
                oldest_key = next(iter(cls._summary_cache))
                del cls._summary_cache[oldest_key]

            cls._summary_cache[video_key] = {
                "summary": summary,
                "expire": time.time() + ttl,
            }
            logger.debug(f"[媒体处理] 视频总结已缓存: {video_key}")

    def _get_source_file_key(self) -> Optional[str]:
        """从事件消息链中提取 CQ file 字段作为内容指纹。"""
        try:
            message_obj = getattr(self.event, "message_obj", None)
            chain = getattr(message_obj, "message", []) if message_obj else None
            if not isinstance(chain, list):
                return None
            for seg in chain:
                if isinstance(seg, dict):
                    data = seg.get("data") or {}
                    file_val = str(data.get("file") or "").strip()
                    if file_val:
                        return file_val
                elif hasattr(seg, "type") and hasattr(seg, "data"):
                    file_val = str(getattr(seg.data, "file", "") or "").strip()
                    if file_val:
                        return file_val
        except Exception:
            pass
        return None

    async def _get_cached_summary_dual(self, msg_id: str) -> Optional[str]:
        """按 msg_id → file 键序查找缓存。"""
        cached = await self.get_cached_summary(msg_id)
        if cached:
            return cached
        file_key = self._get_source_file_key()
        if file_key:
            cached = await self.get_cached_summary(f"file:{file_key}")
        return cached

    async def _set_cached_summary_dual(self, msg_id: str, summary: str) -> None:
        """同时以 msg_id 和 file 键写入缓存。"""
        await self.set_cached_summary(msg_id, summary)
        file_key = self._get_source_file_key()
        if file_key:
            await self.set_cached_summary(f"file:{file_key}", summary)

    async def process_long_video(
        self,
        req: ProviderRequest,
        video_path: str,
        duration: float,
        sender_name: str = None,
        msg_id: str = None,
    ) -> bool:
        """【场景：视频】frame_pass → 抽帧直传；full_analysis → 逐帧汇总"""
        logger.debug(f"[媒体处理] 开始处理视频: {video_path}, msg_id: {msg_id}")
        try:
            video_mode = (
                str(self._get_cfg("video_mode", "full_analysis") or "full_analysis")
                .strip()
                .lower()
            )

            # frame_pass 不缓存。
            if video_mode != "frame_pass" and msg_id:
                cached_summary = await self._get_cached_summary_dual(msg_id)
                if cached_summary:
                    self._inject_summary(
                        req,
                        cached_summary,
                        "视频转述(缓存复用)",
                        sender_name=sender_name,
                    )
                    return True

            (
                frames,
                cleanup_paths,
                local_video_path,
                status,
            ) = await self._extract_video_frames(video_path, duration)

            if not frames:
                if status == "too_large":
                    self._inject_summary(
                        req,
                        "用户发送/引用了一个视频消息，但是文件太大了，我懒得看（已跳过解析）。",
                        "系统提示",
                        sender_name=sender_name,
                    )
                    return True
                if status == "too_long":
                    self._inject_summary(
                        req,
                        "用户发送/引用了一个视频消息，但是时间太长了，我没耐心看（已跳过解析）。",
                        "系统提示",
                        sender_name=sender_name,
                    )
                    return True
                logger.debug(
                    "[媒体处理] 视频解析跳过: status=%s, sender=%s, msg_id=%s",
                    status,
                    sender_name or "",
                    msg_id or "",
                )
                self._inject_summary(
                    req,
                    "用户发送/引用了一个视频消息，但未能成功解析（已跳过解析）。",
                    "系统提示",
                    sender_name=sender_name,
                )
                return True

            if cleanup_paths:
                req._cleanup_paths.extend(cleanup_paths)

            # frame_pass: 抽帧直传主模型，可选音频。
            if video_mode == "frame_pass":
                req.image_urls = [str(f) for f in frames]
                append_text_part_to_request(
                    req, "\n\n用户发了一个视频，以下为其关键帧：\n"
                )
                if self._get_cfg("video_asr_enable", False) and local_video_path:
                    audio_mode = (
                        str(self._get_cfg("audio_mode", "asr") or "asr").strip().lower()
                    )
                    if audio_mode == "passthrough":
                        provider = self._get_current_provider()
                        if provider and _provider_supports_audio_input(provider):
                            audio_path = await extract_audio_wav(
                                self._get_cfg("ffmpeg_path", ""),
                                local_video_path,
                            )
                            if audio_path:
                                if (
                                    not hasattr(req, "audio_urls")
                                    or req.audio_urls is None
                                ):
                                    req.audio_urls = []
                                req.audio_urls.append(audio_path)
                                req._cleanup_paths = req._cleanup_paths or []
                                req._cleanup_paths.append(audio_path)
                        else:
                            asr_text = await self._extract_and_transcribe_audio(
                                local_video_path
                            )
                            if asr_text:
                                append_text_part_to_request(
                                    req, f"[语音转写] {asr_text}\n"
                                )
                    else:
                        asr_text = await self._extract_and_transcribe_audio(
                            local_video_path
                        )
                        if asr_text:
                            append_text_part_to_request(req, f"[语音转写] {asr_text}\n")
                for frame in frames:
                    req._cleanup_paths = req._cleanup_paths or []
                    req._cleanup_paths.append(frame)
                return True

            # full_analysis: 并发 ASR + 逐帧识图 → 汇总。
            asr_text = None
            asr_task = None
            if self._get_cfg("video_asr_enable", False) and local_video_path:
                asr_task = asyncio.create_task(
                    self._extract_and_transcribe_audio(local_video_path),
                )

            frame_descriptions = None
            provider = self._get_vision_provider()
            if provider and frames:
                frame_descriptions = await self._describe_frames(
                    provider,
                    frames,
                    max_concurrency=2,
                    frame_timeout_sec=60,
                )

            if asr_task:
                try:
                    asr_text = await asyncio.wait_for(asr_task, timeout=60)
                except asyncio.TimeoutError:
                    logger.warning("[媒体处理] ASR 超时")
                    asr_text = None

            summary = await self._aggregate_frames_helper(
                frames,
                len(frames),
                duration,
                asr_text=asr_text,
                max_summary_length=200,
                frame_descriptions=frame_descriptions,
            )

            if summary:
                if msg_id:
                    await self._set_cached_summary_dual(msg_id, summary)
                else:
                    logger.warning("[媒体处理] msg_id 为空，跳过缓存写入")

                self._inject_summary(req, summary, "视频转述", sender_name=sender_name)
                logger.debug(
                    f"[媒体处理] 视频总结成功并注入，来源: {sender_name or '当前消息'}"
                )
                for frame in frames:
                    req._cleanup_paths = req._cleanup_paths or []
                    req._cleanup_paths.append(frame)
                return True
            else:
                logger.warning("[媒体处理] 视频转述生成失败，回退到首帧")
                if asr_text:
                    append_text_part_to_request(req, f"[语音转写] {asr_text}\n")
                req.image_urls.append(frames[0])
                for frame in frames:
                    req._cleanup_paths = req._cleanup_paths or []
                    req._cleanup_paths.append(frame)
                return True
        except Exception as e:
            logger.warning(f"[媒体处理] 视频处理失败: {e}")
            return False

    async def process_gif(
        self,
        req: ProviderRequest,
        gif_path: str,
        sender_name: str = None,
        msg_id: str = None,
    ) -> bool:
        """【场景：GIF 动图】frame_pass → 抽帧直传；full_analysis → 逐帧汇总"""
        try:
            # 1. 尝试从缓存获取（仅 full_analysis）
            gif_mode = (
                str(self._get_cfg("gif_mode", "full_analysis") or "full_analysis")
                .strip()
                .lower()
            )
            logger.debug(
                f"[GIF处理] GIF 处理: {gif_path}, mode: {gif_mode}, msg_id: {msg_id}"
            )
            if gif_mode != "frame_pass" and msg_id:
                cached_summary = await self._get_cached_summary_dual(msg_id)
                if cached_summary:
                    self._inject_summary(
                        req,
                        cached_summary,
                        "内容摘要(缓存复用)",
                        sender_name=sender_name,
                    )
                    return True

            ffmpeg_path = self._get_cfg("ffmpeg_path")
            duration = await self._probe_duration(gif_path)
            if not duration or duration <= 0:
                duration = 1.0

            # 参数化抽帧
            interval = float(self._get_cfg("gif_frame_interval_sec", 0.5) or 0.5)
            max_frames = int(self._get_cfg("gif_max_frames", 3) or 3)
            ideal = max(1, math.ceil(duration / interval)) if interval > 0 else 1
            sample_count = max(1, min(ideal, max_frames))

            logger.debug(
                f"[GIF处理] GIF 抽帧: 时长 {duration:.2f}s, 间隔 {interval}s, 上限 {max_frames}, 实抽 {sample_count}"
            )

            frame_paths = await sample_frames_equidistant(
                ffmpeg_path,
                gif_path,
                duration,
                sample_count,
            )

            if not frame_paths:
                logger.warning(f"[GIF处理] GIF 解析未产生有效帧: {gif_path}")
                return False

            frames = [str(p) for p in frame_paths]

            # frame_pass: 抽帧直传主模型。
            if gif_mode == "frame_pass":
                req.image_urls.extend([str(f) for f in frames])
                append_text_part_to_request(
                    req, "\n\n用户发了一张动图，以下为其关键帧：\n"
                )
                for frame in frames:
                    req._cleanup_paths = req._cleanup_paths or []
                    req._cleanup_paths.append(frame)
                return True

            # full_analysis: 帧聚合汇总。
            frame_count = len(frames)
            if frame_count == 1:
                logger.debug("[GIF处理] GIF仅抽取到1帧，跳过汇总流程")
                req.image_urls.append(frames[0])
                for frame in frames:
                    req._cleanup_paths = req._cleanup_paths or []
                    req._cleanup_paths.append(frame)
                return True

            summary = await self._aggregate_frames_helper(
                frames,
                frame_count,
                duration,
                asr_text=None,
                max_summary_length=50,
            )

            if summary:
                if msg_id:
                    await self._set_cached_summary_dual(msg_id, summary)
                else:
                    logger.warning("[GIF处理] GIF msg_id 为空，跳过缓存写入")

                self._inject_summary(req, summary, "内容摘要", sender_name=sender_name)
                req.image_urls.append(frames[-1])
                logger.debug("[GIF处理] GIF 总结成功")
            else:
                logger.warning("[GIF处理] GIF 总结为空，回退到首帧")
                req.image_urls.append(frames[0])

            for frame in frames:
                req._cleanup_paths = req._cleanup_paths or []
                req._cleanup_paths.append(frame)
            return True

        except Exception as e:
            logger.error(f"[GIF处理] GIF 处理异常: {e}", exc_info=True)
            return False

    # ==================== 私有辅助方法 ====================

    async def _probe_duration(self, media_path: str) -> float:
        """探测媒体时长"""
        try:
            duration = (
                await asyncio.to_thread(
                    probe_duration_sec,
                    self._get_cfg("ffmpeg_path", ""),
                    media_path,
                )
                or 0
            )
            return duration
        except Exception as e:
            logger.debug(f"[媒体处理] 时长探测失败: path={media_path}, err={e}")
            return 0

    async def _extract_video_frames(
        self, video_path: str, duration: float
    ) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
        """多帧抽取（读取配置参数）"""
        interval = self._get_cfg("video_frame_interval_sec", 12)
        max_frame_count = self._get_cfg("video_max_frame_count", 8)

        sample_count = 1
        if duration > 0:
            # 动态计算抽帧数：取 (时长/间隔) 和 (抽帧上限) 的较小值。
            ideal_count = math.ceil(duration / interval) if interval > 0 else 1
            sample_count = max(1, min(ideal_count, max_frame_count))

            # 如果实际抽帧数因为达到上限而被压缩，日志记录一下。
            if ideal_count > max_frame_count:
                actual_interval = duration / sample_count
                logger.debug(
                    f"[媒体处理] 视频时长 {duration:.1f}s 超过间隔覆盖范围，调整抽帧间隔: {interval}s -> {actual_interval:.1f}s (上限 {max_frame_count} 帧)"
                )
        else:
            sample_count = self._get_cfg("video_sample_count", 4)

        logger.debug(f"[媒体处理] 视频多帧抽取: {sample_count} 帧")

        try:
            (
                frames,
                cleanup_paths,
                local_video_path,
                status,
            ) = await prepare_video_context(
                self.event,
                [video_path],
                max_mb=self._get_cfg("video_max_size_mb", 50),
                max_duration=7200,  # 硬编码安全限制 120 分钟。
                sample_count=sample_count,
                ffmpeg_path=self._get_cfg("ffmpeg_path", ""),
                process_timeout=120,
            )

            return frames or [], cleanup_paths or [], local_video_path, status
        except Exception as e:
            logger.warning(f"[媒体处理] 帧抽取失败: {e}")
            return [], [], None, "error"

    async def _extract_and_transcribe_audio(self, video_path: str) -> Optional[str]:
        """ASR：提取音频并转录"""
        try:
            logger.debug("[媒体处理] ASR: 正在从视频中提取音频...")
            ffmpeg_path = self._get_cfg("ffmpeg_path", "")
            wav_path = await extract_audio_wav(ffmpeg_path, video_path)

            if not wav_path or not os.path.exists(wav_path):
                logger.warning("[媒体处理] ASR: 提取音频 WAV 失败")
                return None

            asr_text = None
            try:
                logger.debug("[媒体处理] ASR: 正在请求转录...")
                asr_text, llm_cleanup_paths = await transcribe_audio_with_fallback(
                    context=self.context,
                    get_cfg=self._get_cfg,
                    event=self.event,
                    audio_path=wav_path,
                    audio_is_prepared_wav=True,
                )
                for cleanup_path in llm_cleanup_paths:
                    try:
                        os.remove(cleanup_path)
                    except Exception as e:
                        logger.debug(
                            f"[媒体处理] ASR LLM 临时音频清理失败: path={cleanup_path}, err={e}"
                        )

                if asr_text:
                    logger.debug(f"[媒体处理] ASR 成功: {asr_text[:100]}...")
                else:
                    logger.warning(
                        "[媒体处理] ASR 未返回有效文本（Provider 可能不支持音频，或音频内容为空）"
                    )
            except Exception as e:
                logger.warning(f"[媒体处理] ASR 调用失败: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except Exception as e:
                    logger.debug(
                        f"[媒体处理] ASR wav 清理失败: path={wav_path}, err={e}"
                    )

            return asr_text
        except Exception as e:
            logger.warning(f"[媒体处理] ASR 处理失败: {e}")
            return None

    # ---- 逐帧识图（单帧） ----

    async def _describe_single_frame(
        self,
        provider,
        frame_path: str,
        idx: int,
        frame_count: int,
        timeout_sec: int = 60,
    ) -> Optional[str]:
        start_t = time.time()
        logger.debug(f"[帧聚合] 正在请求第 {idx}/{frame_count} 帧 ...")
        for attempt in (1, 2):
            try:
                response = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=f"简要描述这一帧（第 {idx}/{frame_count} 帧）的内容，重点关注动作、表情和关键物体。如果无法处理请直接回复【处理失败】。",
                        system_prompt="你是一个视频分析助手。请用 1-2 句话简洁描述图片内容。",
                        image_urls=[frame_path],
                        context=[],
                    ),
                    timeout=max(timeout_sec, 5),
                )
                desc = self._extract_completion_text(response)
                cost = time.time() - start_t
                if desc:
                    if "处理失败" in desc:
                        logger.warning(f"[帧聚合] 第 {idx}/{frame_count} 帧被拒识")
                    else:
                        logger.debug(
                            f"[帧聚合] 第 {idx}/{frame_count} 帧解析成功 (耗时: {cost:.2f}s). 响应: {desc}"
                        )
                        return f"[第 {idx} 帧] {desc}"
                else:
                    logger.warning(f"[帧聚合] 第 {idx}/{frame_count} 帧解析响应为空")
            except asyncio.TimeoutError:
                logger.warning(
                    f"[帧聚合] 第 {idx}/{frame_count} 帧识图超时 ({timeout_sec}s)"
                )
                if attempt == 1:
                    await asyncio.sleep(2)
                    continue
                break
            except Exception as e:
                logger.warning(
                    f"[帧聚合] 第 {idx}/{frame_count} 帧识图异常 (attempt {attempt}): {e}"
                )
                if attempt == 1:
                    await asyncio.sleep(2)
                    continue
            break
        return f"[第 {idx} 帧] [处理失败]"

    # ---- 逐帧识图（有限并发） ----

    async def _describe_frames(
        self,
        provider,
        frames: List[str],
        max_concurrency: int = 2,
        frame_timeout_sec: int = 60,
    ) -> List[str]:
        sem = asyncio.Semaphore(max_concurrency)
        frame_count = len(frames)

        async def _describe_one(idx: int, frame_path: str) -> Optional[str]:
            async with sem:
                return await self._describe_single_frame(
                    provider,
                    frame_path,
                    idx,
                    frame_count,
                    timeout_sec=frame_timeout_sec,
                )

        tasks = [_describe_one(idx, fp) for idx, fp in enumerate(frames, 1)]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    async def _aggregate_frames_helper(
        self,
        frames: List[str],
        frame_count: int,
        duration: float,
        asr_text: Optional[str] = None,
        max_summary_length: int = 200,
        frame_descriptions: Optional[List[str]] = None,
    ) -> Optional[str]:
        """【公用】帧聚合汇总"""
        if not frames:
            return None

        max_summary_length = max_summary_length or 200

        logger.debug(f"[帧聚合] 开始处理 {len(frames)} 帧图片...")

        # 1. 逐帧识图（支持外部传入，否则自动并行处理）
        if frame_descriptions is None:
            provider = self._get_vision_provider()
            if not provider:
                logger.warning("[帧聚合] 无可用 Vision Provider")
                return None
            frame_descriptions = await self._describe_frames(
                provider,
                frames,
                max_concurrency=2,
                frame_timeout_sec=60,
            )

        if not frame_descriptions:
            return None

        # 预检：全部帧描述均为处理失败 → 仅在有 ASR 时继续（仅用音频生成摘要）。
        if all(d.endswith("[处理失败]") for d in frame_descriptions if d):
            if not asr_text:
                logger.warning("[帧聚合] 所有帧画面分析均失败且无 ASR 内容，跳过汇总")
                return None
            logger.warning("[帧聚合] 所有帧画面分析均失败，仅基于 ASR 音频转写生成摘要")
            frame_descriptions = None

        summary_start_t = time.time()

        # 2. 合并帧描述和 ASR
        comprehensive_context = ""
        if frame_descriptions:
            comprehensive_context = "\n".join(frame_descriptions)

        if asr_text:
            if comprehensive_context:
                comprehensive_context += f"\n\n[视频语音转写]\n{asr_text}"
            else:
                comprehensive_context = f"[视频语音转写]\n{asr_text}"
            logger.debug("[帧聚合] 已融合 ASR 内容")

        # 3. LLM 汇总
        try:
            llm_provider = self._get_summary_provider()
            if not llm_provider:
                logger.warning("[帧聚合] 无可用 LLM Provider")
                return None

            logger.debug("[帧聚合] 正在请求汇总摘要...")

            summary_prompt = (
                f"根据以下逐帧视觉描述和语音转写（如有），生成连贯的汇总摘要。要求：\n"
                f"1. 重点概括视频/动图的核心内容、动作变化和视觉亮点。\n"
                f"2. 严格控制字数在 {max_summary_length} 字以内。\n"
                f"3. 直接输出摘要内容，不要有任何前缀或解释。\n\n"
                f"--- 原始素材数据开始 ---\n"
                f"{comprehensive_context}\n"
                f"--- 原始素材数据结束 ---\n\n"
                f"请生成汇总摘要："
            )

            response = await asyncio.wait_for(
                llm_provider.text_chat(
                    prompt=summary_prompt,
                    system_prompt="你是媒体摘要助手。直接输出摘要，不要前缀或解释。",
                    image_urls=[],
                    context=[],
                ),
                timeout=30,
            )

            summary = self._extract_completion_text(response)

            if summary:
                if len(summary) > max_summary_length:
                    logger.debug(
                        f"[帧聚合] 摘要长度 {len(summary)} 超过限制 {max_summary_length}，截断处理"
                    )
                    summary = summary[:max_summary_length].rsplit("。", 1)[0] + "。"

                logger.debug(
                    f"[帧聚合] 汇总成功 (耗时: {time.time() - summary_start_t:.2f}s, 字数: {len(summary)}): {summary}"
                )
                return summary
            else:
                logger.warning("[帧聚合] 汇总摘要响应为空")
        except asyncio.TimeoutError:
            logger.warning("[帧聚合] 汇总摘要超时")
        except Exception as e:
            logger.warning(f"[帧聚合] LLM 汇总失败: {e}")

        return None

    def _inject_summary(
        self, req: ProviderRequest, summary: str, label: str, sender_name: str = None
    ):
        """注入总结"""
        user_question = req.prompt.strip()
        sender_prefix = (
            f"该媒体消息由 {sender_name} 发送/提供。\n" if sender_name else ""
        )
        context_prompt = (
            f"\n\n以下是系统为你分析的{label}，请结合此{label}来响应用户的要求。信息如下：\n"
            f"--- 注入内容开始 ---\n"
            f"{sender_prefix}[{label}] {summary}\n"
            f"--- 注入内容结束 ---"
        )
        if not append_text_part_to_request(req, context_prompt, mark_temp=False):
            req.prompt = user_question + context_prompt

    def _find_provider(self, provider_id: str):
        """通用方法：从所有 Provider（包含 LLM 和 STT）中查找匹配的 ID/Name"""
        return find_provider(self.context, provider_id)

    def _get_current_provider(self):
        """获取当前会话 Provider"""
        try:
            return self.context.get_using_provider(umo=self.event.unified_msg_origin)
        except Exception as e:
            logger.debug(f"[媒体处理] 获取会话 Provider 失败: err={e}")
        return None

    def _get_vision_provider(self):
        """获取 Vision Provider"""
        provider_id = self._get_cfg("video_image_provider_id")
        p = self._find_provider(provider_id)
        if p:
            return p

        if provider_id:
            logger.warning(f"[媒体处理] 指定的 Vision Provider {provider_id} 未找到")

        # 自动选择（当前正在使用的）
        try:
            default_p = self.context.get_using_provider(
                umo=self.event.unified_msg_origin
            )
            if default_p:
                return default_p
        except Exception as e:
            logger.debug(f"[媒体处理] 获取会话 Vision Provider 失败: err={e}")

        return None

    def _get_stt_provider(self):
        """获取 STT Provider"""
        asr_pid = self._get_cfg("audio_asr_provider_id")
        p = self._find_provider(asr_pid)
        if p:
            return p

        if asr_pid:
            logger.warning(f"[媒体处理] 未找到指定的 STT Provider: {asr_pid}")

        try:
            stt_p = self.context.get_using_stt_provider(
                umo=self.event.unified_msg_origin
            )
            return stt_p
        except Exception as e:
            logger.debug(f"[媒体处理] 获取会话 STT Provider 失败: err={e}")

        return None

    def _get_summary_provider(self):
        """获取视频/GIF 汇总摘要 Provider"""
        provider_id = self._get_cfg("video_summary_provider_id")
        p = self._find_provider(provider_id)
        if p:
            return p

        if provider_id:
            logger.warning(f"[媒体处理] 未找到指定的视频摘要 Provider: {provider_id}")

        try:
            default_p = self.context.get_using_provider(
                umo=self.event.unified_msg_origin
            )
            if default_p:
                return default_p
        except Exception as e:
            logger.debug(f"[媒体处理] 获取会话摘要 Provider 失败: err={e}")

        return None

    def _extract_completion_text(self, response) -> str:
        """提取响应文本"""
        if hasattr(response, "completion_text"):
            return response.completion_text.strip()
        elif isinstance(response, dict):
            return response.get("completion_text", "").strip()
        return ""


def _extract_videos_from_raw_event(event: AstrMessageEvent) -> List[str]:
    """从 raw_message 结构兜底提取视频源。"""
    try:
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        raw_data = ob_data(raw)
        if not raw_data and hasattr(raw, "get"):
            try:
                message = raw.get("message")
                if isinstance(message, list):
                    raw_data = {"message": message}
            except Exception:
                raw_data = {}
        if not raw_data and hasattr(raw, "message"):
            message = getattr(raw, "message", None)
            if isinstance(message, list):
                raw_data = {"message": message}
        if not raw_data and hasattr(event, "event"):
            event_raw = getattr(event, "event", None)
            raw_data = ob_data(event_raw)
            if not raw_data and hasattr(event_raw, "get"):
                try:
                    message = event_raw.get("message")
                    if isinstance(message, list):
                        raw_data = {"message": message}
                except Exception:
                    raw_data = {}
            if not raw_data and hasattr(event_raw, "message"):
                message = getattr(event_raw, "message", None)
                if isinstance(message, list):
                    raw_data = {"message": message}
        if not isinstance(raw_data, dict):
            return []
        chain = raw_data.get("message")
        if not isinstance(chain, list) or not chain:
            return []
        return extract_videos_from_chain(chain)
    except Exception:
        return []


def _normalize_video_source_for_event(event: AstrMessageEvent, source: str) -> str:
    """按平台归一化视频来源路径，优先返回可下载 URL。"""
    src = str(source or "").strip()
    if not src:
        return ""
    if src.startswith(("http://", "https://", "file://")) or os.path.isabs(src):
        return src

    # Telegram 常见 file_path 为相对远程路径，需拼接 base_file_url。
    try:
        bot = getattr(event, "bot", None)
        base_file_url = getattr(bot, "base_file_url", None) if bot else None
        if isinstance(base_file_url, str) and base_file_url.strip():
            return f"{base_file_url.rstrip('/')}/{src.lstrip('/')}"
    except Exception:
        pass
    return src


def _is_direct_media_source(source: str) -> bool:
    src = str(source or "").strip()
    if not src:
        return False
    return src.startswith(("http://", "https://", "file://")) or os.path.isabs(src)


async def _extract_videos_from_telegram_update(event: AstrMessageEvent) -> List[str]:
    """从 Telegram Update 对象兜底提取视频源。"""
    try:
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        update_message = getattr(raw, "message", None)
        if update_message is None:
            return []

        sources: List[str] = []
        video_obj = getattr(update_message, "video", None)
        if video_obj is not None:
            try:
                file_obj = await video_obj.get_file()
                file_path = str(getattr(file_obj, "file_path", "") or "").strip()
                if file_path:
                    sources.append(file_path)
            except Exception:
                pass

        doc_obj = getattr(update_message, "document", None)
        if doc_obj is not None:
            mime_type = str(getattr(doc_obj, "mime_type", "") or "").lower()
            file_name = str(getattr(doc_obj, "file_name", "") or "").lower()
            if mime_type.startswith("video/") or file_name.endswith(
                (
                    ".mp4",
                    ".mov",
                    ".avi",
                    ".mkv",
                    ".webm",
                    ".flv",
                    ".wmv",
                    ".m4v",
                    ".gif",
                ),
            ):
                try:
                    file_obj = await doc_obj.get_file()
                    file_path = str(getattr(file_obj, "file_path", "") or "").strip()
                    if file_path:
                        sources.append(file_path)
                except Exception:
                    pass

        deduped: List[str] = []
        for src in sources:
            if src and src not in deduped:
                deduped.append(src)
        return deduped
    except Exception:
        return []


async def _probe_duration_helper(get_cfg, media_path: str) -> float:
    try:
        duration = (
            await asyncio.to_thread(
                probe_duration_sec,
                get_cfg("ffmpeg_path", ""),
                media_path,
            )
            or 0
        )
        return duration
    except Exception as e:
        logger.warning(f"[时长探测] 异常: {e}")
        return 0


async def _extract_video_sources_with_msg_ids_via_get_msg(
    event: AstrMessageEvent,
    msg_ids: List[str],
) -> List[Tuple[str, str]]:
    if not isinstance(event, AiocqhttpMessageEvent):
        return []
    if not msg_ids:
        return []
    try:
        client = event.bot
    except Exception:
        return []

    pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for msg_id in msg_ids:
        normalized_msg_id = str(msg_id).strip()
        if not normalized_msg_id:
            continue
        try:
            original_msg = await client.api.call_action(
                "get_msg", message_id=normalized_msg_id
            )
            if _is_unavailable_get_msg_payload(original_msg):
                continue
            if original_msg and "message" in original_msg:
                extracted = extract_videos_from_chain(original_msg["message"])
                for source in extracted:
                    key = (normalized_msg_id, source)
                    if source and key not in seen:
                        seen.add(key)
                        pairs.append(key)
        except Exception:
            continue
    return pairs


async def detect_media_scenario(
    req: ProviderRequest,
    get_cfg,
    video_sources: Optional[List[str]] = None,
) -> MediaContext:
    """检测当前媒体场景并返回上下文。"""
    ctx = MediaContext()

    if video_sources and len(video_sources) > 0:
        ctx.media_path = video_sources[0]
    else:
        ctx.scenario = MediaScenario.NONE
        return ctx

    first_path = ctx.media_path
    if not first_path or not isinstance(first_path, str):
        ctx.scenario = MediaScenario.NONE
        return ctx

    if first_path.startswith(("http://", "https://")):
        try:
            max_size = get_cfg("video_max_size_mb", 50)
            local_path = await download_media_to_temp(first_path, max_size)
            if local_path:
                ctx.media_path = local_path
                ctx.cleanup_paths.append(local_path)
                first_path = local_path
            else:
                ctx.scenario = MediaScenario.NONE
                return ctx
        except Exception:
            ctx.scenario = MediaScenario.NONE
            return ctx

    ctx.media_path = first_path
    if is_gif_file(first_path):
        if not bool(get_cfg("gif_parse_enable", True)):
            logger.debug(
                "[GIF处理] GIF 解析已关闭，保留框架原生图片处理链: %s", first_path
            )
            ctx.scenario = MediaScenario.NONE
            return ctx
        gif_mode = (
            str(get_cfg("gif_mode", "full_analysis") or "full_analysis").strip().lower()
        )
        if gif_mode == "direct":
            ctx.scenario = MediaScenario.GIF_DIRECT
            ctx.media_path = first_path
            return ctx
        ctx.duration = await _probe_duration_helper(get_cfg, first_path)
        if ctx.duration <= 0:
            ctx.scenario = MediaScenario.NONE
            return ctx
        ctx.scenario = MediaScenario.GIF_ANIMATED
        return ctx

    suffix = Path(first_path).suffix.lower()
    is_from_video_source = bool(
        video_sources and len(video_sources) > 0 and ctx.media_path == video_sources[0]
    )
    is_video_format = (
        suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]
        or is_from_video_source
    )
    if not is_video_format:
        try:
            with open(first_path, "rb") as f:
                header = f.read(32)
                if b"ftyp" in header or b"matroska" in header or b"fLaC" in header:
                    is_video_format = True
        except Exception:
            pass

    if is_video_format:
        if not bool(get_cfg("video_parse_enable", True)):
            logger.debug("[媒体处理] 视频解析已关闭，跳过: %s", first_path)
            ctx.scenario = MediaScenario.NONE
            return ctx
        ctx.duration = await _probe_duration_helper(get_cfg, first_path)
        if ctx.duration <= 0:
            ctx.scenario = MediaScenario.NONE
            return ctx
        ctx.scenario = MediaScenario.VIDEO
        return ctx

    ctx.scenario = MediaScenario.NONE
    return ctx


async def _scan_gifs_from_image_urls(
    req: ProviderRequest,
    get_cfg,
) -> Tuple[List[Tuple[str, MediaScenario, List[str]]], List[str]]:
    """扫描 req.image_urls 中的 GIF 文件并分类。

    Returns:
        gif_infos: [(local_path, scenario, cleanup_paths), ...]
        rest_urls: 非 GIF 图片 URL 保留列表
    """
    image_urls = getattr(req, "image_urls", []) or []
    if not image_urls:
        return [], []

    gif_infos: List[Tuple[str, MediaScenario, List[str]]] = []
    rest: List[str] = []

    for url in image_urls:
        first_path = url
        cleanup: List[str] = []

        if isinstance(first_path, str) and first_path.startswith(
            ("http://", "https://")
        ):
            try:
                max_size = int(get_cfg("video_max_size_mb", 50))
                local_path = await download_media_to_temp(first_path, max_size)
                if local_path:
                    cleanup.append(local_path)
                    first_path = local_path
                else:
                    rest.append(url)
                    continue
            except Exception:
                rest.append(url)
                continue

        if not is_gif_file(first_path):
            rest.append(url)
            continue

        gif_mode = (
            str(get_cfg("gif_mode", "full_analysis") or "full_analysis").strip().lower()
        )
        if gif_mode == "direct":
            gif_infos.append((first_path, MediaScenario.GIF_DIRECT, cleanup))
        else:
            duration = await _probe_duration_helper(get_cfg, first_path)
            if duration and duration > 0:
                gif_infos.append((first_path, MediaScenario.GIF_ANIMATED, cleanup))
            else:
                rest.append(url)

    return gif_infos, rest


async def process_media_content(
    context: Any,
    event: AstrMessageEvent,
    req: ProviderRequest,
    all_components: List[Any],
    reply_seg: Optional[Comp.Reply],
    get_cfg,
) -> bool:
    """处理视频/GIF 媒体注入。"""
    if not get_cfg("video_parse_enable", True) and not get_cfg(
        "gif_parse_enable", True
    ):
        return False

    dynamic_batch_msg_ids = (
        event.get_extra("_llme_dynamic_batch_msg_ids", default=[]) or []
    )
    dynamic_batch_msg_ids = [
        str(mid).strip() for mid in dynamic_batch_msg_ids if str(mid).strip()
    ]
    video_sources = extract_videos_from_chain(all_components)
    video_source_msg_id: Optional[str] = None
    raw_video_sources: List[str] = []
    if not dynamic_batch_msg_ids:
        raw_video_sources = _extract_videos_from_raw_event(event)
        if not raw_video_sources:
            raw_video_sources = await _extract_videos_from_telegram_update(event)
    if raw_video_sources:
        video_sources = raw_video_sources + [
            src for src in video_sources if src not in raw_video_sources
        ]

    if dynamic_batch_msg_ids:
        fetched_video_pairs = await _extract_video_sources_with_msg_ids_via_get_msg(
            event, dynamic_batch_msg_ids
        )
        if fetched_video_pairs:
            fetched_video_sources = [src for _msg_id, src in fetched_video_pairs]
            video_source_msg_id = fetched_video_pairs[0][0]
            video_sources = fetched_video_sources + [
                src for src in video_sources if src not in fetched_video_sources
            ]
            logger.debug(
                "[LLMEnhancement] 媒体解析 get_msg 兜底命中："
                f"batch_msg_ids={dynamic_batch_msg_ids}, source_msg_id={video_source_msg_id}, fetched_video_sources={fetched_video_sources}",
            )
    elif not raw_video_sources:
        current_msg_id = getattr(
            getattr(event, "message_obj", None), "message_id", None
        )
        current_msg_id = (
            str(current_msg_id).strip() if current_msg_id is not None else ""
        )
        if current_msg_id:
            fetched_video_pairs = await _extract_video_sources_with_msg_ids_via_get_msg(
                event, [current_msg_id]
            )
            if fetched_video_pairs:
                fetched_video_sources = [src for _msg_id, src in fetched_video_pairs]
                video_source_msg_id = fetched_video_pairs[0][0]
                video_sources = fetched_video_sources + [
                    src for src in video_sources if src not in fetched_video_sources
                ]
                logger.debug(
                    "[LLMEnhancement] 媒体解析 get_msg 兜底命中："
                    f"msg_id={current_msg_id}, source_msg_id={video_source_msg_id}, fetched_video_sources={fetched_video_sources}",
                )

    if not video_sources and reply_seg:
        if isinstance(event, AiocqhttpMessageEvent):
            try:
                client = event.bot
                original_msg = await client.api.call_action(
                    "get_msg", message_id=reply_seg.id
                )
                if _is_unavailable_get_msg_payload(original_msg):
                    original_msg = None
                if original_msg and "message" in original_msg:
                    video_sources = extract_videos_from_chain(original_msg["message"])
            except Exception:
                pass
        reply_chain = getattr(reply_seg, "chain", None)
        if isinstance(reply_chain, list) and reply_chain:
            reply_chain_sources = extract_videos_from_chain(reply_chain)
            if reply_chain_sources:
                video_sources.extend(
                    [src for src in reply_chain_sources if src not in video_sources],
                )

    if video_sources:
        normalized_sources: List[str] = []
        for src in video_sources:
            normalized = _normalize_video_source_for_event(event, src)
            if (
                normalized
                and (not normalized.startswith(("http://", "https://", "file://")))
                and (not os.path.isabs(normalized))
            ):
                resolved = await napcat_resolve_file_url(event, normalized)
                if resolved:
                    normalized = resolved
            if normalized and normalized not in normalized_sources:
                normalized_sources.append(normalized)
        normalized_sources.sort(
            key=lambda src: 0 if _is_direct_media_source(src) else 1
        )
        video_sources = normalized_sources

    media_ctx = await detect_media_scenario(req, get_cfg, video_sources)
    req._cleanup_paths.extend(media_ctx.cleanup_paths)
    processor = MediaFrameProcessor(context, event, get_cfg)

    # VIDEO 场景（优先，走完整抽帧+ASR+汇总管线）
    if media_ctx.scenario == MediaScenario.VIDEO:
        if len(req.image_urls) > 0:
            req.image_urls = [
                url
                for url in req.image_urls
                if not any(
                    url.lower().endswith(s)
                    for s in [".mp4", ".mov", ".avi", ".wmv", ".flv", ".m4v"]
                )
            ]
        quoted_sender = getattr(req, "_quoted_sender", None) if reply_seg else None
        current_msg_id = getattr(
            getattr(event, "message_obj", None), "message_id", None
        )
        msg_id = (
            str(reply_seg.id)
            if reply_seg
            else str(video_source_msg_id or current_msg_id)
        )
        return await processor.process_long_video(
            req,
            media_ctx.media_path,
            media_ctx.duration,
            sender_name=quoted_sender,
            msg_id=msg_id,
        )

    # GIF 处理：合并来自 video_sources 的第一张 GIF + image_urls 中的多 GIF。
    gif_batch: List[Tuple[str, MediaScenario, List[str]]] = []
    if media_ctx.scenario in (MediaScenario.GIF_DIRECT, MediaScenario.GIF_ANIMATED):
        gif_batch.append((media_ctx.media_path, media_ctx.scenario, []))
    if not video_sources and get_cfg("gif_parse_enable", True):
        extra_gifs, rest_urls = await _scan_gifs_from_image_urls(req, get_cfg)
        req.image_urls = rest_urls
        gif_batch.extend(extra_gifs)

    # ==================== 音频文件 ASR ====================
    audio_handled = False
    audio_sources = extract_audios_from_chain(all_components)
    if audio_sources:
        audio_mode = str(get_cfg("audio_mode", "asr") or "asr").strip().lower()
        if audio_mode != "off":
            for audio_url in audio_sources:
                try:
                    if (
                        audio_url
                        and (
                            not audio_url.startswith(("http://", "https://", "file://"))
                        )
                        and (not os.path.isabs(audio_url))
                    ):
                        resolved = await napcat_resolve_file_url(event, audio_url)
                        if resolved:
                            audio_url = resolved
                    audio_path = await download_media_to_temp(
                        audio_url, size_mb_limit=10
                    )
                    if not audio_path:
                        continue
                    text, cleanup_paths = await transcribe_audio_with_fallback(
                        context=context,
                        get_cfg=get_cfg,
                        event=event,
                        audio_path=audio_path,
                    )
                    req._cleanup_paths.extend(cleanup_paths)
                    text = str(text or "").strip()
                    if text:
                        append_text_part_to_request(req, f"\n\n[音频文件转写] {text}\n")
                        audio_handled = True
                except Exception as e:
                    logger.debug(f"[LLMEnhancement] 音频文件 ASR 失败: {e}")

    if gif_batch:
        handled = False
        quoted_sender = getattr(req, "_quoted_sender", None) if reply_seg else None
        current_msg_id = getattr(
            getattr(event, "message_obj", None), "message_id", None
        )
        msg_id = (
            str(reply_seg.id)
            if reply_seg
            else str(video_source_msg_id or current_msg_id)
        )
        for gif_path, scenario, cleanups in gif_batch:
            req._cleanup_paths.extend(cleanups)
            if scenario == MediaScenario.GIF_DIRECT:
                cached_desc = await MediaFrameProcessor.get_cached_summary(msg_id)
                gif_file_key = (
                    processor._get_source_file_key() if not cached_desc else None
                )
                if not cached_desc and gif_file_key:
                    cached_desc = await MediaFrameProcessor.get_cached_summary(
                        f"file:{gif_file_key}"
                    )
                if cached_desc:
                    append_text_part_to_request(req, f"\n\n[GIF 描述] {cached_desc}\n")
                    handled = True
                else:
                    gif_pid = get_cfg("gif_provider_id", "")
                    provider = (
                        processor._find_provider(gif_pid)
                        if gif_pid
                        else processor._get_vision_provider()
                    )
                    if provider:
                        try:
                            response = await provider.text_chat(
                                prompt="请用中文描述这个动图的内容，包含：主体动作变化、场景切换、表情互动、画面中的文字。控制在60字以内。如果无法处理请直接回复【处理失败】。",
                                image_urls=[gif_path],
                                context=[],
                            )
                            desc = processor._extract_completion_text(response)
                            if desc and "处理失败" not in desc:
                                append_text_part_to_request(
                                    req, f"\n\n[GIF 描述] {desc}\n"
                                )
                                await MediaFrameProcessor.set_cached_summary(
                                    msg_id, desc
                                )
                                if gif_file_key:
                                    await MediaFrameProcessor.set_cached_summary(
                                        f"file:{gif_file_key}", desc
                                    )
                                handled = True
                        except Exception as e:
                            logger.warning(f"[LLMEnhancement] GIF direct 描述失败: {e}")
            elif scenario == MediaScenario.GIF_ANIMATED:
                if await processor.process_gif(
                    req, gif_path, sender_name=quoted_sender, msg_id=msg_id
                ):
                    handled = True
        return handled or audio_handled

    return audio_handled
