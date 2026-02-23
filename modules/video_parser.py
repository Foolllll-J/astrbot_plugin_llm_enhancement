from __future__ import annotations
import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import math
import json
from pathlib import Path
from enum import Enum
from typing import List, Optional, Any, Dict, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
import astrbot.api.message_components as Comp
from .provider_utils import find_provider

# ==================== 媒体处理场景枚举 ====================

class MediaScenario(Enum):
    """媒体处理场景"""
    NONE = "none"                          # 无媒体
    FORWARD_MESSAGE = "forward_message"    # 转发消息（包含文本/图片/视频）
    VIDEO = "video"                        # 视频（统一抽帧流程）
    GIF_ANIMATED = "gif_animated"          # GIF 动图

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

    # GIF87a, GIF89a
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
        ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv", 
        ".flv", ".wmv", ".ts", ".mpeg", ".mpg", ".3gp", ".gif",
    )

    def _looks_like_video(name_or_url: str) -> bool:
        if not isinstance(name_or_url, str) or not name_or_url:
            return False
        s = name_or_url.lower()
        return any(s.endswith(ext) for ext in video_exts)

    for seg in chain:
        try:
            if isinstance(seg, dict):
                # 处理 OneBot v11 原始字典格式
                stype = seg.get("type")
                sdata = seg.get("data", {})
                if stype == "video":
                    u = sdata.get("url")
                    f = sdata.get("file")
                    if isinstance(u, str) and u: videos.append(u)
                    elif isinstance(f, str) and f: videos.append(f)
                elif stype == "file":
                    u = sdata.get("url")
                    f = sdata.get("file")
                    n = sdata.get("name")
                    cand = None
                    if isinstance(u, str) and u and _looks_like_video(u): cand = u
                    elif isinstance(f, str) and f and (_looks_like_video(f) or os.path.isabs(f)): cand = f
                    elif isinstance(n, str) and n and _looks_like_video(n) and isinstance(f, str) and f: cand = f
                    if cand: videos.append(cand)
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
                elif isinstance(f, str) and f and (_looks_like_video(f) or os.path.isabs(f)):
                    cand = f
                elif isinstance(n, str) and n and _looks_like_video(n) and isinstance(f, str) and f:
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
                # Forward 组件可能包含 nodes
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            videos.extend(extract_videos_from_chain(c))
        except Exception:
            continue
    return videos

async def napcat_resolve_file_url(event: AstrMessageEvent, file_id: str) -> Optional[str]:
    """使用 Napcat 接口将文件/视频的 file_id 解析为可下载 URL 或本地路径。"""
    if not (isinstance(file_id, str) and file_id):
        return None
    if not (hasattr(event, "bot") and hasattr(event.bot, "api") and hasattr(event.bot.api, "call_action")):
        return None
    
    try:
        gid = event.get_group_id()
    except Exception:
        gid = None

    actions = [
        {"action": "get_file", "params": {"file_id": file_id}},
        {"action": "get_file", "params": {"file": file_id}},
    ]
    if gid:
        actions.append({"action": "get_group_file_url", "params": {"group_id": gid, "file_id": file_id}})

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
                    if fp.startswith("/") and len(fp) > 3 and fp[2] == ":": fp = fp[1:]
                    if os.path.exists(fp): return os.path.abspath(fp)
                if os.path.exists(f): return os.path.abspath(f)
        except Exception:
            continue
    return None

async def download_video_to_temp(url: str, size_mb_limit: int) -> Optional[str]:
    """下载媒体到临时文件。"""
    max_bytes = size_mb_limit * 1024 * 1024
    
    try:
        if aiohttp:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=30) as resp:
                    if resp.status != 200: 
                        logger.warning(f"[VideoFrameProcessor] 下载失败: HTTP {resp.status} (URL: {url})")
                        return None
                    
                    # 检查内容长度
                    cl = resp.headers.get("Content-Length")
                    if cl and cl.isdigit() and int(cl) > max_bytes: 
                        logger.warning(f"[VideoFrameProcessor] 下载终止: 文件过大 ({int(cl)/(1024*1024):.1f}MB > {size_mb_limit}MB)")
                        return None
                    
                    # 根据 Content-Type 决定后缀
                    content_type = resp.headers.get("Content-Type", "").lower()
                    if "image/gif" in content_type:
                        ext = ".gif"
                    elif "image/" in content_type:
                        ext = ".jpg"
                    elif "video/" in content_type:
                        ext = ".mp4"
                    else:
                        ext = ".mp4" # 默认
                    
                    # 创建临时文件
                    tmp = tempfile.NamedTemporaryFile(prefix="llm_media_", suffix=ext, delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    
                    total = 0
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            total += len(chunk)
                            if total > max_bytes: 
                                os.remove(tmp_path)
                                logger.warning(f"[VideoFrameProcessor] 下载终止: 实际下载数据超过限制")
                                return None
                            f.write(chunk)
                    return tmp_path
        else:
            logger.error("[VideoFrameProcessor] download_video_to_temp: aiohttp is not installed")
    except Exception as e:
        logger.error(f"[VideoFrameProcessor] 下载异常: {e} (URL: {url})")
    return None

def probe_duration_sec(ffmpeg_path: str, video_path: str) -> Optional[float]:
    """探测视频时长。"""
    # 优先使用与 ffmpeg 同目录的 ffprobe
    ffprobe_path = None
    if ffmpeg_path:
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if ffmpeg_dir:
            cand = os.path.join(ffmpeg_dir, "ffprobe.exe" if os.name == "nt" else "ffprobe")
            if os.path.exists(cand):
                ffprobe_path = cand
    
    if not ffprobe_path:
        ffprobe_path = shutil.which("ffprobe")
    
    if not ffprobe_path: return None
    
    cmd = [ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
    try:
        res = _safe_subprocess_run(cmd)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return None

async def sample_frames_equidistant(ffmpeg_path: str, video_path: str, duration_sec: float, count: int) -> List[str]:
    """等距抽帧。"""
    if not ffmpeg_path or not shutil.which(ffmpeg_path):
        ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path: return []

    out_dir = tempfile.mkdtemp(prefix="llm_frames_")
    frames = []
    loop = asyncio.get_running_loop()
    
    try:
        for i in range(1, count + 1):
            t = (i / (count + 1.0)) * duration_sec
            out_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            cmd = [ffmpeg_path, "-y", "-ss", f"{t:.3f}", "-i", video_path, "-frames:v", "1", "-qscale:v", "2", out_path]
            res = await loop.run_in_executor(None, lambda: _safe_subprocess_run(cmd))
            if res.returncode == 0 and os.path.exists(out_path):
                frames.append(out_path)
    except Exception:
        pass
    return frames

DEFAULT_FRAME_CAPTION_PROMPT = "请根据这张关键帧图片，用一句中文描述画面要点；少于25字。若无法判断，请回答‘未识别’。禁止输出政治有关内容。"


async def extract_forward_video_keyframes(
    event: AstrMessageEvent,
    video_sources: List[str],
    max_count: int,
    ffmpeg_path: str,
    max_mb: int,
    max_duration: int,
    timeout_sec: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    将合并转发中的视频源转换为少量关键帧图片（每个视频 1 张）。
    返回: (帧路径列表, 待清理路径列表, 本地视频路径列表)
    """
    frames = []
    cleanup_paths = []
    local_video_paths = []
    
    for src in video_sources[:max_count]:
        video_path = src
        is_temp_video = False
        
        # 1. 解析 Napcat file_id
        if not os.path.exists(src) and not src.startswith(("http://", "https://")):
            logger.info(f"video_parser: 尝试解析本地/NapCat文件 ID: {src}")
            resolved = await napcat_resolve_file_url(event, src)
            if resolved:
                logger.info(f"video_parser: 文件 ID 解析成功 -> {resolved}")
                video_path = resolved
            else:
                logger.warning(f"video_parser: 文件 ID 解析失败: {src}")
                continue
            
        # 2. 下载远程视频
        if video_path.startswith(("http://", "https://")):
            logger.info(f"video_parser: 正在下载并解析合并转发中的视频: {video_path}")
            try:
                downloaded = await asyncio.wait_for(
                    download_video_to_temp(video_path, max_mb),
                    timeout=timeout_sec
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
        
        # 安全限制：硬编码 120 分钟 (7200秒)
        safety_max_duration = 7200
        
        if duration is None or duration > safety_max_duration or duration <= 0:
            if is_temp_video and video_path in cleanup_paths:
                os.remove(video_path)
                cleanup_paths.remove(video_path)
            continue
            
        # 4. 抽 1 帧
        sampled = await sample_frames_equidistant(ffmpeg_path, video_path, duration, 1)
        if sampled:
            frames.append(sampled[0])
            cleanup_paths.append(sampled[0])
            local_video_paths.append(video_path)
                
    return frames, cleanup_paths, local_video_paths

async def extract_audio_wav(ffmpeg_path: str, video_path: str) -> Optional[str]:
    """从视频提取音频保存为 WAV 格式。"""
    if not os.path.exists(video_path):
        return None
    tmp = tempfile.NamedTemporaryFile(prefix="video_audio_", suffix=".wav", delete=False)
    out_path = tmp.name
    tmp.close()

    # ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 -f wav output.wav
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
    process_timeout: int = 30
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """
    处理视频源，返回抽取的帧路径列表、待清理的路径列表、最终使用的视频本地路径，以及处理状态（失败原因）。
    """
    frames = []
    cleanup_paths = []
    final_video_path = None
    status = None # 记录失败原因
    
    start_time = time.time()
    
    for src in video_sources:
        # 检查总耗时
        if time.time() - start_time > process_timeout:
            logger.warning("video_parser: processing timeout")
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
                # 注意：download_video_to_temp 内部已经处理了大小限制
                downloaded = await asyncio.wait_for(
                    download_video_to_temp(video_path, max_mb),
                    timeout=process_timeout - (time.time() - start_time)
                )
                if downloaded:
                    video_path = downloaded
                    is_temp_video = True
                    cleanup_paths.append(video_path)
                else:
                    # 如果返回 None，很可能是因为文件过大
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
        
        # 安全限制：硬编码 120 分钟 (7200秒)
        safety_max_duration = 7200
        
        if duration is None or duration > safety_max_duration or duration <= 0:
            if is_temp_video and video_path in cleanup_paths:
                try: os.remove(video_path)
                except Exception as e:
                    logger.debug(f"video_parser: failed to remove temp video {video_path}: {e}")
                cleanup_paths.remove(video_path)
            status = "too_long"
            continue
            
        # 4. 抽帧
        sampled = await sample_frames_equidistant(ffmpeg_path, video_path, duration, sample_count)
        if sampled:
            frames.extend(sampled)
            cleanup_paths.extend(sampled)
            final_video_path = video_path
            status = "success"
            break
        else:
            status = "sample_failed"
                
    return frames, cleanup_paths, final_video_path, status

class VideoFrameProcessor:
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
                    logger.debug(f"[VideoFrameProcessor] 视频总结缓存命中: {video_key[:50]}...")
                    return item["summary"]
                else:
                    del cls._summary_cache[video_key]
        return None

    @classmethod
    async def set_cached_summary(cls, video_key: str, summary: str, ttl: int = 3600):
        """设置视频总结缓存，默认有效期 1 小时"""
        async with cls._cache_lock:
            # 简单的容量清理：如果缓存超过 100 条，清空全部
            if len(cls._summary_cache) > 100:
                cls._summary_cache.clear()
            
            cls._summary_cache[video_key] = {
                "summary": summary,
                "expire": time.time() + ttl
            }
            logger.debug(f"[VideoFrameProcessor] 视频总结已缓存: {video_key}")

    async def process_long_video(self, req: ProviderRequest, video_path: str, duration: float, sender_name: str = None, msg_id: str = None) -> bool:
        """【场景：视频】多帧抽取 + ASR → 帧聚合汇总"""
        logger.debug(f"[VideoFrameProcessor] 开始处理视频: {video_path}, msg_id: {msg_id}")
        try:
            # 1. 尝试从缓存获取
            cached_summary = None
            if msg_id:
                cached_summary = await self.get_cached_summary(msg_id)
                if cached_summary:
                    self._inject_summary(req, cached_summary, "视频转述(缓存复用)", sender_name=sender_name)
                    return True

            frames, cleanup_paths, local_video_path, status = await self._extract_video_frames(video_path, duration)
            
            if not frames:
                if status == "too_large":
                    self._inject_summary(req, "用户发送/引用了一个视频消息，但是文件太大了，我懒得看（已跳过解析）。", "系统提示", sender_name=sender_name)
                    return True
                elif status == "too_long":
                    self._inject_summary(req, "用户发送/引用了一个视频消息，但是时间太长了，我没耐心看（已跳过解析）。", "系统提示", sender_name=sender_name)
                    return True
                return False
            
            # 注册清理路径
            if cleanup_paths:
                req._cleanup_paths.extend(cleanup_paths)
            
            # 步骤：语音转录（可选）
            asr_text = None
            if self._get_cfg("video_asr_enable", True) and local_video_path:
                asr_text = await self._extract_and_transcribe_audio(local_video_path)
            
            # 步骤：帧聚合汇总
            summary = await self._aggregate_frames_helper(
                frames, 
                len(frames), 
                duration,
                asr_text=asr_text,
                max_summary_length=100
            )
            
            if summary:
                if msg_id:
                    await self.set_cached_summary(msg_id, summary)
                else:
                    logger.warning("[VideoFrameProcessor] msg_id 为空，跳过缓存写入")
                
                self._inject_summary(req, summary, "视频转述", sender_name=sender_name)
                logger.info(f"[VideoFrameProcessor] 视频总结成功并注入，来源: {sender_name or '当前消息'}")
                # 注册清理
                for frame in frames:
                    req._cleanup_paths = req._cleanup_paths or []
                    req._cleanup_paths.append(frame)
                return True
            else:
                logger.warning("[VideoFrameProcessor] 视频转述生成失败，回退到首帧")
                req.image_urls.append(frames[0])
                for frame in frames:
                    req._cleanup_paths = req._cleanup_paths or []
                    req._cleanup_paths.append(frame)
                return True
        except Exception as e:  
            logger.warning(f"[VideoFrameProcessor] 视频处理失败: {e}")
            return False
    
    async def process_gif(self, req: ProviderRequest, gif_path: str, sender_name: str = None, msg_id: str = None) -> bool:
        """【场景：GIF 动图】强制抽帧 → 帧聚合汇总"""
        logger.debug(f"[VideoFrameProcessor] 开始处理 GIF: {gif_path}, msg_id: {msg_id}")
        try:
            # 1. 尝试从缓存获取
            cached_summary = None
            if msg_id:
                cached_summary = await self.get_cached_summary(msg_id)
                if cached_summary:
                    self._inject_summary(req, cached_summary, "内容摘要(缓存复用)", sender_name=sender_name)
                    return True

            logger.info(f"[VideoFrameProcessor] GIF 处理: {gif_path}")
            
            ffmpeg_path = self._get_cfg("ffmpeg_path")
            duration = await self._probe_duration(gif_path)
            if not duration or duration <= 0:
                duration = 1.0
            
            # GIF 不再尝试直接解析，统一走抽帧
            
            # GIF 固定参数：1秒/帧，最多 10 帧。使用 math.ceil 确保 1.1s -> 2 帧
            sample_count = max(1, min(math.ceil(duration), 10))
            
            logger.info(f"[VideoFrameProcessor] GIF 抽帧: 时长 {duration:.2f}s, 抽帧数 {sample_count}")
            
            frame_paths = await sample_frames_equidistant(
                ffmpeg_path,
                gif_path,
                duration,
                sample_count
            )
            
            if not frame_paths:
                logger.warning(f"[VideoFrameProcessor] GIF 解析未产生有效帧: {gif_path}")
                return False
            
            frames = [str(p) for p in frame_paths]
            frame_count = len(frames)
            
            # GIF 帧聚合
            summary = await self._aggregate_frames_helper(
                frames,
                frame_count,
                duration,
                asr_text=None,
                max_summary_length=50
            )
            
            if summary:
                if msg_id:
                    await self.set_cached_summary(msg_id, summary)
                else:
                    logger.warning("[VideoFrameProcessor] GIF msg_id 为空，跳过缓存写入")
                
                self._inject_summary(req, summary, "内容摘要", sender_name=sender_name)
                # 只保留最后一帧
                req.image_urls = [frames[-1]]
                logger.info("[VideoFrameProcessor] GIF 总结成功")
            else:
                logger.warning("[VideoFrameProcessor] GIF 总结为空，回退到首帧")
                req.image_urls = [frames[0]]
            
            # 注册清理
            for frame in frames:
                req._cleanup_paths = req._cleanup_paths or []
                req._cleanup_paths.append(frame)
            return True
            
        except Exception as e: 
            logger.error(f"[VideoFrameProcessor] GIF 处理异常: {e}", exc_info=True)
            return False
    
    # ==================== 私有辅助方法 ====================
    
    async def _probe_duration(self, media_path: str) -> float:
        """探测媒体时长"""
        try:
            duration = await asyncio.to_thread(
                probe_duration_sec, 
                self._get_cfg("ffmpeg_path", ""), 
                media_path
            ) or 0
            return duration
        except Exception as e:
            logger.debug(f"[VideoFrameProcessor] 时长探测失败: path={media_path}, err={e}")
            return 0
    

    async def _extract_video_frames(self, video_path: str, duration: float) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
        """多帧抽取（读取配置参数）"""
        interval = self._get_cfg("video_frame_interval_sec", 12)
        max_frame_count = self._get_cfg("video_max_frame_count", 10)
        
        sample_count = 1
        if duration > 0:
            # 动态计算抽帧数：取 (时长/间隔) 和 (抽帧上限) 的较小值
            ideal_count = math.ceil(duration / interval) if interval > 0 else 1
            sample_count = max(1, min(ideal_count, max_frame_count))
            
            # 如果实际抽帧数因为达到上限而被压缩，日志记录一下
            if ideal_count > max_frame_count:
                actual_interval = duration / sample_count
                logger.info(f"[VideoFrameProcessor] 视频时长 {duration:.1f}s 超过间隔覆盖范围，调整抽帧间隔: {interval}s -> {actual_interval:.1f}s (上限 {max_frame_count} 帧)")
        else:
            sample_count = self._get_cfg("video_sample_count", 3)
        
        logger.info(f"[VideoFrameProcessor] 视频多帧抽取: {sample_count} 帧")
        
        try:
            frames, cleanup_paths, local_video_path, status = await prepare_video_context(
                self.event,
                [video_path],
                max_mb=self._get_cfg("video_max_size_mb", 50),
                max_duration=7200, # 硬编码安全限制 120 分钟
                sample_count=sample_count,
                ffmpeg_path=self._get_cfg("ffmpeg_path", ""),
                process_timeout=30
            )
            
            return frames or [], cleanup_paths or [], local_video_path, status
        except Exception as e:
            logger.warning(f"[VideoFrameProcessor] 帧抽取失败: {e}")
            return [], [], None, "error"
    
    async def _extract_and_transcribe_audio(self, video_path: str) -> Optional[str]:
        """ASR：提取音频并转录"""
        try:
            logger.debug(f"[VideoFrameProcessor] ASR: 正在从视频中提取音频...")
            ffmpeg_path = self._get_cfg("ffmpeg_path", "")
            wav_path = await extract_audio_wav(ffmpeg_path, video_path)
            
            if not wav_path or not os.path.exists(wav_path):
                logger.warning("[VideoFrameProcessor] ASR: 提取音频 WAV 失败")
                return None
            
            stt = self._get_stt_provider()
            if not stt:
                logger.warning("[VideoFrameProcessor] ASR: 无可用 STT Provider")
                try:
                    os.remove(wav_path)
                except Exception as e:
                    logger.debug(f"[VideoFrameProcessor] ASR wav 清理失败(无 STT): path={wav_path}, err={e}")
                return None
            
            asr_text = None
            try:
                logger.debug(f"[VideoFrameProcessor] ASR: 正在请求转录...")
                if hasattr(stt, "get_text"):
                    asr_text = await stt.get_text(wav_path)
                elif hasattr(stt, "speech_to_text"):
                    res = await stt.speech_to_text(wav_path)
                    if isinstance(res, dict):
                        asr_text = res.get("text", "")
                    elif hasattr(res, "text"):
                        asr_text = res.text
                    else:
                        asr_text = str(res)
                
                if asr_text:
                    logger.info(f"[VideoFrameProcessor] ASR 成功: {asr_text[:100]}...")
                else:
                    logger.info(f"[VideoFrameProcessor] ASR 成功，但未识别到文字内容")
            except Exception as e:
                logger.warning(f"[VideoFrameProcessor] ASR 调用失败: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except Exception as e:
                    logger.debug(f"[VideoFrameProcessor] ASR wav 清理失败: path={wav_path}, err={e}")
            
            return asr_text
        except Exception as e:  
            logger.warning(f"[VideoFrameProcessor] ASR 处理失败: {e}")
            return None
    
    async def _aggregate_frames_helper(self, frames: List[str], frame_count: int, duration: float, asr_text: Optional[str] = None, max_summary_length: int = 100) -> Optional[str]:
        """【公用】帧聚合汇总"""
        if not frames:  
            return None
        
        # 强制字数限制
        max_summary_length = max_summary_length or 100
        
        logger.info(f"[帧聚合汇总] 开始处理 {len(frames)} 帧图片...")
        
        # 步骤 1：逐帧识图
        frame_descriptions = []
        provider = self._get_vision_provider()
        
        if not provider:
            logger.warning("[帧聚合汇总] 无可用 Vision Provider")
            return None
        
        for idx, frame_path in enumerate(frames, 1):
            try:
                start_t = time.time()
                # 获取 provider 名称用于日志
                p_name = getattr(provider, "name", "unknown") or getattr(provider, "id", "unknown")
                logger.info(f"[帧聚合汇总] 正在请求第 {idx}/{frame_count} 帧 ...")
                
                response = await provider.text_chat(
                    prompt=f"简要描述这一帧（第 {idx}/{frame_count} 帧）的内容，重点关注动作、表情和关键物体。",
                    system_prompt="你是一个视频分析助手。请用 1-2 句话简洁描述图片内容。",
                    image_urls=[frame_path],
                    context=[]
                )
                
                desc = self._extract_completion_text(response)
                cost = time.time() - start_t
                if desc:  
                    frame_descriptions.append(f"[第 {idx} 帧] {desc}")
                    logger.info(f"[帧聚合汇总] 第 {idx}/{len(frames)} 帧解析成功 (耗时: {cost:.2f}s). 响应: {desc}")
                else:
                    logger.warning(f"[帧聚合汇总] 第 {idx}/{len(frames)} 帧解析响应为空")
            except Exception as e:
                logger.warning(f"[帧聚合汇总] 第 {idx}/{len(frames)} 帧识图异常: {e}")
                frame_descriptions.append(f"[第 {idx} 帧] [处理失败]")
        
        if not frame_descriptions:
            return None
        
        # 步骤 2：合并帧描述和 ASR
        comprehensive_context = "\n".join(frame_descriptions)
        
        if asr_text:
            comprehensive_context += f"\n\n[视频语音转写]\n{asr_text}"
            logger.info("[帧聚合汇总] 已融合 ASR 内容")
        
        # 步骤 3：LLM 汇总
        try:
            llm_provider = self.context.get_using_provider(umo=self.event.unified_msg_origin)
            if not llm_provider:
                logger.warning("[帧聚合汇总] 无可用 LLM Provider")
                return None
            
            llm_name = getattr(llm_provider, "name", "unknown") or getattr(llm_provider, "id", "unknown")
            logger.info(f"[帧聚合汇总] 正在请求汇总摘要...")
            
            summary_prompt = (
                f"你是一个媒体分析助手。请根据以下提供的逐帧视觉描述和语音转写（如有），"
                f"生成一份简洁连贯的汇总摘要。要求：\n"
                f"1. 重点概括视频/动图的核心内容、动作变化和视觉亮点。\n"
                f"2. 严格控制字数在 {max_summary_length} 字以内。\n"
                f"3. 直接输出摘要内容，不要有任何前缀或解释。\n\n"
                f"--- 原始素材数据开始 ---\n"
                f"{comprehensive_context}\n"
                f"--- 原始素材数据结束 ---\n\n"
                f"请生成汇总摘要："
            )
            
            response = await llm_provider.text_chat(
                prompt=summary_prompt,
                system_prompt="你是一个专业的媒体摘要生成器。请直接输出摘要内容。",
                image_urls=[], # 汇总阶段不传递图片，仅根据文字描述生成
                context=[]
            )
            
            summary = self._extract_completion_text(response)
            
            if summary:
                # 步骤 4：长度控制
                if len(summary) > max_summary_length:
                    logger.info(f"[帧聚合汇总] 摘要长度 {len(summary)} 超过限制 {max_summary_length}，截断处理")
                    summary = summary[:max_summary_length].rsplit('。', 1)[0] + "。"
                
                logger.info(f"[帧聚合汇总] 汇总成功 (耗时: {time.time() - start_t:.2f}s, 字数: {len(summary)}): {summary}")
                return summary
            else:
                logger.warning(f"[帧聚合汇总] 汇总摘要响应为空")
        except Exception as e:
            logger.warning(f"[帧聚合汇总] LLM 汇总失败: {e}")
        
        return None
    
    def _inject_summary(self, req: ProviderRequest, summary: str, label: str, sender_name: str = None):
        """注入总结"""
        user_question = req.prompt.strip()
        sender_prefix = f"该媒体消息由 {sender_name} 发送/提供。\n" if sender_name else ""
        context_prompt = (
            f"\n\n以下是系统为你分析的{label}，请结合此{label}来响应用户的要求。信息如下：\n"
            f"--- 注入内容开始 ---\n"
            f"{sender_prefix}[{label}] {summary}\n"
            f"--- 注入内容结束 ---"
        )
        req.prompt = user_question + context_prompt
        logger.debug(f"[VideoFrameProcessor] 成功注入{label}")
    
    def _find_provider(self, provider_id: str):
        """通用方法：从所有 Provider（包含 LLM 和 STT）中查找匹配的 ID/Name"""
        return find_provider(self.context, provider_id)

    def _get_vision_provider(self):
        """获取 Vision Provider"""
        provider_id = self._get_cfg("image_provider_id")
        p = self._find_provider(provider_id)
        if p:
            logger.debug(f"[VideoFrameProcessor] 使用指定 Vision Provider: {provider_id}")
            return p
        
        if provider_id:
            logger.warning(f"[VideoFrameProcessor] 指定的 Vision Provider {provider_id} 未找到")

        # 自动选择（当前正在使用的）
        try:
            default_p = self.context.get_using_provider(umo=self.event.unified_msg_origin)
            if default_p:
                logger.debug(f"[VideoFrameProcessor] 使用当前会话 Provider 进行识图")
                return default_p
        except Exception as e:
            logger.debug(f"[VideoFrameProcessor] 获取会话 Vision Provider 失败: err={e}")
            
        return None

    def _get_stt_provider(self):
        """获取 STT Provider"""
        asr_pid = self._get_cfg("asr_provider_id")
        p = self._find_provider(asr_pid)
        if p:
            logger.debug(f"[VideoFrameProcessor] 成功匹配到指定的 STT Provider: {asr_pid}")
            return p
        
        if asr_pid:
            logger.warning(f"[VideoFrameProcessor] 未找到指定的 STT Provider: {asr_pid}")
        
        try:
            stt_p = self.context.get_using_stt_provider(umo=self.event.unified_msg_origin)
            return stt_p
        except Exception as e:
            logger.debug(f"[VideoFrameProcessor] 获取会话 STT Provider 失败: err={e}")
        
        return None
    
    def _extract_completion_text(self, response) -> str:
        """提取响应文本"""
        if hasattr(response, "completion_text"):
            return response.completion_text.strip()
        elif isinstance(response, dict):
            return response.get("completion_text", "").strip()
        return ""
