from __future__ import annotations
import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Any, Dict, Tuple
from urllib.parse import urlparse, unquote

try:
    import aiohttp
except ImportError:
    aiohttp = None

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp

import json

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
    """下载视频到临时文件。"""
    max_bytes = size_mb_limit * 1024 * 1024
    ext = os.path.splitext(urlparse(url).path)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(prefix="llm_video_", suffix=ext, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        if aiohttp:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=30) as resp:
                    if resp.status != 200: return None
                    cl = resp.headers.get("Content-Length")
                    if cl and cl.isdigit() and int(cl) > max_bytes: return None
                    total = 0
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            total += len(chunk)
                            if total > max_bytes: return None
                            f.write(chunk)
            return tmp_path
    except Exception:
        if os.path.exists(tmp_path): os.remove(tmp_path)
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

def provider_supports_image(provider: Any) -> bool:
    """尽力判断 Provider 是否支持图片/多模态。"""
    try:
        mods = getattr(provider, "modalities", None)
        if isinstance(mods, (list, tuple)):
            ml = [str(m).lower() for m in mods]
            if any(k in ml for k in ["image", "vision", "multimodal", "vl", "picture"]):
                return True
    except (AttributeError, TypeError):
        pass
    for attr in ("config", "model_config", "model"):
        try:
            val = getattr(provider, attr, None)
            text = str(val)
            lt = text.lower()
            if any(k in lt for k in ["image", "vision", "multimodal", "vl", "gpt-4o", "gemini", "minicpm-v"]):
                return True
        except (AttributeError, TypeError, ValueError):
            pass
    return False

def select_vision_provider(context: Any, session_provider: Any, image_provider_id: Optional[str] = None) -> Any:
    """选择一个尽可能支持图片的 Provider。"""
    # 1. 尝试使用 image_provider_id 指定的 provider
    if image_provider_id:
        try:
            all_p = context.get_all_providers()
            for p in all_p:
                pid = getattr(p, "id", None) or getattr(p, "provider_id", None)
                if pid == image_provider_id:
                    return p
        except Exception:
            pass

    # 2. 检查 session_provider 是否支持图片
    if session_provider and provider_supports_image(session_provider):
        return session_provider

    # 3. 遍历所有 provider 找到第一个支持图片的
    try:
        all_p = context.get_all_providers()
        for p in all_p:
            if p is session_provider: continue
            if provider_supports_image(p):
                return p
    except Exception:
        pass
        
    return session_provider

async def get_video_summary(
    context: Any,
    event: AstrMessageEvent,
    frames: List[str],
    duration: float,
    video_name: str,
    image_provider_id: Optional[str] = None,
    asr_text: Optional[str] = None
) -> Optional[str]:
    """
    逐帧描述 -> 汇总文本。
    """
    if not frames:
        return None
        
    # 组装并调用 LLM（多次调用：逐帧→最终汇总）
    try:
        session_provider = context.get_using_provider(umo=event.unified_msg_origin)
    except Exception as e:
        logger.error(f"video_parser: get provider failed: {e}")
        session_provider = None
        
    provider = select_vision_provider(context, session_provider, image_provider_id)
    
    if not provider:
        logger.error("video_parser: no active provider found")
        return None
    
    # 获取 provider id 用于日志
    pid = (
        getattr(provider, "id", None) 
        or getattr(provider, "provider_id", None) 
        or provider.__class__.__name__
    )
    logger.info(f"video_parser: using provider {pid} for video summary")
        
    # 2. 逐帧描述 (Captions) - 处理前 n-1 帧
    captions = []
    system_prompt = "你是一个视频分析助手。请根据提供的视频帧画面，进行客观、准确的描述。"
    
    # 如果只有一帧，则跳过逐帧描述，直接进入汇总阶段
    if len(frames) > 1:
        for idx, frame_path in enumerate(frames[:-1], start=1):
            logger.info(f"video_parser: caption frame {idx}/{len(frames)-1}")
            
            try:
                response = await provider.text_chat(
                    prompt=DEFAULT_FRAME_CAPTION_PROMPT,
                    system_prompt=system_prompt,
                    image_urls=[frame_path],
                    context=[]
                )
                
                # 兼容不同版本的返回结果
                completion_text = ""
                if hasattr(response, "completion_text"):
                    completion_text = response.completion_text
                elif isinstance(response, dict):
                    completion_text = response.get("completion_text", "")
                
                if completion_text:
                    captions.append(completion_text.strip())
                    logger.info(f"video_parser: caption len={len(completion_text)}")
                else:
                    captions.append("未识别")
            except Exception as e:
                logger.warning(f"video_parser: caption failed on frame {idx}: {e}")
                captions.append("未识别")

    # 3. 构造汇总提示词 (无论几帧都要汇总)
    meta_items = []
    if video_name:
        meta_items.append(f"视频: {video_name}")
    if duration > 0:
        meta_items.append(f"时长: {int(duration)}s")
    meta_items.append(f"关键帧: {len(frames)} 张")
    
    if asr_text:
        meta_items.append(f"ASR语音转写: {asr_text}")
        
    meta_block = "\n".join(meta_items)

    caps_block = "\n".join(
        [f"- {c.strip()}" for c in captions if isinstance(c, str) and c.strip()]
    )
    
    final_summary_prompt = (
        "请根据以下关键帧描述与最后一张关键帧图片，总结整段视频的主要内容（中文，不超过100字）。"
        "仅依据已给信息，信息不足请说明‘无法判断’，不要编造未出现的内容。\n"
        f"{meta_block}\n关键帧描述：\n{caps_block}"
    )
    
    try:
        final_response = await provider.text_chat(
            prompt=final_summary_prompt,
            system_prompt=system_prompt,
            image_urls=[frames[-1]],
            context=[]
        )
        
        completion_text = ""
        if hasattr(final_response, "completion_text"):
            completion_text = final_response.completion_text
        elif isinstance(final_response, dict):
            completion_text = final_response.get("completion_text", "")

        if completion_text:
            return completion_text.strip()
    except Exception as e:
        logger.error(f"video_parser: summary failed: {e}")
        
    return None

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
        duration = probe_duration_sec(ffmpeg_path, video_path)
        if duration is None or duration > max_duration or duration <= 0:
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
) -> Tuple[List[str], List[str], Optional[str]]:
    """
    处理视频源，返回抽取的帧路径列表、待清理的路径列表、以及最终使用的视频本地路径。
    """
    frames = []
    cleanup_paths = []
    final_video_path = None
    
    start_time = time.time()
    
    for src in video_sources:
        # 检查总耗时
        if time.time() - start_time > process_timeout:
            logger.warning("video_parser: processing timeout")
            break

        video_path = src
        is_temp_video = False
        
        # 1. 解析 Napcat file_id
        if not os.path.exists(src) and not src.startswith(("http://", "https://")):
            resolved = await napcat_resolve_file_url(event, src)
            if resolved: video_path = resolved
            
        # 2. 下载远程视频
        if video_path.startswith(("http://", "https://")):
            downloaded = await download_video_to_temp(video_path, max_mb)
            if downloaded:
                video_path = downloaded
                is_temp_video = True
                cleanup_paths.append(video_path)
            else:
                continue
                
        if not os.path.exists(video_path):
            continue
            
        # 3. 探测时长
        duration = probe_duration_sec(ffmpeg_path, video_path)
        if duration is None or duration > max_duration or duration <= 0:
            if is_temp_video and video_path in cleanup_paths:
                os.remove(video_path)
                cleanup_paths.remove(video_path)
            continue
            
        # 4. 抽帧
        sampled = await sample_frames_equidistant(ffmpeg_path, video_path, duration, sample_count)
        if sampled:
            frames.extend(sampled)
            cleanup_paths.extend(sampled)
            final_video_path = video_path
            # 只处理第一个成功的视频
            break
            
    return frames, cleanup_paths, final_video_path
