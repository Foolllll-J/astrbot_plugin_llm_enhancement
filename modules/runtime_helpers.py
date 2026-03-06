import asyncio
import json
import os
import shutil
from typing import Any, Callable, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .provider_utils import find_provider


def resolve_provider(context: Any, provider_id: str):
    """Find provider by configured id or name among available providers."""
    return find_provider(context, provider_id)


def get_llm_provider(
    context: Any,
    event: Optional[AstrMessageEvent] = None,
    umo: Optional[str] = None,
):
    """Get current session LLM provider."""
    try:
        using_umo = umo or (getattr(event, "unified_msg_origin", None) if event is not None else None)
        if using_umo:
            return context.get_using_provider(umo=using_umo)
        return context.get_using_provider()
    except Exception:
        return None


def get_stt_provider(
    context: Any,
    get_cfg: Callable[[str, Any], Any],
    event: Optional[AstrMessageEvent] = None,
    umo: Optional[str] = None,
):
    """Get configured STT provider with fallback to current session provider."""
    asr_pid = get_cfg("asr_provider_id")
    p = resolve_provider(context, asr_pid)
    if p:
        logger.debug(f"[LLMEnhancement] 成功匹配到指定 STT Provider: {asr_pid}")
        return p

    if asr_pid:
        logger.warning(f"[LLMEnhancement] 未找到指定 STT Provider: {asr_pid}")

    try:
        using_umo = umo or (getattr(event, "unified_msg_origin", None) if event is not None else None)
        if using_umo:
            return context.get_using_stt_provider(umo=using_umo)
        return context.get_using_stt_provider()
    except Exception:
        return None


def get_vision_provider(
    context: Any,
    get_cfg: Callable[[str, Any], Any],
    event: Optional[AstrMessageEvent] = None,
    umo: Optional[str] = None,
):
    """Get configured vision provider with fallback to current LLM provider."""
    image_pid = get_cfg("image_provider_id")
    p = resolve_provider(context, image_pid)
    if p:
        logger.debug(f"[LLMEnhancement] 成功匹配到指定 Vision Provider: {image_pid}")
        return p

    if image_pid:
        logger.warning(f"[LLMEnhancement] 未找到指定 Vision Provider: {image_pid}，回退到会话 Provider")

    return get_llm_provider(context=context, event=event, umo=umo)


async def cleanup_paths_later(cleanup_paths: list[str], delay_sec: int = 120):
    """Cleanup temporary paths after delay, fire-and-forget."""
    if not cleanup_paths:
        return

    snapshot_paths = [p for p in dict.fromkeys(cleanup_paths) if p]
    if not snapshot_paths:
        return

    async def final_cleanup(paths: list[str]):
        await asyncio.sleep(delay_sec)
        for p in paths:
            try:
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                logger.debug(f"[LLMEnhancement] cleanup path 失败: {p}, err={e}")

    asyncio.create_task(final_cleanup(snapshot_paths))


def _history_segment_to_text(seg: Any) -> str:
    if isinstance(seg, str):
        return seg
    if not isinstance(seg, dict):
        return ""

    seg_type = str(seg.get("type") or "").strip().lower()
    data = seg.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    if seg_type == "text":
        return str(data.get("text") or seg.get("text") or "")
    if seg_type == "at":
        qq = str(data.get("qq") or seg.get("qq") or "").strip()
        return "@全体成员" if qq.lower() == "all" else (f"@{qq}" if qq else "@")
    if seg_type == "reply":
        rid = str(data.get("id") or seg.get("id") or "").strip()
        return f"[回复:{rid}]" if rid else "[回复]"
    if seg_type == "image":
        return "[图片]"
    if seg_type == "video":
        return "[视频]"
    if seg_type == "record":
        return "[语音]"
    if seg_type == "file":
        name = str(data.get("name") or seg.get("name") or "").strip()
        return f"[文件:{name}]" if name else "[文件]"
    if seg_type == "face":
        return "[表情]"
    if seg_type == "json":
        return "[JSON]"
    if seg_type == "xml":
        return "[XML]"
    if seg_type == "forward":
        return "[合并转发]"
    return ""


def _history_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(_history_segment_to_text(seg) for seg in content).strip()
    if isinstance(content, dict):
        return _history_segment_to_text(content).strip()
    return str(content or "").strip()


async def get_history_messages(
    context: Any,
    event: AstrMessageEvent,
    role: str | None = "assistant",
    count: int | None = 0,
    with_role_prefix: bool = False,
) -> list[str]:
    """Read current conversation history contents by role."""
    try:
        umo = event.unified_msg_origin
        curr_cid = await context.conversation_manager.get_curr_conversation_id(umo)
        if not curr_cid:
            return []
        conversation = await context.conversation_manager.get_conversation(umo, curr_cid)
        if not conversation:
            return []

        history = json.loads(conversation.history or "[]")
        role_filter = str(role or "").strip().lower()
        allow_all_roles = role is None or role_filter in {"", "*", "all"}

        contexts: list[str] = []
        for record in history:
            rec_role = str(record.get("role") or "").strip().lower()
            if (not allow_all_roles) and rec_role != role_filter:
                continue

            text = _history_content_to_text(record.get("content"))
            if not text:
                continue

            if with_role_prefix:
                role_label = rec_role or "unknown"
                contexts.append(f"{role_label}: {text}")
            else:
                contexts.append(text)

        return contexts[-count:] if count else contexts
    except Exception as e:
        logger.error(f"获取历史消息失败：{e}")
        return []
