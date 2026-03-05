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


async def get_history_messages(
    context: Any,
    event: AstrMessageEvent,
    role: str = "assistant",
    count: int | None = 0,
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
        contexts = [
            record["content"]
            for record in history
            if record.get("role") == role and record.get("content")
        ]
        return contexts[-count:] if count else contexts
    except Exception as e:
        logger.error(f"获取历史消息失败：{e}")
        return []
