from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
from contextlib import asynccontextmanager
from typing import Any

from astrbot.api import logger
from astrbot.core.utils.session_lock import session_lock_manager


_ctx_current_event: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "_llme_group_concurrency_event", default=None
)


def _get_runner_event(runner: Any) -> Any:
    return getattr(
        getattr(getattr(runner, "run_context", None), "context", None), "event", None
    )


def _fingerprint(msg: Any) -> str:
    if isinstance(msg, dict):
        return json.dumps(msg, ensure_ascii=False, sort_keys=True, default=str)
    return str(msg)


def _format_sender_key(umo: str, sender_id: str) -> str:
    return f"{umo}#sender:{sender_id}"


def _sender_key_from_event(event: Any, umo: str) -> str:
    sid = getattr(event, "get_sender_id", None)
    gid = getattr(event, "get_group_id", None)
    if callable(sid) and callable(gid):
        try:
            s = sid()
            g = gid()
            if g and s:
                return _format_sender_key(umo, str(s))
        except Exception:
            pass
    return umo


class GroupConcurrencyModule:
    """群聊按用户并发模块。通过猴子补丁修改 AstrBot 内部行为。"""

    def __init__(self) -> None:
        self._installed = False
        self._patch_state: dict[str, Any] = {}

    def install(self) -> bool:
        if self._installed:
            return True
        self._patch_state.clear()
        patches = [
            self._install_process_patch,
            self._install_lock_patch,
            self._install_update_conversation_patch,
            self._install_follow_up_patch,
        ]
        for patch in patches:
            if not patch():
                self._restore_all()
                logger.warning("[LLMEnhancement][GroupConcurrency] 补丁安装失败，已回滚。")
                return False
        self._installed = True
        logger.info("[LLMEnhancement][GroupConcurrency] 群聊按用户并发功能已启用。")
        return True

    def terminate(self) -> None:
        if self._installed:
            self._restore_all()
        self._installed = False

    def _restore_all(self) -> None:
        for target, original in self._patch_state.items():
            parts = target.split(".")
            if len(parts) == 3:
                _, cls_name, attr = parts
                try:
                    if cls_name == "ConversationManager":
                        from astrbot.core.conversation_mgr import ConversationManager
                        setattr(ConversationManager, attr, original)
                    elif cls_name == "InternalAgentSubStage":
                        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
                            InternalAgentSubStage,
                        )
                        setattr(InternalAgentSubStage, attr, original)
                except Exception:
                    pass
            elif len(parts) == 2:
                if parts[0] == "session_lock_manager":
                    setattr(session_lock_manager, parts[1], original)
                elif parts[0] in ("follow_up", "internal"):
                    try:
                        mod_path = (
                            "astrbot.core.pipeline.process_stage.follow_up"
                            if parts[0] == "follow_up"
                            else "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
                        )
                        mod = __import__(mod_path, fromlist=[""])
                        setattr(mod, parts[1], original)
                    except Exception:
                        pass
        self._patch_state.clear()

    def _save(self, key: str, original: Any) -> None:
        if key not in self._patch_state:
            self._patch_state[key] = original

    def _install_process_patch(self) -> bool:
        try:
            from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
                InternalAgentSubStage,
            )
        except Exception:
            return False
        original = getattr(InternalAgentSubStage, "process", None)
        if not callable(original):
            return False
        self._save("InternalAgentSubStage.process", original)

        async def patched_process(stage_self: Any, event: Any, *args: Any, **kwargs: Any):
            token = _ctx_current_event.set(event)
            try:
                gen = original(stage_self, event, *args, **kwargs)
                if inspect.isasyncgen(gen):
                    async for item in gen:
                        yield item
                elif inspect.isawaitable(gen):
                    await gen
            finally:
                _ctx_current_event.reset(token)

        InternalAgentSubStage.process = patched_process
        return True

    def _install_lock_patch(self) -> bool:
        original = getattr(session_lock_manager, "acquire_lock", None)
        if not callable(original):
            return False
        self._save("session_lock_manager.acquire_lock", original)

        @asynccontextmanager
        async def patched_acquire_lock(session_id: str):
            event = _ctx_current_event.get()
            if event is not None:
                key = _sender_key_from_event(event, str(session_id))
            else:
                key = str(session_id)
            async with original(key):
                yield

        session_lock_manager.acquire_lock = patched_acquire_lock
        return True

    def _install_update_conversation_patch(self) -> bool:
        try:
            from astrbot.core.conversation_mgr import ConversationManager
        except Exception:
            return False
        original = getattr(ConversationManager, "update_conversation", None)
        if not callable(original):
            return False
        self._save("ConversationManager.update_conversation", original)

        async def patched_update(
            mgr_self: Any,
            unified_msg_origin: str,
            conversation_id: str | None = None,
            history: list[dict] | None = None,
            **kwargs: Any,
        ):
            if history and conversation_id:
                try:
                    latest = await mgr_self.db.get_conversation_by_id(conversation_id)
                    if latest and latest.content and isinstance(latest.content, list):
                        latest_hist: list[dict] = latest.content
                        common = 0
                        for a, b in zip(latest_hist, history):
                            if _fingerprint(a) == _fingerprint(b):
                                common += 1
                            else:
                                break
                        if common > 0 and len(latest_hist) > common:
                            history = latest_hist + history[common:]
                except Exception:
                    pass
            return await original(
                mgr_self,
                unified_msg_origin,
                conversation_id,
                history=history,
                **kwargs,
            )

        ConversationManager.update_conversation = patched_update
        return True

    def _install_follow_up_patch(self) -> bool:
        try:
            from astrbot.core.pipeline.process_stage import follow_up as _fu
            from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal as _int
        except Exception:
            return False

        orig_register = getattr(_fu, "register_active_runner", None)
        orig_unregister = getattr(_fu, "unregister_active_runner", None)
        orig_try_capture = getattr(_fu, "try_capture_follow_up", None)
        if not all(callable(x) for x in (orig_register, orig_unregister, orig_try_capture)):
            return False

        self._save("follow_up.register_active_runner", orig_register)
        self._save("follow_up.unregister_active_runner", orig_unregister)
        self._save("follow_up.try_capture_follow_up", orig_try_capture)

        def _runner_key(umo: str, runner: Any) -> str:
            ev = _get_runner_event(runner)
            if ev is not None:
                return _sender_key_from_event(ev, umo)
            return umo

        def patched_register(umo: str, runner: Any) -> None:
            orig_register(_runner_key(umo, runner), runner)

        def patched_unregister(umo: str, runner: Any) -> None:
            orig_unregister(_runner_key(umo, runner), runner)

        def patched_try_capture(event: Any) -> Any:
            sender_id = getattr(event, "get_sender_id", None)
            if not callable(sender_id):
                return None
            try:
                sid = sender_id()
                if not sid:
                    return None
            except Exception:
                return None

            key = _sender_key_from_event(event, event.unified_msg_origin)
            runner = _fu._ACTIVE_AGENT_RUNNERS.get(key)
            if not runner:
                return None

            runner_event = getattr(
                getattr(getattr(runner, "run_context", None), "context", None),
                "event",
                None,
            )
            if runner_event is None:
                return None
            active_sender_id = getattr(runner_event, "get_sender_id", None)
            if not callable(active_sender_id):
                return None
            try:
                if str(active_sender_id()) != str(sid):
                    return None
            except Exception:
                return None
            if runner_event.get_extra("agent_stop_requested"):
                return None
            ticket = runner.follow_up(
                message_text=_fu._event_follow_up_text(event)
            )
            if not ticket:
                return None
            order_seq = _fu._allocate_follow_up_order(key)
            monitor_task = asyncio.create_task(
                _fu._monitor_follow_up_ticket(key, ticket, order_seq)
            )
            return _fu.FollowUpCapture(
                umo=key,
                ticket=ticket,
                order_seq=order_seq,
                monitor_task=monitor_task,
            )

        for mod in (_fu, _int):
            setattr(mod, "register_active_runner", patched_register)
            setattr(mod, "unregister_active_runner", patched_unregister)
            setattr(mod, "try_capture_follow_up", patched_try_capture)
        return True
