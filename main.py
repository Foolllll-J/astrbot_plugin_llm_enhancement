import asyncio
import json
import time
import random
from typing import List, Any, Optional
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools
from astrbot.api import logger, AstrBotConfig 
import astrbot.api.message_components as Comp 
from astrbot.api.provider import LLMResponse, ProviderRequest 
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from .modules.sentiment import Sentiment
from .modules.similarity import Similarity
from .modules.state_manager import StateManager, MemberState
from .modules.forward_parser import process_forward_record_content
from .modules.reference_parser import process_reference_context
from .modules.video_parser import (
    download_video_to_temp,
    process_media_content,
)
import os
import shutil
from .modules.info_utils import process_group_members_info, set_group_ban_logic
from .modules.provider_utils import find_provider
from .modules.blacklist import BlacklistManager
from .modules.wake_logic import (
    evaluate_wake_extend,
    detect_wake_media_components,
    normalize_wake_trigger_message,
    evaluate_mention_wake,
    contains_forbidden_wake_word,
)
from .modules.merge_flow import (
    get_event_msg_id,
    prune_member_msg_cache,
    build_event_snapshot,
    upsert_recent_wake_snapshot,
    prepare_initial_merge_snapshots,
    add_pending_msg_id,
    remove_pending_msg_id,
    clear_pending_msg_ids,
    is_msg_still_available,
)
try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False

# ==================== 常量定义 ====================

# AstrBot 内置指令列表
BUILT_CMDS = [
    "llm", "t2i", "tts", "sid", "op", "wl",
    "dashboard_update", "alter_cmd", "provider", "model",
    "plugin", "plugin ls", "new", "switch", "rename",
    "del", "reset", "history", "persona", "tool ls",
    "key", "websearch", "help",
]
RECENT_RECALL_TTL_SEC = 120.0


def _raw_get(raw: Any, key: str, default: Any = None) -> Any:
    """兼容 dict / aiocqhttp Event 的字段读取。"""
    if raw is None:
        return default
    try:
        if isinstance(raw, dict):
            return raw.get(key, default)
        if hasattr(raw, "get"):
            return raw.get(key, default)
        return getattr(raw, key, default)
    except Exception:
        return default

class LLMEnhancement(Star): 
    def __init__(self, context: Context, config: AstrBotConfig): 
        super().__init__(context) 
        self.config = config
        self.cfg = {}
        self._recent_recalled_msg: dict[str, float] = {}
        self._refresh_config()
        self.sent = Sentiment()
        self.similarity = Similarity()
        self.blacklist = BlacklistManager(
            data_dir=StarTools.get_data_dir("astrbot_plugin_llm_enhancement"),
            get_cfg=self._get_cfg,
        )
        logger.info(f"[LLMEnhancement] 插件初始化完成。IS_AIOCQHTTP: {IS_AIOCQHTTP}")

    async def initialize(self):
        await self.blacklist.initialize()

    def _refresh_config(self):
        """将 object 格式的配置平铺到 self.cfg 中"""
        self.cfg = {}
        # 1. 获取顶级配置项
        for k in ["group_whitelist", "group_blacklist"]:
            self.cfg[k] = self.config.get(k)
        
        # 2. 平铺对象配置
        for section in [
            "intelligent_wake",
            "parse_switches",
            "video_injection",
            "forward_parsing",
            "file_parsing",
            "blacklist",
        ]:
            section_cfg = self.config.get(section, {})
            if isinstance(section_cfg, dict):
                for k, v in section_cfg.items():
                    self.cfg[k] = v
                

    def _get_cfg(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        if key in self.cfg:
            return self.cfg[key]
        return self.config.get(key, default)


    def _build_recall_key(self, umo: str, msg_id: str) -> str:
        return f"{umo}::{msg_id}"

    def _mark_recent_recall(self, umo: str, msg_id: str, ttl_sec: float = RECENT_RECALL_TTL_SEC) -> None:
        sid = str(msg_id or "").strip()
        if not sid:
            return
        now_ts = time.time()
        self._recent_recalled_msg[self._build_recall_key(umo, sid)] = now_ts + max(10.0, float(ttl_sec))
        expired = [k for k, exp in self._recent_recalled_msg.items() if exp <= now_ts]
        for k in expired:
            self._recent_recalled_msg.pop(k, None)

    def _consume_recent_recall(self, umo: str, msg_id: str) -> bool:
        sid = str(msg_id or "").strip()
        if not sid:
            return False
        now_ts = time.time()
        key = self._build_recall_key(umo, sid)
        exp = self._recent_recalled_msg.get(key, 0.0)
        if exp <= now_ts:
            if key in self._recent_recalled_msg:
                self._recent_recalled_msg.pop(key, None)
            return False
        self._recent_recalled_msg.pop(key, None)
        return True

    # ==================== 唤醒消息级别 ====================
    
    @filter.event_message_type(filter.EventMessageType.ALL, priority=1)
    async def on_message_event(self, event: AstrMessageEvent):
        """处理消息的初步过滤、黑白名单检查及唤醒逻辑。支持群聊和私聊。"""
        self._refresh_config()
        raw_message = event.message_obj.raw_message if (event.message_obj and hasattr(event.message_obj, "raw_message")) else {}
        if not raw_message and hasattr(event, "event"):
            raw_message = event.event
        if (
            IS_AIOCQHTTP
            and _raw_get(raw_message, "post_type") not in (None, "", "message")
        ):
            raw_post_type = _raw_get(raw_message, "post_type")
            raw_notice_type = _raw_get(raw_message, "notice_type")
            recalled_msg_id = str(_raw_get(raw_message, "message_id") or "").strip()
            if raw_post_type == "notice" and raw_notice_type in {"group_recall", "friend_recall"} and recalled_msg_id:
                self._mark_recent_recall(event.unified_msg_origin, recalled_msg_id)
                logger.debug(
                    "[LLMEnhancement] 记录撤回消息："
                    f"umo={event.unified_msg_origin}, recalled_msg_id={recalled_msg_id}, "
                    f"notice_type={raw_notice_type}"
                )
            logger.debug(
                "[LLMEnhancement] 忽略非消息事件："
                f"post_type={raw_post_type}, "
                f"notice_type={raw_notice_type}, "
                f"request_type={_raw_get(raw_message, 'request_type')}, "
                f"sender={event.get_sender_id()}, group={event.get_group_id()}, "
                f"umo={event.unified_msg_origin}"
            )
            return
        bid: str = event.get_self_id()
        gid: str = event.get_group_id() # 私聊下为 None
        uid: str = event.get_sender_id()
        msg: str = event.message_str.strip() if event.message_str else ""
        
        g = StateManager.get_group(gid or f"private_{uid}")

        # 1. 全局屏蔽检查
        if uid == bid:
            return
        
        # 1.0 内置黑名单拦截（开启“拦截指令”时在消息阶段直接拦截）
        if self._get_cfg("blacklist_block_commands", True):
            if await self.blacklist.intercept_event(event):
                return

        # 仅在群聊环境下检查群黑白名单
        if gid:
            whitelist = self._get_cfg("group_whitelist")
            if whitelist and gid not in whitelist:
                return
            
            blacklist = self._get_cfg("group_blacklist")
            if blacklist and gid in blacklist and not event.is_admin():
                event.stop_event()
                return
            
        # 2. 内置指令屏蔽
        if self._get_cfg("block_builtin"):
            if not event.is_admin() and msg in BUILT_CMDS:
                event.stop_event()
                return
        
        # 3. 初始化状态
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        member = g.members[uid]
        now = time.time()

        # 4. 唤醒条件判断
        wake = event.is_at_or_wake_command
        reason = "at_or_cmd"

        message_chain = []
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            message_chain = event.message_obj.message or []
        (
            has_video_component,
            has_file_component,
            has_forward_component,
            has_json_component,
            file_name,
        ) = detect_wake_media_components(message_chain)

        normalized_msg, normalized_reason = normalize_wake_trigger_message(
            wake=wake,
            msg=msg,
            gid=gid,
            sender_name=event.get_sender_name(),
            has_video_component=has_video_component,
            has_file_component=has_file_component,
            has_forward_component=has_forward_component,
            has_json_component=has_json_component,
            file_name=file_name,
        )
        if normalized_reason:
            msg = normalized_msg
            event.message_str = msg
            reason = normalized_reason

        if not msg:
            if not wake:
                return

        # 提及唤醒 (仅群聊)
        if gid and not wake:
            matched_mention = evaluate_mention_wake(msg, self._get_cfg("mention_wake"))
            if matched_mention:
                wake = True
                reason = f"提及唤醒({matched_mention})"

        # 唤醒延长 (仅群聊)
        if gid and not wake:
            wake, wake_reason = await evaluate_wake_extend(
                event=event,
                msg=msg,
                gid=gid,
                uid=uid,
                now=now,
                group_state=g,
                member=member,
                get_cfg=self._get_cfg,
                get_history_msg=lambda ev, count: self._get_history_msg(ev, count=count),
                similarity_fn=self.similarity.similarity,
                find_provider=self._find_provider,
            )
            if wake and wake_reason:
                reason = wake_reason

        # 话题相关性唤醒 (仅群聊)
        if gid and not wake:
            relevant_wake = self._get_cfg("relevant_wake")
            if relevant_wake:
                if bmsgs := await self._get_history_msg(event, count=5):
                    simi = await self.similarity.similarity(gid, msg, bmsgs)
                    if simi >= relevant_wake:
                        wake = True
                        reason = f"话题相关性{simi:.2f}>={relevant_wake}"

        # 答疑唤醒 (仅群聊)
        if gid and not wake:
            ask_wake = self._get_cfg("ask_wake")
            if ask_wake:  
                if (await self.sent.ask(msg)) >= ask_wake:
                    wake = True
                    reason = "答疑唤醒"

        # 概率唤醒 (仅群聊)
        if gid and not wake:
            prob_wake = self._get_cfg("prob_wake")
            if prob_wake:  
                if random.random() < prob_wake:
                    wake = True
                    reason = "概率唤醒"

        # 违禁词检查
        if not event.is_admin():
            forbidden_word = contains_forbidden_wake_word(
                event.message_str or "",
                self._get_cfg("wake_forbidden_words"),
            )
            if forbidden_word:
                event.stop_event()
                return

        if wake:
            event.is_at_or_wake_command = True
            log_prefix = f"群({gid})" if gid else "私聊"
            logger.info(f"{log_prefix}用户({uid}) {reason}: {msg[:50]}")
            merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
            keep_sec = max(merge_delay * 6, 60.0)
            async with member.lock:
                prune_member_msg_cache(member, keep_sec=keep_sec)
                snap = build_event_snapshot(event, gid, uid)
                upsert_recent_wake_snapshot(member, snap)

    # ==================== 消息合并处理 ====================

    async def _handle_message_merge(self, event: AstrMessageEvent, req: ProviderRequest, gid: str, uid: str, member: MemberState) -> List[Any]:
        """执行消息合并逻辑，根据配置决定是否收集多用户消息并格式化。"""
        group_state = StateManager.get_group(gid or f"private_{uid}")
        merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
        allow_multi_user = self._get_cfg("merge_multi_user", False)
        try:
            merge_max_count = max(1, int(self._get_cfg("merge_max_count", 5) or 5))
        except (TypeError, ValueError):
            merge_max_count = 5
        cache_keep_sec = max(merge_delay * 6, 60.0)
        merged_skip_ttl = max(merge_delay * 2, 20.0)
        merged_window_tolerance = 0.3

        async with member.lock:
            # 初始化状态
            prune_member_msg_cache(member, keep_sec=cache_keep_sec)
            clear_pending_msg_ids(group_state, member)
            member.cancel_merge = False
            member.trigger_msg_id = None
            member.in_merging = True  # 标记正在合并中，避免并发请求重复进入
            preselected_snapshots, member.trigger_msg_id = prepare_initial_merge_snapshots(
                event=event,
                gid=gid,
                uid=uid,
                member=member,
                merge_delay=merge_delay,
                merged_window_tolerance=merged_window_tolerance,
                merged_skip_ttl=merged_skip_ttl,
                add_pending_msg_id=lambda msg_id: add_pending_msg_id(group_state, member, msg_id),
            )

        # buffer 结构: List[Tuple[msg_id, sender_name, message_str]]
        message_buffer = [
            (
                str(item.get("msg_id") or "") or None,
                str(item.get("sender_name") or event.get_sender_name()),
                str(item.get("text") or ""),
            )
            for item in preselected_snapshots
        ]
        def _is_merge_component(seg: Any) -> bool:
            if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File, Comp.Json)):
                return True
            if isinstance(seg, dict):
                seg_type = str(seg.get("type") or "").lower()
                return seg_type in {"forward", "reply", "video", "file", "json"}
            return False

        def _is_message_component(seg: Any) -> bool:
            if isinstance(seg, (Comp.Image, Comp.Video, Comp.File, Comp.Forward, Comp.Json)):
                return True
            if isinstance(seg, dict):
                seg_type = str(seg.get("type") or "").lower()
                return seg_type in {"image", "video", "file", "forward", "json"}
            return False

        additional_components: List[Any] = []
        for item in preselected_snapshots:
            for seg in item.get("components", []) or []:
                if _is_merge_component(seg):
                    additional_components.append(seg)

        @session_waiter(timeout=merge_delay, record_history_chains=False)
        async def collect_messages(controller: SessionController, ev: AstrMessageEvent):
            nonlocal message_buffer, additional_components
            if member.cancel_merge:
                controller.stop()
                return

            # 单通道实时撤回监控：撤回事件由 session_controller 转发到此处处理。
            raw = (
                ev.message_obj.raw_message
                if (hasattr(ev, "message_obj") and hasattr(ev.message_obj, "raw_message"))
                else {}
            )
            if not raw and hasattr(ev, "event"):
                raw = ev.event
            raw_post_type = _raw_get(raw, "post_type")
            raw_notice_type = _raw_get(raw, "notice_type")
            if (
                IS_AIOCQHTTP
                and raw_post_type == "notice"
                and raw_notice_type in {"group_recall", "friend_recall"}
            ):
                recalled_msg_id = str(_raw_get(raw, "message_id") or "").strip()
                if not recalled_msg_id:
                    logger.debug(
                        "[LLMEnhancement] 实时撤回监控：message_id 为空。"
                        f"uid={uid}, notice_type={raw_notice_type}"
                    )
                    return

                async with member.lock:
                    in_pending = recalled_msg_id in member.pending_msg_ids
                    in_buffer = any(str(mid or "") == recalled_msg_id for mid, _n, _c in message_buffer)
                    if not (in_pending or in_buffer):
                        return

                    before_count = len(message_buffer)
                    message_buffer = [
                        (mid, name, content)
                        for mid, name, content in message_buffer
                        if str(mid or "") != recalled_msg_id
                    ]
                    remove_pending_msg_id(group_state, member, recalled_msg_id)
                    member.merged_msg_ids.pop(recalled_msg_id, None)
                    after_count = len(message_buffer)
                    is_trigger_recalled = str(member.trigger_msg_id or "") == recalled_msg_id
                    if is_trigger_recalled and after_count > 0:
                        member.trigger_msg_id = str(message_buffer[0][0] or "")
                    should_stop = after_count <= 0
                    if should_stop:
                        member.cancel_merge = True
                        clear_pending_msg_ids(group_state, member)

                    logger.debug(
                        "[LLMEnhancement] 实时撤回监控命中："
                        f"uid={uid}, recalled_msg_id={recalled_msg_id}, trigger_msg_id={member.trigger_msg_id or 'unknown'}, "
                        f"in_pending={in_pending}, in_buffer={in_buffer}, "
                        f"is_trigger_recalled={is_trigger_recalled}, "
                        f"before={before_count}, after={after_count}, "
                        f"should_stop={should_stop}, new_trigger_msg_id={member.trigger_msg_id or 'unknown'}"
                    )

                    if should_stop:
                        logger.info(
                            "[LLMEnhancement] 实时撤回触发终止，已立即取消本次合并会话。"
                            f"uid={uid}, recalled_msg_id={recalled_msg_id}, after={after_count}"
                        )
                        controller.stop()
                    else:
                        controller.keep(timeout=merge_delay, reset_timeout=True)
                return
            elif IS_AIOCQHTTP and raw_post_type == "notice":
                pass

            # 环境检查
            if gid:
                # 群聊环境：必须是同个群
                if ev.get_group_id() != gid:
                    return
                # 如果不允许跨用户合并，则检查发送者是否一致
                if not allow_multi_user and ev.get_sender_id() != uid:
                    return
            else:
                # 私聊环境：必须是同个发送者
                if ev.get_sender_id() != uid:
                    return
            
            # 检查是否是消息（排除 notice 等）
            if not ev.message_str and not any(_is_message_component(seg) for seg in ev.message_obj.message):
                return
            
            # 获取新消息 ID
            new_msg_id = None
            if hasattr(ev, "message_obj") and hasattr(ev.message_obj, "message_id"):
                new_msg_id = str(ev.message_obj.message_id)

            # 去重：如果只有一条且内容相同（防止重复触发）
            if len(message_buffer) == 1 and ev.message_str == message_buffer[0][2] and ev.get_sender_id() == uid:  
                ev.stop_event()
                return
            
            ev.stop_event()

            if len(message_buffer) >= merge_max_count:
                controller.stop()
                return
            
            # 记录消息 ID
            if new_msg_id:
                async with member.lock:
                    add_pending_msg_id(group_state, member, new_msg_id)
                    member.merged_msg_ids[new_msg_id] = time.time() + merged_skip_ttl

            message_buffer.append((new_msg_id, ev.get_sender_name(), ev.message_str))
            for seg in ev.message_obj.message:
                if _is_merge_component(seg):
                    additional_components.append(seg)
            controller.keep(timeout=merge_delay, reset_timeout=True)
        
        try:
            await collect_messages(event)
        except TimeoutError:
            logger.debug(
                "[LLMEnhancement] collect_messages 等待超时："
                f"uid={uid}, group={gid or 'private'}, trigger_msg_id={member.trigger_msg_id or 'unknown'}, "
                f"buffer_count={len(message_buffer)}"
            )
            pass # 继续后续处理
        finally:
            async with member.lock:
                member.in_merging = False # 合并结束
            
        # 无论是否超时，如果已取消，直接返回
        if member.cancel_merge:
            logger.info(f" [LLMEnhancement] 合并流程已因消息撤回而取消 (用户: {uid})")
            event.stop_event()
            return []

        # 合并结束后做一次协议端校验，兜底撤回事件延迟/丢失。
        if message_buffer and IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent):
            validation_before_count = len(message_buffer)
            filtered_buffer = []
            removed_msg_ids = []
            for mid, name, content in message_buffer:
                if mid and not await is_msg_still_available(event, mid):
                    removed_msg_ids.append(mid)
                    continue
                filtered_buffer.append((mid, name, content))

            if removed_msg_ids:
                async with member.lock:
                    for rid in removed_msg_ids:
                        remove_pending_msg_id(group_state, member, rid)
                removed_ids_text = ",".join(removed_msg_ids)
                logger.info(
                    f"[LLMEnhancement] 最终校验已从本次合并队列移除消息："
                    f"before={validation_before_count}, removed={len(removed_msg_ids)}, "
                    f"remaining={len(filtered_buffer)}, removed_ids={removed_ids_text}, "
                    f"trigger_msg_id={member.trigger_msg_id or 'unknown'}"
                )
            else:
                logger.debug(
                    f"[LLMEnhancement] 最终校验完成（未移除消息）："
                    f"before={validation_before_count}, removed=0, remaining={len(filtered_buffer)}, "
                    f"trigger_msg_id={member.trigger_msg_id or 'unknown'}"
                )
            message_buffer = filtered_buffer

        if len(message_buffer) > 0:
            senders = set(name for _, name, _ in message_buffer)
            if len(senders) > 1:
                merged_msg = "\n".join([f"[{name}]: {msg}" for _, name, msg in message_buffer])
            else:
                merged_msg = " ".join([msg for _, _, msg in message_buffer])

            event.message_str = merged_msg
            req.prompt = merged_msg
            log_prefix = f"群({gid})" if gid else "私聊"
            logger.info(f"{log_prefix}合并：用户({uid})触发，共合并了{len(message_buffer)}条消息 (涉及{len(senders)}人)")
        else:
            member.cancel_merge = True
            logger.info(
                f"[LLMEnhancement] 本次合并上下文消息均已被撤回或不可获取，取消本次请求。"
                f"trigger_msg_id={member.trigger_msg_id or 'unknown'}"
            )
            event.stop_event()
            return []

        return additional_components

    # ==================== LLM 工具注册 ====================

    @filter.command_group("黑名单", alias={"bl"})
    def blacklist():
        """黑名单管理命令组。用法: /黑名单 列表|添加|移除|详情|清空 ..."""
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @blacklist.command("列表", alias={"ls"})
    async def blacklist_ls(self, event: AstrMessageEvent, page: int = 1, page_size: int = 10):
        """查看黑名单列表。用法: /黑名单 列表 [页码] [每页数量]"""
        result = await self.blacklist.command_ls(page=page, page_size=page_size)
        if result.image_base64:
            yield event.chain_result([Comp.Image.fromBase64(result.image_base64)])
        else:
            yield event.plain_result(result.text)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @blacklist.command("添加", alias={"add", "拉黑"})
    async def blacklist_add(
        self,
        event: AstrMessageEvent,
        user_ref: str = "",
        duration: str = "0",
        reason: str = "",
    ):
        """添加黑名单。支持用户ID或@目标。用法: /黑名单 添加 <用户ID/@用户> [时长秒] [原因]"""
        result = await self.blacklist.command_add(
            event=event,
            user_ref=user_ref,
            duration=duration,
            reason=reason,
        )
        yield event.plain_result(result)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @blacklist.command("移除", alias={"rm", "解除"})
    async def blacklist_rm(self, event: AstrMessageEvent, user_ref: str = ""):
        """移除黑名单。支持用户ID或@目标。用法: /黑名单 移除 <用户ID/@用户>"""
        result = await self.blacklist.command_rm(event=event, user_ref=user_ref)
        yield event.plain_result(result)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @blacklist.command("详情", alias={"info", "状态"})
    async def blacklist_info(self, event: AstrMessageEvent, user_ref: str = ""):
        """查看黑名单详情。支持用户ID或@目标。用法: /黑名单 详情 <用户ID/@用户>"""
        result = await self.blacklist.command_info(event=event, user_ref=user_ref)
        if result.image_base64:
            yield event.chain_result([Comp.Image.fromBase64(result.image_base64)])
        else:
            yield event.plain_result(result.text)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @blacklist.command("清空", alias={"clear", "清除"})
    async def blacklist_clear(self, event: AstrMessageEvent):
        """清空黑名单。用法: /黑名单 清空"""
        result = await self.blacklist.command_clear()
        yield event.plain_result(result)

    @filter.llm_tool(name="get_user_avatar")
    async def get_user_avatar(self, event: AstrMessageEvent, user_id: str) -> str:
        """
        获取指定 QQ 用户的头像并将其作为图片附件注入到当前对话中。
        当你需要识别、描述某个人头像特征，或者用户明确要求“看看某人的头像”时使用。

        Args:
            user_id (str): 目标用户的 QQ 号。必须是纯数字字符串。
        """
        if not (IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent)):
            return "当前平台未查看到目标头像。"

        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        
        try:
            req:  ProviderRequest = getattr(event, "_provider_req", None)
            if not req:
                req = getattr(event, "request", None)
            
            if not req:
                logger.error("无法获取当前请求对象 ProviderRequest，注入失败。")
                return f"获取头像成功，但内部错误导致无法注入到当前请求中。"

            user_question = req.prompt.strip()
            context_prompt = (
                f"\n\n以下是系统为你获取到的用户 {user_id} 的头像信息，请根据该头像内容来响应用户的要求。信息如下：\n"
                f"--- 注入内容开始 ---\n"
                f"[图片] 用户 {user_id} 的头像已作为图片附件注入到本次请求的 image_urls 中。\n"
                f"--- 注入内容结束 ---"
            )
            req.prompt = user_question + context_prompt
            req.image_urls.append(avatar_url)
            
            logger.info(f"成功将用户 {user_id} 的头像注入到 LLM 请求中。")
            return f"已成功获取用户 {user_id} 的头像并注入到请求上下文中。"
        except Exception as e:
            logger.error(f"注入头像到请求时发生错误: {e}")
            return f"注入头像失败:  {e}"

    @filter.llm_tool(name="get_group_members_info")
    async def get_group_members(self, event: AstrMessageEvent, group_id: str = None) -> str:
        """
        获取指定 QQ 群的成员列表。
        当需要了解群成员构成、获取成员昵称/ID、统计人数或确认某人是否在群内时调用。
        返回数据包含：
        - user_id: QQ 号
        - nickname: 账户昵称
        - card: 群名片/备注
        - role: 权限角色 (owner/admin/member)

        Args:
            group_id (str, optional): 目标 QQ 群号。如果未提供，将默认获取当前所在群聊的成员。
        """
        return await process_group_members_info(event, group_id)

    @filter.llm_tool(name="set_group_ban")
    async def set_group_ban(self, event: AstrMessageEvent, user_id: str, duration: int, user_name: str, group_id: str = None) -> str:
        """
        在群聊中禁言或解除禁言某位成员。
        支持在群聊中直接使用，或在私聊中指定 group_id 使用，由你根据请求者身份与上下文判断是否应执行。
        
        Args:
            user_id (str): 目标用户的 QQ 号。必须是纯数字字符串。
            duration (int): 禁言时长（秒）。0 为解禁；60-600 为警告级；3600-86400 为惩罚级；最大为 2592000 (30天)。请根据违规严重程度灵活选择。
            user_name (str): 目标用户的昵称或称呼，用于回复确认。
            group_id (str, optional): 目标 QQ 群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
        """
        return await set_group_ban_logic(
            event,
            user_id,
            duration,
            user_name,
            group_id,
            strict_permission_check=bool(self._get_cfg("tool_write_require_admin", False)),
        )

    @filter.llm_tool(name="block_user")
    async def block_user(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        user_name: str = "",
        duration: int = 0,
        reason: str = "",
    ) -> str:
        """
        将指定用户加入黑名单，加入后将忽略对方消息，由你根据请求者身份与上下文判断是否应执行。
        该工具既可用于拉黑，也可用于“未来一段时间不再与该用户对话”的短时策略。
        这是黑名单语义，不是禁言/解禁语义；请避免使用“解封/解禁”等措辞。

        Args:
            user_id (str, optional): 目标用户 ID。为空时会直接拒绝执行。
            user_name (str, optional): 目标用户昵称（可选）。可用于黑名单记录展示。
            duration (int, optional): 拉黑时长（秒）。0 表示按 max_blacklist_duration 处理（其值为 0 时表示永久）；60-600 适合轻度冷却/短时不回应；600-3600 适合明确隔离；86400 及以上用于高风险持续骚扰场景。
            reason (str, optional): 拉黑原因，用于记录与审计。
        """
        return await self.blacklist.tool_block_user(
            event=event,
            user_id=user_id,
            user_name=user_name,
            duration=duration,
            reason=reason,
        )

    @filter.llm_tool(name="unblock_user")
    async def unblock_user(self, event: AstrMessageEvent, user_id: str) -> str:
        """
        将指定用户从黑名单中移除，即将其解除拉黑可重新与其对话，由你根据请求者身份与上下文判断是否应执行。
        请将结果表述为“解除拉黑”，不要使用“解封/解禁”。

        Args:
            user_id (str): 目标用户 ID。
        """
        return await self.blacklist.tool_unblock_user(event=event, user_id=user_id)

    @filter.llm_tool(name="list_blacklist")
    async def list_blacklist(self, event: AstrMessageEvent, page: int = 1, page_size: int = 20) -> str:
        """
        获取黑名单列表（分页）。
        返回中的 expire_time 表示该用户黑名单失效时间，失效后会自动移出黑名单，可重新与其进行对话。
        这是黑名单语义，不是封禁语义。

        Args:
            page (int, optional): 页码，从 1 开始。
            page_size (int, optional): 每页数量，默认 20，最大 50。
        """
        return await self.blacklist.tool_list_blacklist(event=event, page=page, page_size=page_size)

    @filter.llm_tool(name="get_blacklist_status")
    async def get_blacklist_status(self, event: AstrMessageEvent, user_id: str = "") -> str:
        """
        查询用户是否在黑名单中，即是否被拉黑。
        若返回 expire_time，表示黑名单失效时间，失效后会自动移出黑名单，可重新与其进行对话。
        这是黑名单语义，不是封禁语义。

        Args:
            user_id (str, optional): 目标用户 ID，默认当前发送者。
        """
        return await self.blacklist.tool_get_blacklist_status(event=event, user_id=user_id)

    # ==================== LLM 请求级别逻辑 ====================

    @filter.on_llm_request(priority=15)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在请求发送给 LLM 前执行，处理防护、合并及多媒体注入。"""
        setattr(event, "_provider_req", req)
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not uid:
            return

        # 关闭“拦截指令”时，在请求阶段拦截黑名单用户，避免误伤指令。
        if not self._get_cfg("blacklist_block_commands", True):
            if await self.blacklist.intercept_llm_request(event):
                return
        
        g = StateManager.get_group(gid or f"private_{uid}")
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        member = g.members[uid]
        now = time.time()
        msg = event.message_str
        current_msg_id = get_event_msg_id(event)
        if current_msg_id and self._consume_recent_recall(event.unified_msg_origin, current_msg_id):
            logger.info(
                "[LLMEnhancement] on_llm_request 拦截：触发消息已在本次请求前撤回。"
                f"umo={event.unified_msg_origin}, msg_id={current_msg_id}, uid={uid}"
            )
            event.stop_event()
            return
        merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
        cache_keep_sec = max(merge_delay * 6, 60.0)
        async with member.lock:
            prune_member_msg_cache(member, keep_sec=cache_keep_sec)
            if current_msg_id and current_msg_id in member.merged_msg_ids:
                event.stop_event()
                return

        message_chain = []
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            message_chain = event.message_obj.message or []
        if not msg and not message_chain:
            logger.debug(f"[LLMEnhancement] 忽略空消息事件: gid={gid or 'private'}, uid={uid}")
            event.stop_event()
            return
        
        if member.in_merging:
            logger.debug(f"[LLMEnhancement] 当前存在进行中的合并会话，跳过重复请求: gid={gid or 'private'}, uid={uid}")
            event.stop_event()
            return

        if gid and g.shutup_until > now:
            event.stop_event()
            return
        if not event.is_admin() and member.silence_until > now:
            event.stop_event()
            return
        
        request_cd_value = self._get_cfg("request_cd", 0)
        if request_cd_value > 0:
            if now - member.last_request < request_cd_value:
                event.stop_event()
                return
        
        member.last_request = now
        
        try:
            # ==================== 1. 消息合并（前置，避免异步防护导致并发请求拆分） ====================
            all_components = await self._handle_message_merge(event, req, gid, uid, member)
            if member.cancel_merge:
                event.stop_event()
                return

            # ==================== 2. 防护机制（使用合并后的文本） ====================
            msg = event.message_str
            now = time.time()

            if gid:
                shutup_th_cfg = self._get_cfg("shutup")
                if shutup_th_cfg:
                    shut_th = await self.sent.shut(msg)
                    if shut_th > shutup_th_cfg:
                        silence_sec = shut_th * self._get_cfg("silence_multiple", 500)
                        g.shutup_until = now + silence_sec
                        logger.info(f"群({gid})触发闭嘴，沉默{silence_sec:.1f}秒")
                        event.stop_event()
                        return

            insult_th_cfg = self._get_cfg("insult")
            if insult_th_cfg:
                insult_th = await self.sent.insult(msg)
                if insult_th > insult_th_cfg:
                    silence_sec = insult_th * self._get_cfg("silence_multiple", 500)
                    member.silence_until = now + silence_sec
                    logger.info(f"用户({uid})触发辱骂沉默{silence_sec:.1f}秒(下次生效)")

            # 注册清理路径容器
            req._cleanup_paths = []
            
            # ==================== 3. 引用/JSON/文件上下文注入（不含转发聊天记录解析） ====================
            ref_result = await process_reference_context(
                event=event,
                req=req,
                all_components=all_components,
                get_cfg=self._get_cfg,
                download_media=download_video_to_temp,
            )
            injection_summary = {
                "json": bool(getattr(ref_result, "injected_json", False)),
                "file": bool(getattr(ref_result, "injected_file", False)),
                "forward": False,
                "video": False,
            }
            if ref_result.blocked:
                logger.debug(
                    "[LLMEnhancement] 注入摘要: "
                    f"uid={uid}, group={gid or 'private'}, "
                    f"json={injection_summary['json']}, file={injection_summary['file']}, "
                    f"forward={injection_summary['forward']}, video={injection_summary['video']}, "
                    "blocked_by_reference=true"
                )
                event.stop_event()
                return
            reply_seg = ref_result.reply_seg

            # ==================== 4. 转发聊天记录解析 ====================
            handled_forward = await process_forward_record_content(
                event=event,
                req=req,
                forward_id=ref_result.forward_id,
                get_cfg=self._get_cfg,
                get_stt_provider=lambda ev: self._get_stt_provider(event=ev),
                get_vision_provider=lambda ev: self._get_vision_provider(event=ev),
                cleanup_paths=self._cleanup_paths,
            )
            injection_summary["forward"] = bool(handled_forward)
            if handled_forward:
                logger.debug(
                    "[LLMEnhancement] 注入摘要: "
                    f"uid={uid}, group={gid or 'private'}, "
                    f"json={injection_summary['json']}, file={injection_summary['file']}, "
                    f"forward={injection_summary['forward']}, video={injection_summary['video']}"
                )
                return
             
            # ==================== 5. 媒体场景检测与处理 ====================
            injection_summary["video"] = bool(
                await process_media_content(
                    context=self.context,
                    event=event,
                    req=req,
                    all_components=all_components,
                    reply_seg=reply_seg,
                    get_cfg=self._get_cfg,
                )
            )
            logger.debug(
                "[LLMEnhancement] 注入摘要: "
                f"uid={uid}, group={gid or 'private'}, "
                f"json={injection_summary['json']}, file={injection_summary['file']}, "
                f"forward={injection_summary['forward']}, video={injection_summary['video']}"
            )
        
        finally:
            # 6. 统一清理文件
            if hasattr(req, "_cleanup_paths"):
                await self._cleanup_paths(req._cleanup_paths)

    async def _cleanup_paths(self, cleanup_paths: List[str]):
        """延迟清理临时文件"""
        if not cleanup_paths:
            return

        snapshot_paths = [p for p in dict.fromkeys(cleanup_paths) if p]
        if not snapshot_paths:
            return

        async def final_cleanup(paths: List[str]):
            await asyncio.sleep(120)
            for p in paths:
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                    elif os.path.isdir(p):
                        shutil.rmtree(p)
                except Exception as e:
                    logger.debug(f"[LLMEnhancement] cleanup path 失败: {p}, err={e}")

        asyncio.create_task(final_cleanup(snapshot_paths))

    def _find_provider(self, provider_id: str):
        """通用方法：从所有 Provider（包含 LLM 和 STT）中查找匹配的 ID/Name"""
        return find_provider(self.context, provider_id)

    def _get_stt_provider(self, event: Optional[AstrMessageEvent] = None, umo: Optional[str] = None):
        """获取 STT Provider"""
        asr_pid = self._get_cfg("asr_provider_id")
        p = self._find_provider(asr_pid)
        if p:
            logger.debug(f"[LLMEnhancement] 成功匹配到指定 STT Provider: {asr_pid}")
            return p

        if asr_pid:
            logger.warning(f"[LLMEnhancement] 未找到指定 STT Provider: {asr_pid}")

        try:
            using_umo = umo or (getattr(event, "unified_msg_origin", None) if event is not None else None)
            if using_umo:
                return self.context.get_using_stt_provider(umo=using_umo)
            return self.context.get_using_stt_provider()
        except Exception:
            pass

        return None

    def _get_llm_provider(self, event: Optional[AstrMessageEvent] = None, umo: Optional[str] = None):
        """获取当前会话 LLM Provider。"""
        try:
            using_umo = umo or (getattr(event, "unified_msg_origin", None) if event is not None else None)
            if using_umo:
                return self.context.get_using_provider(umo=using_umo)
            return self.context.get_using_provider()
        except Exception:
            return None

    def _get_vision_provider(self, event: Optional[AstrMessageEvent] = None, umo: Optional[str] = None):
        """获取视觉解析 Provider；未指定时回退到当前会话 LLM Provider。"""
        image_pid = self._get_cfg("image_provider_id")
        p = self._find_provider(image_pid)
        if p:
            logger.debug(f"[LLMEnhancement] 成功匹配到指定 Vision Provider: {image_pid}")
            return p

        if image_pid:
            logger.warning(f"[LLMEnhancement] 未找到指定 Vision Provider: {image_pid}，回退到会话 Provider")

        return self._get_llm_provider(event=event, umo=umo)

    @filter.on_llm_response(priority=20)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """在 LLM 返回结果后执行，用于更新会话状态并处理撤回拦截。"""
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not uid:
            return
            
        target_id = gid or f"private_{uid}"
        g = StateManager.get_group(target_id)
        member = g.members.get(uid)
        
        if member:
            # 检查是否在此期间发生了撤回
            if member.cancel_merge:
                logger.info(f" [LLMEnhancement] LLM 响应生成完成，但检测到消息已撤回，拦截回复 (用户: {uid})。")
                member.cancel_merge = False
                clear_pending_msg_ids(g, member)
                event.stop_event() # 拦截响应
                return

            resp_ts = time.time()
            member.last_response = resp_ts
            g.last_response_uid = uid
            g.last_response_ts = resp_ts
            # 清理待处理消息 ID
            clear_pending_msg_ids(g, member)
            member.cancel_merge = False

    async def _get_history_msg(self, event: AstrMessageEvent, role: str = "assistant", count: int | None = 0) -> List[str]:
        try:
            umo = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if not curr_cid:
                return []
            conversation = await self.context.conversation_manager.get_conversation(umo, curr_cid)
            if not conversation:
                return []
            history = json.loads(conversation.history or "[]")
            contexts = [record["content"] for record in history if record.get("role") == role and record.get("content")]
            return contexts[-count:] if count else contexts
        except Exception as e:
            logger.error(f"获取历史消息失败：{e}")
            return []

    async def terminate(self):
        await self.blacklist.terminate()
        logger.info("[LLMEnhancement] 插件已终止")
