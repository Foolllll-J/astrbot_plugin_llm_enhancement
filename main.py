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
from .modules.state_manager import StateManager, GroupState, MemberState
from .modules.forward_parser import process_forward_record_content
from .modules.reference_parser import process_reference_context
from .modules.video_parser import (
    download_video_to_temp,
    process_media_content,
)
from .modules.qq_utils import (
    process_contact_list,
    send_message_logic,
    process_group_info,
    process_group_notices,
    process_group_essence,
    process_group_msg_history,
    process_friend_msg_history,
    process_group_members_info,
    process_group_member_info,
    inject_perception_context_info,
    inject_sender_group_member_info,
    inject_bot_group_member_info,
    kick_group_member_logic,
    set_group_whole_ban_logic,
    set_group_admin_logic,
    set_group_card_logic,
    set_group_special_title_logic,
    set_essence_msg_logic,
    delete_essence_msg_logic,
    delete_msg_logic,
    set_group_name_logic,
    send_group_notice_logic,
    delete_group_notice_logic,
    dismiss_group_logic,
    set_group_kick_members_logic,
    set_group_ban_logic,
)
from .modules.blacklist import BlacklistManager
from .modules.runtime_helpers import (
    cleanup_paths_later,
    resolve_provider,
    get_stt_provider,
    get_vision_provider,
    get_history_messages,
)
from .modules.wake_logic import (
    evaluate_wake_extend,
    detect_wake_media_components,
    normalize_wake_trigger_message,
    evaluate_mention_wake,
    contains_forbidden_wake_word,
)
from .modules.merge_flow import (
    get_event_msg_id,
    extract_merge_components,
    build_message_buffer_from_snapshots,
    collect_additional_components_from_snapshots,
    filter_unavailable_message_buffer,
    apply_merged_message_to_request,
    apply_recall_to_message_buffer_state,
    evaluate_followup_collectability,
    is_duplicate_followup_message,
    append_followup_to_merge_buffer,
    has_recent_wake_in_window,
    prune_member_msg_cache,
    build_event_snapshot,
    ensure_snapshot_merge_key,
    upsert_dynamic_unresolved_snapshot,
    upsert_recent_wake_snapshot,
    prepare_initial_merge_snapshots,
    select_dynamic_owner_uid,
    mark_dynamic_soft_recompute,
    drop_dynamic_batch_from_unresolved,
    execute_dynamic_merge,
    member_contains_msg_id,
    remove_recalled_msg_from_member,
    add_pending_msg_id,
    remove_pending_msg_id,
    clear_pending_msg_ids,
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
            "qq_platform",
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

    def _confirm_timeout_sec(self) -> int:
        raw = self._get_cfg("confirm_timeout_sec", 90)
        try:
            value = int(raw)
        except Exception:
            value = 90
        return max(10, min(600, value))

    def _is_command_trigger_event(self, event: AstrMessageEvent) -> bool:
        handlers = event.get_extra("activated_handlers", default=[]) or []
        for handler in handlers:
            for event_filter in getattr(handler, "event_filters", []) or []:
                if event_filter.__class__.__name__ in {"CommandFilter", "CommandGroupFilter"}:
                    return True
        return False

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

    async def _apply_recall_to_group_state(
        self,
        group_state: GroupState,
        recalled_msg_id: str,
        group_scope: str,
        notice_type: str,
    ) -> bool:
        """把撤回消息同步到本插件状态，返回是否命中并处理。"""
        if not recalled_msg_id:
            return False

        hit = False
        target_uid = group_state.pending_msg_index.get(recalled_msg_id)
        candidate_members: list[MemberState] = []
        if target_uid and target_uid in group_state.members:
            candidate_members.append(group_state.members[target_uid])
        else:
            for m in group_state.members.values():
                if member_contains_msg_id(m, recalled_msg_id):
                    candidate_members.append(m)

        for m in candidate_members:
            async with m.lock:
                if not member_contains_msg_id(m, recalled_msg_id):
                    continue
                summary = remove_recalled_msg_from_member(
                    group_state=group_state,
                    member=m,
                    msg_id=recalled_msg_id,
                )
                if summary.get("marked_cancel") and m.dynamic_inflight_seq > 0:
                    m.dynamic_discard_before_seq = max(
                        m.dynamic_discard_before_seq,
                        int(m.dynamic_inflight_seq),
                    )
                hit = True
                logger.info(
                    "[LLMEnhancement] 撤回命中动态/合并状态："
                    f"group={group_scope}, uid={m.uid}, msg_id={recalled_msg_id}, notice_type={notice_type}, "
                    f"removed_pending={summary.get('removed_pending')}, "
                    f"removed_dynamic={summary.get('removed_dynamic_unresolved')}, "
                    f"marked_cancel={summary.get('marked_cancel')}, inflight_seq={summary.get('inflight_seq')}"
                )
        return hit

    async def _handle_recall_notice(
        self,
        event: AstrMessageEvent,
        recalled_msg_id: str,
        notice_type: str,
    ) -> None:
        """在收到撤回 notice 时同步清理动态合并状态。"""
        if not recalled_msg_id:
            return
        gid = event.get_group_id()
        uid = event.get_sender_id()

        candidates: list[str] = []
        if gid:
            candidates.append(gid)
        if uid:
            candidates.append(f"private_{uid}")

        processed = False
        for scope in dict.fromkeys([c for c in candidates if c]):
            gs = StateManager.get_group_if_exists(scope)
            if not gs:
                continue
            if await self._apply_recall_to_group_state(gs, recalled_msg_id, scope, notice_type):
                processed = True

        if not processed:
            for scope, gs in list(StateManager.iter_groups_items()):
                if await self._apply_recall_to_group_state(gs, recalled_msg_id, scope, notice_type):
                    processed = True
                    break

        if not processed:
            logger.debug(
                "[LLMEnhancement] 撤回未命中插件缓存状态："
                f"group={gid or 'unknown'}, uid={uid or 'unknown'}, msg_id={recalled_msg_id}, notice_type={notice_type}"
            )

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
                await self._handle_recall_notice(event, recalled_msg_id, str(raw_notice_type))
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
        
        # 1.0 内置黑名单拦截（三档：仅LLM/指令+LLM/全消息）
        blacklist_level = self.blacklist.blacklist_intercept_level()
        if blacklist_level == "all_messages":
            if await self.blacklist.intercept_event(event):
                return
        elif blacklist_level == "command_and_llm" and self._is_command_trigger_event(event):
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

        dynamic_merge_mode = bool(self._get_cfg("merge_dynamic_mode", False))
        allow_multi_user = bool(self._get_cfg("merge_multi_user", False))
        followup_require_wake = bool(self._get_cfg("merge_followup_require_wake", False))
        force_dynamic_followup = False
        dynamic_state_uid: Optional[str] = None

        # 动态模式 + 不要求再次唤醒：优先走软重算，不再浪费一次唤醒延长/相关性判定请求。
        if dynamic_merge_mode and (not followup_require_wake) and (not wake):
            merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
            prejoin_window = max(0.5, merge_delay + 0.3)
            prejoin_candidate = False
            target_uid = select_dynamic_owner_uid(
                own_inflight_seq=member.dynamic_inflight_seq,
                dynamic_owner_uid=g.dynamic_owner_uid,
                incoming_uid=uid,
                allow_multi_user=allow_multi_user,
            )
            if not target_uid:
                async with member.lock:
                    prejoin_candidate = has_recent_wake_in_window(
                        member=member,
                        now_ts=now,
                        window_sec=prejoin_window,
                    )
                if prejoin_candidate:
                    target_uid = uid
            if target_uid:
                target_member = g.members.get(target_uid)
                if target_member:
                    dynamic_snap = build_event_snapshot(event, gid, uid)
                    ensure_snapshot_merge_key(dynamic_snap)
                    inflight_seq = await mark_dynamic_soft_recompute(target_member, dynamic_snap)
                    if inflight_seq > 0:
                        wake = True
                        reason = "动态合并跟进"
                        force_dynamic_followup = True
                        dynamic_state_uid = str(target_uid)
                        event.set_extra("_llme_dynamic_followup", True)
                        event.set_extra("_llme_dynamic_state_uid", dynamic_state_uid)
                        logger.debug(
                            "[LLMEnhancement] 动态合并检测到新消息，已标记丢弃旧响应并等待下一次软重算："
                            f"group={gid or 'private'}, owner_uid={target_uid}, incoming_uid={uid}, "
                            f"inflight_seq={inflight_seq}, msg_id={str(dynamic_snap.get('msg_id') or 'unknown')}, "
                            f"require_wake={followup_require_wake}"
                        )
                    elif prejoin_candidate and str(target_uid) == str(uid):
                        async with target_member.lock:
                            upsert_dynamic_unresolved_snapshot(target_member, dynamic_snap)
                        wake = True
                        reason = "动态合并跟进(预请求窗口)"
                        force_dynamic_followup = True
                        dynamic_state_uid = str(target_uid)
                        event.set_extra("_llme_dynamic_state_uid", dynamic_state_uid)
                        logger.debug(
                            "[LLMEnhancement] 动态合并在预请求窗口捕获到新消息："
                            f"group={gid or 'private'}, owner_uid={target_uid}, incoming_uid={uid}, "
                            f"msg_id={str(dynamic_snap.get('msg_id') or 'unknown')}, window={prejoin_window:.2f}s"
                        )

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
                get_history_msg=lambda ev, count: get_history_messages(self.context, ev, count=count),
                similarity_fn=self.similarity.similarity,
                find_provider=lambda provider_id: resolve_provider(self.context, provider_id),
            )
            if wake and wake_reason:
                reason = wake_reason

        # 话题相关性唤醒 (仅群聊)
        if gid and not wake:
            relevant_wake = self._get_cfg("relevant_wake")
            if relevant_wake:
                if bmsgs := await get_history_messages(self.context, event, count=5):
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

        if dynamic_merge_mode and (not force_dynamic_followup):
            target_uid = select_dynamic_owner_uid(
                own_inflight_seq=member.dynamic_inflight_seq,
                dynamic_owner_uid=g.dynamic_owner_uid,
                incoming_uid=uid,
                allow_multi_user=allow_multi_user,
            )
            if target_uid:
                target_member = g.members.get(target_uid)
                if target_member and wake:
                    dynamic_snap = build_event_snapshot(event, gid, uid)
                    ensure_snapshot_merge_key(dynamic_snap)
                    inflight_seq = await mark_dynamic_soft_recompute(target_member, dynamic_snap)
                    if inflight_seq > 0:
                        dynamic_state_uid = str(target_uid)
                        event.set_extra("_llme_dynamic_followup", True)
                        event.set_extra("_llme_dynamic_state_uid", dynamic_state_uid)
                        logger.debug(
                            "[LLMEnhancement] 动态合并检测到新消息，已标记丢弃旧响应并等待下一次软重算："
                            f"group={gid or 'private'}, owner_uid={target_uid}, incoming_uid={uid}, "
                            f"inflight_seq={inflight_seq}, msg_id={str(dynamic_snap.get('msg_id') or 'unknown')}, "
                            f"require_wake={followup_require_wake}"
                        )

        # 违禁词检查
        if (not force_dynamic_followup) and (not event.is_admin()):
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
            logger.debug(f"{log_prefix}用户({uid}) {reason}: {msg[:50]}")
            merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
            keep_sec = max(merge_delay * 6, 60.0)
            snap = {}
            async with member.lock:
                prune_member_msg_cache(member, keep_sec=keep_sec)
                snap = build_event_snapshot(event, gid, uid)
                ensure_snapshot_merge_key(snap)
                upsert_recent_wake_snapshot(member, snap)
                if dynamic_merge_mode and dynamic_state_uid is None:
                    upsert_dynamic_unresolved_snapshot(member, snap)

    # ==================== 消息合并处理 ====================

    async def _handle_message_merge(self, event: AstrMessageEvent, req: ProviderRequest, gid: str, uid: str, member: MemberState) -> List[Any]:
        """执行消息合并逻辑，根据配置决定是否收集多用户消息并格式化。"""
        group_state = StateManager.get_group(gid or f"private_{uid}")
        merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
        allow_multi_user = self._get_cfg("merge_multi_user", False)
        followup_require_wake = bool(self._get_cfg("merge_followup_require_wake", False))
        try:
            merge_max_count = max(1, int(self._get_cfg("merge_max_count", 5) or 5))
        except (TypeError, ValueError):
            merge_max_count = 5
        cache_keep_sec = max(merge_delay * 6, 60.0)
        merged_skip_ttl = max(merge_delay * 6, 120.0)
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
        message_buffer = build_message_buffer_from_snapshots(
            preselected_snapshots,
            default_sender_name=event.get_sender_name(),
        )
        additional_components: List[Any] = collect_additional_components_from_snapshots(preselected_snapshots)

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
                    recall_result = apply_recall_to_message_buffer_state(
                        group_state=group_state,
                        member=member,
                        message_buffer=message_buffer,
                        recalled_msg_id=recalled_msg_id,
                    )
                    if not recall_result.get("hit"):
                        return

                    message_buffer = recall_result.get("message_buffer", message_buffer)
                    in_pending = bool(recall_result.get("in_pending", False))
                    in_buffer = bool(recall_result.get("in_buffer", False))
                    before_count = int(recall_result.get("before_count", 0))
                    after_count = int(recall_result.get("after_count", 0))
                    is_trigger_recalled = bool(recall_result.get("is_trigger_recalled", False))
                    should_stop = bool(recall_result.get("should_stop", False))
                    new_trigger_msg_id = str(recall_result.get("new_trigger_msg_id") or "unknown")

                    logger.debug(
                        "[LLMEnhancement] 实时撤回监控命中："
                        f"uid={uid}, recalled_msg_id={recalled_msg_id}, trigger_msg_id={member.trigger_msg_id or 'unknown'}, "
                        f"in_pending={in_pending}, in_buffer={in_buffer}, "
                        f"is_trigger_recalled={is_trigger_recalled}, "
                        f"before={before_count}, after={after_count}, "
                        f"should_stop={should_stop}, new_trigger_msg_id={new_trigger_msg_id}"
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

            can_collect, reject_reason = evaluate_followup_collectability(
                ev=ev,
                gid=gid,
                uid=uid,
                allow_multi_user=bool(allow_multi_user),
                followup_require_wake=followup_require_wake,
            )
            if not can_collect:
                if reject_reason == "wake_required":
                    logger.debug(
                        "[LLMEnhancement] 硬等待合并跳过未唤醒后续消息："
                        f"uid={uid}, group={gid or 'private'}, sender={ev.get_sender_id()}"
                    )
                return

            if is_duplicate_followup_message(message_buffer, ev, uid):
                ev.stop_event()
                return

            ev.stop_event()

            if len(message_buffer) >= merge_max_count:
                controller.stop()
                return

            message_buffer, additional_components, _new_msg_id = await append_followup_to_merge_buffer(
                group_state=group_state,
                member=member,
                message_buffer=message_buffer,
                additional_components=additional_components,
                ev=ev,
                merged_skip_ttl=merged_skip_ttl,
            )
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
            filtered_buffer, removed_msg_ids = await filter_unavailable_message_buffer(event, message_buffer)

            if removed_msg_ids:
                async with member.lock:
                    for rid in removed_msg_ids:
                        remove_pending_msg_id(group_state, member, rid)
                removed_ids_text = ",".join(removed_msg_ids)
                logger.debug(
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
            sender_count = apply_merged_message_to_request(event, req, message_buffer)
                
            log_prefix = f"群({gid})" if gid else "私聊"
            logger.debug(f"{log_prefix}合并：用户({uid})触发，共合并了{len(message_buffer)}条消息 (涉及{sender_count}人)")
        else:
            member.cancel_merge = True
            logger.info(
                f"[LLMEnhancement] 本次合并上下文消息均已被撤回或不可获取，取消本次请求。"
                f"trigger_msg_id={member.trigger_msg_id or 'unknown'}"
            )
            event.stop_event()
            return []

        return additional_components

    async def _handle_message_merge_dynamic(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        gid: str,
        uid: str,
        member: MemberState,
    ) -> List[Any]:
        """动态合并模式：不硬等待，直接使用待确认消息池做软重算。"""
        allow_multi_user = bool(self._get_cfg("merge_multi_user", False))
        merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
        try:
            merge_max_count = max(1, int(self._get_cfg("merge_max_count", 5) or 5))
        except (TypeError, ValueError):
            merge_max_count = 5
        result = await execute_dynamic_merge(
            event=event,
            req=req,
            gid=gid,
            uid=uid,
            member=member,
            group_state=StateManager.get_group(gid or f"private_{uid}"),
            allow_multi_user=allow_multi_user,
            merge_delay=merge_delay,
            merge_max_count=merge_max_count,
            is_recent_recalled=lambda msg_id: self._consume_recent_recall(event.unified_msg_origin, msg_id),
        )
        if result.get("cancelled"):
            logger.info(
                "[LLMEnhancement] 动态合并无可用消息，已取消本次请求："
                f"uid={uid}, group={gid or 'private'}, request_seq={int(result.get('request_seq') or 0)}"
            )
            return []
        removed_recalled_ids = result.get("removed_recalled_ids", []) or []
        if removed_recalled_ids:
            logger.debug(
                "[LLMEnhancement] 动态合并已过滤撤回消息："
                f"uid={uid}, group={gid or 'private'}, request_seq={int(result.get('request_seq') or 0)}, "
                f"removed_ids={','.join([str(x) for x in removed_recalled_ids])}"
            )
        removed_unavailable_ids = result.get("removed_unavailable_ids", []) or []
        if removed_unavailable_ids:
            logger.debug(
                "[LLMEnhancement] 动态合并兜底校验已过滤不可用消息："
                f"uid={uid}, group={gid or 'private'}, request_seq={int(result.get('request_seq') or 0)}, "
                f"removed_ids={','.join([str(x) for x in removed_unavailable_ids])}"
            )
        log_prefix = f"群({gid})" if gid else "私聊"
        logger.debug(
            f"{log_prefix}动态合并：请求seq={int(result.get('request_seq') or 0)}，"
            f"合并{int(result.get('message_count') or 0)}条消息 "
            f"(涉及{int(result.get('sender_count') or 0)}人，待确认池{int(result.get('unresolved_count') or 0)}条)"
        )
        logger.debug(
            "[LLMEnhancement] 动态合并批次详情："
            f"uid={uid}, group={gid or 'private'}, request_seq={int(result.get('request_seq') or 0)}, "
            f"batch_keys={result.get('selected_keys', [])}"
        )
        return result.get("additional_components", [])

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
            
            logger.debug(f"成功将用户 {user_id} 的头像注入到 LLM 请求中。")
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
            group_id (str, optional): 目标 QQ 群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
        """
        return await process_group_members_info(event, group_id)

    @filter.llm_tool(name="get_group_member_info")
    async def get_group_member_info(
        self,
        event: AstrMessageEvent,
        user_id: str,
        group_id: str = None,
        no_cache: bool = False,
    ) -> str:
        """
        获取指定 QQ 群成员详情。
        适合用于确认某个成员的身份信息、群身份、群头衔、群名片等。

        Args:
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            user_id (str): 目标用户 QQ 号（必填）。
            no_cache (bool, optional): 是否跳过缓存直接查询 OneBot。
        """
        return await process_group_member_info(
            event=event,
            user_id=user_id,
            group_id=group_id,
            no_cache=no_cache,
        )

    @filter.llm_tool(name="get_group_info")
    async def get_group_info(self, event: AstrMessageEvent, group_id: str = None, no_cache: bool = False) -> str:
        """
        获取指定 QQ 群的群信息（群名、人数等）。

        Args:
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            no_cache (bool, optional): 是否跳过缓存直接查询 OneBot。
        """
        return await process_group_info(event=event, group_id=group_id, no_cache=no_cache)

    @filter.llm_tool(name="get_group_notices")
    async def get_group_notices(self, event: AstrMessageEvent, group_id: str = None, limit: int = 10) -> str:
        """
        获取指定 QQ 群的群公告列表。

        Args:
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            limit (int, optional): 返回条数上限，默认 10，最大 50。
        """
        return await process_group_notices(event=event, group_id=group_id, limit=limit)

    @filter.llm_tool(name="get_group_essence")
    async def get_group_essence(self, event: AstrMessageEvent, group_id: str = None, limit: int = 10) -> str:
        """
        获取指定 QQ 群的精华消息列表。

        Args:
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            limit (int, optional): 返回条数上限，默认 10，最大 50。
        """
        return await process_group_essence(event=event, group_id=group_id, limit=limit)

    @filter.llm_tool(name="get_contact_list")
    async def get_contact_list(self, event: AstrMessageEvent, limit_each: int = 200) -> str:
        """
        查看通讯录：同时返回群列表与好友列表。

        Args:
            limit_each (int, optional): 群列表和好友列表各自返回上限，默认 200，最大 1000。
        """
        return await process_contact_list(event=event, limit_each=limit_each)

    @filter.llm_tool(name="send_message")
    async def send_message(
        self,
        event: AstrMessageEvent,
        chat_type: str,
        message: str,
        group_id: str = None,
        user_id: str = None,
        auto_escape: bool = False,
    ) -> str:
        """
        发送消息到 QQ 私聊或群聊。仅用于被要求发送消息至非当前会话时调用，例如“帮我发消息给某人/某群”。
        本工具只执行基础发送，不参与其他插件的二次语义解析。
        不要在 message 中附加“表情包触发标记/特殊控制标记”等约定字符串来触发额外行为。

        Args:
            chat_type (str): 发送类型。仅支持 group 或 private。
            message (str): 要发送的消息文本。
            group_id (str, optional): 当 chat_type=group 时必填。
            user_id (str, optional): 当 chat_type=private 时必填。
            auto_escape (bool, optional): 是否将 CQ 码按纯文本发送（True=不解析，False=按 CQ 码解析），默认 False。
        """
        return await send_message_logic(
            event=event,
            chat_type=chat_type,
            message=message,
            group_id=group_id,
            user_id=user_id,
            auto_escape=auto_escape,
        )

    @filter.llm_tool(name="get_msg_history")
    async def get_msg_history(
        self,
        event: AstrMessageEvent,
        chat_type: str,
        group_id: str = "",
        user_id: str = "",
        count: int = 50,
        search_keywords: str = "",
        time_range: str = "",
    ) -> str:
        """
        获取历史消息（群聊/私聊），支持关键词搜索与时间范围过滤。
        仅在用户明确要求需查历史消息时调用，避免无关上下文扩张。

        Args:
            chat_type (str): 查询类型。仅支持 group 或 private。
            group_id (str, optional): 当 chat_type=group 时必填。
            user_id (str, optional): 当 chat_type=private 时必填。
            count (int, optional): 返回条数上限，默认 50，最大 300。
            search_keywords (str, optional): 搜索关键词。支持多个，使用逗号/竖线/换行分隔。
            time_range (str, optional): 时间范围（严格格式，不要使用“今天下午/刚刚/晚点”等自然语言）。
                仅支持:
                1) 今天
                2) 昨天
                3) 最近N小时 (示例: 最近6小时)
                4) YYYY-MM-DD HH:MM 到 YYYY-MM-DD HH:MM
                5) 今天 HH:MM 到 HH:MM / 昨天 HH:MM 到 HH:MM
        """
        mode = str(chat_type or "").strip().lower()
        if mode == "group":
            return await process_group_msg_history(
                event=event,
                group_id=group_id or None,
                count=count,
                search_keywords=search_keywords,
                time_range=time_range,
            )
        if mode == "private":
            return await process_friend_msg_history(
                event=event,
                user_id=user_id,
                count=count,
                search_keywords=search_keywords,
                time_range=time_range,
            )
        return json.dumps(
            {"error": "chat_type 参数无效。仅支持 group 或 private。"},
            ensure_ascii=False,
        )

    @filter.llm_tool(name="set_group_ban")
    async def set_group_ban(
        self,
        event: AstrMessageEvent,
        user_id: str,
        duration: int,
        user_name: str,
        group_id: str = None,
    ) -> str:
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
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
        )

    @filter.llm_tool(name="kick_group_member")
    async def kick_group_member(
        self,
        event: AstrMessageEvent,
        user_id: str,
        group_id: str = None,
        reject_add_request: bool = False,
        confirm_token: str = "",
    ) -> str:
        """
        将指定成员踢出群聊。

        Args:
            user_id (str): 目标用户 ID。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            reject_add_request (bool, optional): 是否拒绝该用户再次加群请求。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await kick_group_member_logic(
            event=event,
            user_id=user_id,
            group_id=group_id,
            reject_add_request=reject_add_request,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
        )

    @filter.llm_tool(name="set_group_whole_ban")
    async def set_group_whole_ban(
        self,
        event: AstrMessageEvent,
        enable: bool,
        group_id: str = None,
        confirm_token: str = "",
    ) -> str:
        """
        开启或关闭群全员禁言。

        Args:
            enable (bool): True 表示开启，False 表示关闭。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await set_group_whole_ban_logic(
            event=event,
            enable=enable,
            group_id=group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
        )

    @filter.llm_tool(name="set_group_admin")
    async def set_group_admin(
        self,
        event: AstrMessageEvent,
        user_id: str,
        enable: bool,
        group_id: str = None,
        confirm_token: str = "",
    ) -> str:
        """
        设置或取消群管理员。

        Args:
            user_id (str): 目标用户 ID。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            enable (bool): True 设为管理员，False 取消管理员。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await set_group_admin_logic(
            event=event,
            user_id=user_id,
            enable=enable,
            group_id=group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
        )

    @filter.llm_tool(name="set_group_card")
    async def set_group_card(self, event: AstrMessageEvent, user_id: str, card: str, group_id: str = None) -> str:
        """
        设置群成员名片即群昵称。

        Args:
            user_id (str): 目标用户 ID。
            card (str): 新的群名片文本。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
        """
        return await set_group_card_logic(
            event=event,
            user_id=user_id,
            card=card,
            group_id=group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
        )

    @filter.llm_tool(name="set_group_special_title")
    async def set_group_special_title(
        self,
        event: AstrMessageEvent,
        user_id: str,
        special_title: str,
        group_id: str = None,
    ) -> str:
        """
        设置群成员专属头衔。

        Args:
            user_id (str): 目标用户 ID。
            special_title (str): 头衔内容。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
        """
        return await set_group_special_title_logic(
            event=event,
            user_id=user_id,
            special_title=special_title,
            group_id=group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
        )

    @filter.llm_tool(name="set_essence_msg")
    async def set_essence_msg(self, event: AstrMessageEvent, message_id: str = "") -> str:
        """
        将指定消息设置为群精华消息。仅支持群聊中使用。

        Args:
            message_id (str, optional): 目标消息 ID。若未通过工具获取目标 message_id，则无需填写，将尝试从当前消息引用(reply)自动提取。
        """
        return await set_essence_msg_logic(
            event=event,
            message_id=message_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
        )

    @filter.llm_tool(name="delete_essence_msg")
    async def delete_essence_msg(
        self,
        event: AstrMessageEvent,
        message_id: str = "",
        confirm_token: str = "",
    ) -> str:
        """
        将指定消息移出群精华列表。

        Args:
            message_id (str, optional): 目标消息 ID。可先调用 get_group_essence 获取后再传入；若未传入，则无需填写。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await delete_essence_msg_logic(
            event=event,
            message_id=message_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
        )

    @filter.llm_tool(name="delete_msg")
    async def delete_msg(self, event: AstrMessageEvent, message_id: str = "", confirm_token: str = "") -> str:
        """
        撤回一条消息。

        Args:
            message_id (str, optional): 目标消息 ID。若未通过工具获取目标 message_id，则无需填写，将尝试从当前消息引用(reply)自动提取。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await delete_msg_logic(
            event=event,
            message_id=message_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
        )

    @filter.llm_tool(name="set_group_name")
    async def set_group_name(self, event: AstrMessageEvent, group_name: str, group_id: str = None) -> str:
        """
        修改群名称。

        Args:
            group_name (str): 新群名称。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
        """
        return await set_group_name_logic(
            event=event,
            group_name=group_name,
            group_id=group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
        )

    @filter.llm_tool(name="send_group_notice")
    async def send_group_notice(
        self,
        event: AstrMessageEvent,
        content: str,
        group_id: str = None,
        pinned: bool = False,
    ) -> str:
        """
        发送群公告。

        Args:
            content (str): 公告正文内容。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            pinned (bool, optional): 是否置顶公告。
        """
        return await send_group_notice_logic(
            event=event,
            content=content,
            group_id=group_id,
            pinned=pinned,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
        )

    @filter.llm_tool(name="delete_group_notice")
    async def delete_group_notice(
        self,
        event: AstrMessageEvent,
        notice_id: str,
        group_id: str = None,
        confirm_token: str = "",
    ) -> str:
        """
        删除指定群公告。

        Args:
            notice_id (str): 公告 ID。可先调用 get_group_notices 获取公告列表并从中选择目标 notice_id。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await delete_group_notice_logic(
            event=event,
            notice_id=notice_id,
            group_id=group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
        )

    @filter.llm_tool(name="dismiss_group")
    async def dismiss_group(self, event: AstrMessageEvent, group_id: str = None, confirm_token: str = "") -> str:
        """
        解散群聊（高风险操作）。

        Args:
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await dismiss_group_logic(
            event=event,
            group_id=group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
        )

    @filter.llm_tool(name="set_group_kick_members")
    async def set_group_kick_members(
        self,
        event: AstrMessageEvent,
        user_ids: str,
        group_id: str = None,
        reject_add_request: bool = False,
        confirm_token: str = "",
    ) -> str:
        """
        批量踢出群成员。

        Args:
            user_ids (str): 用户 ID 列表，支持逗号分隔字符串。
            group_id (str, optional): 目标群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
            reject_add_request (bool, optional): 是否拒绝这些用户再次加群请求。
            confirm_token (str, optional): 二次确认令牌。首次调用可能返回 token，第二次原参数不变并携带 token 才执行。
        """
        return await set_group_kick_members_logic(
            event=event,
            user_ids=user_ids,
            group_id=group_id,
            reject_add_request=reject_add_request,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
            confirm_required_tools=self._get_cfg("confirm_required_tools", []),
            confirm_timeout_sec=self._confirm_timeout_sec(),
            confirm_token=confirm_token,
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

        # 任意拦截等级都会拦截 LLM 请求。
        if await self.blacklist.intercept_llm_request(event):
            return
        
        dynamic_merge_mode = bool(self._get_cfg("merge_dynamic_mode", False))
        g = StateManager.get_group(gid or f"private_{uid}")
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        sender_member = g.members[uid]
        state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip() if dynamic_merge_mode else uid
        if state_uid not in g.members:
            g.members[state_uid] = MemberState(uid=state_uid)
        merge_member = g.members[state_uid]
        if dynamic_merge_mode and state_uid != uid:
            logger.debug(
                "[LLMEnhancement] 动态合并状态重定向："
                f"group={gid or 'private'}, sender_uid={uid}, state_uid={state_uid}"
            )
        now = time.time()
        msg = event.message_str
        current_msg_id = get_event_msg_id(event)
        is_dynamic_followup = bool(event.get_extra("_llme_dynamic_followup", default=False))
        if current_msg_id and self._consume_recent_recall(event.unified_msg_origin, current_msg_id):
            logger.info(
                "[LLMEnhancement] on_llm_request 拦截：触发消息已在本次请求前撤回。"
                f"umo={event.unified_msg_origin}, msg_id={current_msg_id}, uid={uid}"
            )
            event.stop_event()
            return
        merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
        cache_keep_sec = max(merge_delay * 6, 60.0)
        async with merge_member.lock:
            prune_member_msg_cache(merge_member, keep_sec=cache_keep_sec)
            if (
                dynamic_merge_mode
                and current_msg_id
                and current_msg_id in merge_member.merged_msg_ids
                and (not is_dynamic_followup)
            ):
                logger.debug(
                    "[LLMEnhancement] 动态合并跳过已并入消息，避免队列中的重复请求："
                    f"group={gid or 'private'}, sender_uid={uid}, state_uid={state_uid}, msg_id={current_msg_id}"
                )
                event.stop_event()
                return
            if not dynamic_merge_mode and current_msg_id and current_msg_id in merge_member.merged_msg_ids:
                event.stop_event()
                return

        message_chain = []
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            message_chain = event.message_obj.message or []
        if not msg and not message_chain:
            logger.debug(f"[LLMEnhancement] 忽略空消息事件: gid={gid or 'private'}, uid={uid}")
            event.stop_event()
            return
        
        if (not dynamic_merge_mode) and merge_member.in_merging:
            logger.debug(f"[LLMEnhancement] 当前存在进行中的合并会话，跳过重复请求: gid={gid or 'private'}, uid={uid}")
            event.stop_event()
            return

        if gid and g.shutup_until > now:
            event.stop_event()
            return
        if not event.is_admin() and sender_member.silence_until > now:
            event.stop_event()
            return
        
        request_cd_value = self._get_cfg("request_cd", 0)
        if request_cd_value > 0 and (not is_dynamic_followup):
            if now - sender_member.last_request < request_cd_value:
                event.stop_event()
                return
        
        sender_member.last_request = now
        
        try:
            # ==================== 1. 消息合并（前置，避免异步防护导致并发请求拆分） ====================
            if dynamic_merge_mode:
                all_components = await self._handle_message_merge_dynamic(event, req, gid, uid, merge_member)
            elif merge_delay > 0:
                all_components = await self._handle_message_merge(event, req, gid, uid, merge_member)
            else:
                all_components = extract_merge_components(event)
                logger.debug(
                    "[LLMEnhancement] 消息合并已关闭："
                    f"gid={gid or 'private'}, uid={uid}, merge_delay={merge_delay}"
                )
            if merge_member.cancel_merge:
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
                    sender_member.silence_until = now + silence_sec
                    logger.info(f"用户({uid})触发辱骂沉默{silence_sec:.1f}秒(下次生效)")

            perception_fields = self._get_cfg("perception_injection_fields", [])
            perception_injected = bool(
                await inject_perception_context_info(
                    event=event,
                    req=req,
                    raw_fields=perception_fields,
                    timezone_name="Asia/Shanghai",
                    holiday_country="CN",
                    no_cache=bool(self._get_cfg("extra_info_injection_no_cache", False)),
                )
            )

            selected_fields = self._get_cfg("extra_info_injection_fields", [])
            inject_no_cache = bool(self._get_cfg("extra_info_injection_no_cache", False))
            sender_member_injected = bool(
                await inject_sender_group_member_info(
                    event,
                    req,
                    raw_fields=selected_fields,
                    no_cache=inject_no_cache,
                )
            )
            bot_member_injected = False
            if bool(self._get_cfg("extra_info_injection_include_bot_self", False)):
                bot_member_injected = bool(
                    await inject_bot_group_member_info(
                        event,
                        req,
                        raw_fields=selected_fields,
                        no_cache=inject_no_cache,
                    )
                )

            injection_summary = {
                "perception": perception_injected,
                "member_info": sender_member_injected,
                "bot_member_info": bot_member_injected,
                "json": False,
                "file": False,
                "forward": False,
                "video": False,
            }

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
            injection_summary["json"] = bool(getattr(ref_result, "injected_json", False))
            injection_summary["file"] = bool(getattr(ref_result, "injected_file", False))
            if ref_result.blocked:
                logger.debug(
                    "[LLMEnhancement] 注入摘要: "
                    f"uid={uid}, group={gid or 'private'}, "
                    f"perception={injection_summary['perception']}, "
                    f"member_info={injection_summary['member_info']}, "
                    f"bot_member_info={injection_summary['bot_member_info']}, "
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
                get_stt_provider=lambda ev: get_stt_provider(self.context, self._get_cfg, event=ev),
                get_vision_provider=lambda ev: get_vision_provider(self.context, self._get_cfg, event=ev),
                cleanup_paths=cleanup_paths_later,
            )
            injection_summary["forward"] = bool(handled_forward)
            if handled_forward:
                logger.debug(
                    "[LLMEnhancement] 注入摘要: "
                    f"uid={uid}, group={gid or 'private'}, "
                    f"perception={injection_summary['perception']}, "
                    f"member_info={injection_summary['member_info']}, "
                    f"bot_member_info={injection_summary['bot_member_info']}, "
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
                f"perception={injection_summary['perception']}, "
                f"member_info={injection_summary['member_info']}, "
                f"bot_member_info={injection_summary['bot_member_info']}, "
                f"json={injection_summary['json']}, file={injection_summary['file']}, "
                f"forward={injection_summary['forward']}, video={injection_summary['video']}"
            )
        
        finally:
            if dynamic_merge_mode and event.is_stopped():
                dynamic_req_seq = int(event.get_extra("_llme_dynamic_request_seq") or 0)
                dynamic_batch_keys = event.get_extra("_llme_dynamic_batch_keys", default=[]) or []
                if dynamic_req_seq > 0:
                    async with merge_member.lock:
                        drop_dynamic_batch_from_unresolved(merge_member, dynamic_batch_keys)
                        if merge_member.dynamic_inflight_seq == dynamic_req_seq:
                            merge_member.dynamic_inflight_seq = 0
                        if g.dynamic_owner_uid == state_uid:
                            g.dynamic_owner_uid = None
                    clear_pending_msg_ids(g, merge_member)
                    merge_member.cancel_merge = False

            # 6. 统一清理文件
            if hasattr(req, "_cleanup_paths"):
                await cleanup_paths_later(req._cleanup_paths)

    @filter.on_llm_response(priority=20)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """在 LLM 返回结果后执行，用于更新会话状态并处理撤回拦截。"""
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not uid:
            return
            
        target_id = gid or f"private_{uid}"
        g = StateManager.get_group(target_id)
        state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip()
        member = g.members.get(state_uid)
        
        if member:
            dynamic_merge_mode = bool(self._get_cfg("merge_dynamic_mode", False))
            dynamic_req_seq = int(event.get_extra("_llme_dynamic_request_seq") or 0)
            dynamic_batch_keys = event.get_extra("_llme_dynamic_batch_keys", default=[]) or []
            if dynamic_merge_mode and dynamic_req_seq > 0:
                should_drop = False
                async with member.lock:
                    if dynamic_req_seq <= member.dynamic_discard_before_seq:
                        should_drop = True
                    else:
                        drop_dynamic_batch_from_unresolved(member, dynamic_batch_keys)
                    if member.dynamic_inflight_seq == dynamic_req_seq:
                        member.dynamic_inflight_seq = 0
                    if g.dynamic_owner_uid == state_uid:
                        g.dynamic_owner_uid = None

                if should_drop:
                    logger.debug(
                        "[LLMEnhancement] 动态合并放弃上次请求响应："
                        f"uid={state_uid}, sender_uid={uid}, group={gid or 'private'}, response_seq={dynamic_req_seq}, "
                        f"discard_before_seq={member.dynamic_discard_before_seq}"
                    )
                    clear_pending_msg_ids(g, member)
                    member.cancel_merge = False
                    event.stop_event()
                    return

            # 检查是否在此期间发生了撤回
            if member.cancel_merge:
                logger.info(
                    " [LLMEnhancement] LLM 响应生成完成，但检测到消息已撤回，拦截回复 "
                    f"(state_uid: {state_uid}, sender_uid: {uid})。"
                )
                member.cancel_merge = False
                clear_pending_msg_ids(g, member)
                event.stop_event() # 拦截响应
                return

            event.set_extra("_llme_pending_last_response_update", True)
            # 清理待处理消息 ID
            clear_pending_msg_ids(g, member)
            member.cancel_merge = False

    @filter.after_message_sent(priority=100)
    async def after_message_sent(self, event: AstrMessageEvent):
        """消息实际发送后再更新唤醒延长时间锚点。"""
        if not bool(event.get_extra("_llme_pending_last_response_update", default=False)):
            return

        uid: str = event.get_sender_id()
        if not uid:
            return
        state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip()

        if not getattr(event, "_has_send_oper", False):
            logger.debug(
                "[LLMEnhancement] 跳过唤醒延长锚点更新：本次事件未实际发送消息。"
                f"uid={uid}, state_uid={state_uid}, group={event.get_group_id() or 'private'}"
            )
            event.set_extra("_llme_pending_last_response_update", False)
            return

        gid: str = event.get_group_id()
        target_id = gid or f"private_{uid}"
        g = StateManager.get_group(target_id)
        member = g.members.get(state_uid)
        if not member:
            g.members[state_uid] = MemberState(uid=state_uid)
            member = g.members[state_uid]

        resp_ts = time.time()
        member.last_response = resp_ts
        g.last_response_uid = state_uid
        g.last_response_ts = resp_ts
        event.set_extra("_llme_pending_last_response_update", False)
        logger.debug(
            "[LLMEnhancement] 已更新唤醒延长锚点："
            f"uid={uid}, state_uid={state_uid}, group={gid or 'private'}, ts={resp_ts:.3f}"
        )

    async def terminate(self):
        await self.blacklist.terminate()
        logger.info("[LLMEnhancement] 插件已终止")

