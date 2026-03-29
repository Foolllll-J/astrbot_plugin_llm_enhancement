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
from .modules.state_manager import StateManager, GroupState, MemberState
from .modules.forward_parser import process_forward_record_content
from .modules.reference_parser import (
    process_reference_context,
    inject_current_message_image_context,
    inject_current_message_forward_origin_context,
)
from .modules.video_parser import (
    download_video_to_temp,
    process_media_content,
)
from .modules.qq_utils import (
    process_contact_list,
    process_user_avatar,
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
    BuiltinCommandMatcher,
    EffectiveDialogHistory,
    cleanup_paths_later,
    resolve_provider,
    get_stt_provider,
    get_vision_provider,
    get_discarded_response_ttl_sec,
    clear_discarded_response_cache,
    store_discarded_response_cache,
    is_llm_response_empty_without_tool,
    is_chain_effectively_empty,
    looks_like_error_result,
    apply_discarded_response_fallback,
)
from .modules.wake_logic import (
    normalize_concurrency_limit,
    try_acquire_request_concurrency_slot,
    release_request_concurrency_slot,
    is_wake_prefix_triggered,
    evaluate_wake_extend,
    apply_post_wake_judge_gate,
    build_media_trigger_message,
    detect_wake_media_components,
    normalize_wake_trigger_message,
    prepare_empty_mention_context,
    evaluate_mention_wake,
    contains_forbidden_wake_word,
    compute_relevant_context_substring_downweight,
)
from .modules.merge_flow import (
    load_merge_runtime_config,
    normalize_event_ts,
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
    request_dynamic_recompute_stop,
    schedule_dynamic_recompute_requeue,
    drop_dynamic_batch_from_unresolved,
    reset_dynamic_capture_session,
    execute_dynamic_merge,
    member_contains_msg_id,
    remove_recalled_msg_from_member,
    add_pending_msg_id,
    remove_pending_msg_id,
    clear_pending_msg_ids,
)
from .modules.dialogue_context import (
    is_context_injection_enabled,
    get_context_injection_max_messages,
    get_context_non_text_parse_options,
    get_wake_history_messages,
    get_image_component_label,
    extract_raw_image_datas_from_event,
    compute_active_wake_adjustment,
    extract_addressing_signals,
    try_build_reply_preview,
    try_get_image_caption,
    build_text_context_enrichment,
    build_non_text_context_text,
    build_context_text,
    append_group_context_message,
    append_notice_context_from_raw,
    inject_context_into_request,
)
try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False

# ==================== 常量定义 ====================

RECENT_RECALL_TTL_SEC = 120.0
DYNAMIC_DISCARDED_RESPONSE_TTL_SEC = 300.0

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
        self._request_counter_lock = asyncio.Lock()
        self._active_request_refs: dict[str, int] = {}
        self._active_request_meta: dict[str, tuple[str, str]] = {}
        self._effective_dialog_history = EffectiveDialogHistory(max_turns=4)
        self._builtin_cmd_matcher = BuiltinCommandMatcher(cache_ttl_sec=60.0)
        self._refresh_config()
        self.sent = Sentiment()
        self.similarity = Similarity()
        self.blacklist = BlacklistManager(
            data_dir=StarTools.get_data_dir("astrbot_plugin_llm_enhancement"),
            get_cfg=self._get_cfg,
        )
        qq_adapter_status = "可用" if IS_AIOCQHTTP else "不可用"
        logger.info(f"[LLMEnhancement] 插件初始化完成。QQ 平台适配器（aiocqhttp）状态：{qq_adapter_status}")

    async def initialize(self):
        await self.blacklist.initialize()
        await self._builtin_cmd_matcher.refresh_cache(force=True)

    def _refresh_config(self):
        """将 object 格式的配置平铺到 self.cfg 中"""
        self.cfg = {}
        # 1. 获取顶级配置项
        for k in ["group_whitelist"]:
            self.cfg[k] = self.config.get(k)
        
        # 2. 平铺对象配置
        for section in [
            "intelligent_wake",
            "context_injection",
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

    async def _release_concurrency_slot_if_needed(
        self,
        event: AstrMessageEvent,
        *,
        reason: str,
    ) -> int:
        """Release pre-acquired concurrency slot if still held."""
        if not bool(event.get_extra("_llme_concurrency_acquired", default=False)):
            return -1
        if bool(event.get_extra("_llme_concurrency_released", default=False)):
            return -1

        slot_key = str(event.get_extra("_llme_concurrency_key", default="") or "").strip()
        left_ref = 0
        if slot_key:
            async with self._request_counter_lock:
                left_ref = release_request_concurrency_slot(
                    active_request_refs=self._active_request_refs,
                    active_request_meta=self._active_request_meta,
                    key=slot_key,
                )
        event.set_extra("_llme_concurrency_released", True)
        logger.debug(
            "[LLMEnhancement] 并发槽位释放："
            f"key={slot_key}, left_ref={left_ref}, reason={reason}, "
            f"group={event.get_group_id() or 'private'}, uid={event.get_sender_id()}"
        )
        return left_ref

    def _is_command_trigger_event(self, event: AstrMessageEvent) -> bool:
        handlers_parsed_params = event.get_extra("handlers_parsed_params", default={}) or {}
        if not isinstance(handlers_parsed_params, dict) or not handlers_parsed_params:
            return False

        activated_handlers = event.get_extra("activated_handlers", default=[]) or []
        activated_full_names = {
            str(getattr(h, "handler_full_name", "") or "").strip()
            for h in activated_handlers
            if str(getattr(h, "handler_full_name", "") or "").strip()
        }
        if not activated_full_names:
            return bool(handlers_parsed_params)

        for full_name in handlers_parsed_params.keys():
            if str(full_name or "").strip() in activated_full_names:
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
            raw_group_id = str(_raw_get(raw_message, "group_id") or "").strip()
            if raw_group_id and is_context_injection_enabled(self._get_cfg):
                notice_group_state = StateManager.get_group(raw_group_id)
                appended_notice_context, notice_source = append_notice_context_from_raw(
                    group_state=notice_group_state,
                    raw_message=raw_message,
                    get_cfg=self._get_cfg,
                )
                if appended_notice_context:
                    logger.debug(
                        "[LLMEnhancement][ContextInjection] notice context appended: "
                        f"group={raw_group_id}, source={notice_source}, post_type={raw_post_type}, notice_type={raw_notice_type}"
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
        context_injection_enabled = bool(gid) and is_context_injection_enabled(self._get_cfg)
        context_injection_max_messages = (
            get_context_injection_max_messages(self._get_cfg)
            if context_injection_enabled
            else 0
        )
        
        g = StateManager.get_group(gid or f"private_{uid}")
        async def _get_wake_context_messages(_event, count: int) -> list[str]:
            msgs = get_wake_history_messages(
                group_state=g,
                get_cfg=self._get_cfg,
                count=count,
            )
            if msgs:
                return msgs
            # 兜底：当上下文注入关闭或暂无记录时，回退到旧的有效对话历史。
            return await self._effective_dialog_history.get_history_messages(_event, count)
        command_trigger_event = self._is_command_trigger_event(event)

        # 1. 全局屏蔽检查
        if uid == bid:
            return
        
        # 1.0 内置黑名单拦截（三档：仅LLM/指令+LLM/全消息）
        blacklist_level = self.blacklist.blacklist_intercept_level()
        if blacklist_level == "all_messages":
            if await self.blacklist.intercept_event(event):
                return
        elif blacklist_level == "command_and_llm" and command_trigger_event:
            if await self.blacklist.intercept_event(event):
                return

        # 仅在群聊环境下检查群黑白名单
        if gid:
            whitelist = self._get_cfg("group_whitelist")
            if whitelist and gid not in whitelist:
                return
            
        # 2. 内置指令屏蔽
        if self._get_cfg("block_builtin"):
            if not event.is_admin():
                await self._builtin_cmd_matcher.refresh_cache()
                if self._builtin_cmd_matcher.is_builtin_trigger_event(event) or self._builtin_cmd_matcher.is_builtin_command_text(msg):
                    event.stop_event()
                    return
        
        # 3. 初始化状态
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        member = g.members[uid]
        now = time.time()

        # 4. 唤醒条件判断
        direct_wake = bool(event.is_at_or_wake_command)
        wake = direct_wake
        reason = "at_or_cmd"

        message_chain = []
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            message_chain = event.message_obj.message or []
        (
            has_image_component,
            has_video_component,
            has_file_component,
            has_forward_component,
            has_json_component,
            file_name,
        ) = detect_wake_media_components(message_chain)
        at_targets, at_bot, at_all, reply_to_id, reply_msg_id = extract_addressing_signals(
            message_chain,
            bot_id=bid,
        )
        reply_to_bot = bool(bid and reply_to_id and str(reply_to_id) == str(bid))
        wake_prefixes = []
        try:
            config = self.context.get_config(event.unified_msg_origin)
            wake_prefixes = config.get("wake_prefix", []) or []
        except Exception:
            try:
                config = self.context.get_config()
                wake_prefixes = config.get("wake_prefix", []) or []
            except Exception:
                wake_prefixes = []
        prefix_wake_triggered = is_wake_prefix_triggered(
            original_message=getattr(event.message_obj, "message_str", ""),
            wake_prefixes=wake_prefixes,
        )
        event.set_extra("_llme_prefix_wake_triggered", prefix_wake_triggered)
        event.set_extra("_llme_addressed_to_bot", bool(at_bot or reply_to_bot))
        raw_image_datas = extract_raw_image_datas_from_event(event)
        active_wake_factor = 1.0
        active_wake_reasons: list[str] = []
        relevant_wake_factor = 1.0
        relevant_wake_reasons: list[str] = []
        if gid:
            active_wake_factor, active_wake_reasons = compute_active_wake_adjustment(
                group_state=g,
                current_uid=uid,
                bot_id=bid,
                at_targets=at_targets,
                at_all=at_all,
                reply_to_id=reply_to_id,
                include_state_bias=False,
            )
            relevant_wake_factor, relevant_wake_reasons = compute_active_wake_adjustment(
                group_state=g,
                current_uid=uid,
                bot_id=bid,
                at_targets=at_targets,
                at_all=at_all,
                reply_to_id=reply_to_id,
                include_state_bias=True,
            )

        if context_injection_enabled:
            reply_preview = await try_build_reply_preview(event, reply_msg_id)
            parse_options = get_context_non_text_parse_options(self._get_cfg)
            current_msg_id = get_event_msg_id(event) or ""
            context_text = ""
            if msg:
                text_enrichment = await build_text_context_enrichment(
                    event=event,
                    get_cfg=self._get_cfg,
                    message_chain=message_chain,
                    parse_options=parse_options,
                )
                if "image" in parse_options and has_image_component:
                    image_caption = await try_get_image_caption(
                        event=event,
                        message_chain=message_chain,
                        get_cfg=self._get_cfg,
                        provider_by_id_resolver=lambda provider_id: resolve_provider(self.context, provider_id),
                        default_provider_resolver=lambda: get_vision_provider(self.context, self._get_cfg, event=event),
                    )
                    if image_caption:
                        image_label = get_image_component_label(message_chain, raw_image_datas=raw_image_datas)
                        text_enrichment = (
                            f"{text_enrichment} | {image_label}转述: {image_caption}"
                            if text_enrichment
                            else f"{image_label}转述: {image_caption}"
                        )
                if text_enrichment:
                    context_text = f"{msg} | {text_enrichment}"
            if (not context_text) and (not msg):
                context_text = await build_non_text_context_text(
                    event=event,
                    get_cfg=self._get_cfg,
                    message_chain=message_chain,
                    sender_name=event.get_sender_name(),
                    msg_id=current_msg_id,
                    parse_options=parse_options,
                    raw_image_datas=raw_image_datas,
                    provider_by_id_resolver=lambda provider_id: resolve_provider(self.context, provider_id),
                    default_provider_resolver=lambda: get_vision_provider(self.context, self._get_cfg, event=event),
                )
            if not context_text:
                image_caption = ""
                image_label = "图片"
                if (not msg) and ("image" in parse_options):
                    # 非文本场景下，若勾选了图片解析，再尝试图片转述并参与回退构建。
                    image_caption = await try_get_image_caption(
                        event=event,
                        message_chain=message_chain,
                        get_cfg=self._get_cfg,
                        provider_by_id_resolver=lambda provider_id: resolve_provider(self.context, provider_id),
                        default_provider_resolver=lambda: get_vision_provider(self.context, self._get_cfg, event=event),
                    )
                    image_label = get_image_component_label(message_chain, raw_image_datas=raw_image_datas)
                context_text = build_context_text(
                    msg,
                    event.get_sender_name(),
                    image_caption=image_caption,
                    has_image_component=has_image_component,
                    has_video_component=has_video_component,
                    has_file_component=has_file_component,
                    has_forward_component=has_forward_component,
                    has_json_component=has_json_component,
                    file_name=file_name,
                    image_label=image_label,
                )
            parse_features: list[str] = []
            if "聊天记录抽取:" in context_text:
                parse_features.append("forward")
            if "URL解析:" in context_text:
                parse_features.append("url")
            if "JSON解析:" in context_text:
                parse_features.append("json")
            if "文件信息:" in context_text:
                parse_features.append("file")
            if ("图片转述:" in context_text) or ("表情包转述:" in context_text):
                parse_features.append("image")
            append_group_context_message(
                g,
                uid=uid,
                sender_name=event.get_sender_name(),
                message_text=context_text,
                max_messages=context_injection_max_messages,
                msg_id=current_msg_id,
                is_bot=False,
                source="incoming",
                at_targets=at_targets,
                at_bot=at_bot,
                at_all=at_all,
                reply_to_id=reply_to_id,
                reply_msg_id=reply_msg_id,
                reply_preview=reply_preview,
                parse_features=parse_features,
                now_ts=now,
                get_cfg=self._get_cfg,
            )

        normalized_msg, normalized_reason = normalize_wake_trigger_message(
            wake=wake,
            msg=msg,
            gid=gid,
            sender_name=event.get_sender_name(),
            has_image_component=has_image_component,
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
            if normalized_reason == "空@唤醒":
                await prepare_empty_mention_context(
                    event=event,
                    gid=gid,
                    uid=uid,
                    current_msg_id=get_event_msg_id(event) or "",
                    history_count=30,
                )

        merge_cfg = load_merge_runtime_config(self._get_cfg)
        dynamic_merge_mode = bool(merge_cfg.dynamic_mode and merge_cfg.delay_sec > 0)
        allow_multi_user = merge_cfg.allow_multi_user
        followup_require_wake = merge_cfg.followup_require_wake
        force_dynamic_followup = False
        dynamic_state_uid: Optional[str] = None
        requeued_dynamic_followup = bool(event.get_extra("_llme_dynamic_requeued", default=False))

        if not msg and dynamic_merge_mode and (not followup_require_wake) and (not wake):
            dynamic_msg, dynamic_reason = build_media_trigger_message(
                sender_name=event.get_sender_name(),
                has_image_component=has_image_component,
                has_video_component=has_video_component,
                has_file_component=has_file_component,
                has_forward_component=has_forward_component,
                has_json_component=has_json_component,
                file_name=file_name,
            )
            if dynamic_reason:
                msg = dynamic_msg
                event.message_str = msg
                logger.debug(
                    "[LLMEnhancement] 无文本媒体消息已转换为动态合并占位文本："
                    f"group={gid or 'private'}, uid={uid}, reason={dynamic_reason}, msg={msg}"
                )

        if not msg:
            if not wake:
                return

        if dynamic_merge_mode and requeued_dynamic_followup:
            forced_state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip() or uid
            if forced_state_uid not in g.members:
                g.members[forced_state_uid] = MemberState(uid=forced_state_uid)
            requeue_from_seq = int(event.get_extra("_llme_dynamic_requeue_from_seq", default=0) or 0)
            forced_member = g.members[forced_state_uid]
            async with forced_member.lock:
                current_req_seq = int(forced_member.dynamic_request_seq or 0)
            if requeue_from_seq > 0 and current_req_seq > requeue_from_seq:
                logger.debug(
                    "[LLMEnhancement] 跳过过期动态重排事件："
                    f"group={gid or 'private'}, sender_uid={uid}, state_uid={forced_state_uid}, "
                    f"requeue_from_seq={requeue_from_seq}, current_req_seq={current_req_seq}"
                )
                return
            wake = True
            reason = "动态重排跟进"
            force_dynamic_followup = True
            dynamic_state_uid = forced_state_uid
            event.set_extra("_llme_dynamic_followup", True)
            event.set_extra("_llme_dynamic_state_uid", dynamic_state_uid)
            logger.debug(
                "[LLMEnhancement] 动态软重算重排事件进入强制跟进："
                f"group={gid or 'private'}, sender_uid={uid}, state_uid={dynamic_state_uid}"
            )

        # 动态模式 + 不要求再次唤醒：优先走软重算，不再浪费一次唤醒延长/相关性判定请求。
        if dynamic_merge_mode and (not followup_require_wake) and (not wake):
            prejoin_window = max(0.5, merge_cfg.delay_sec + 0.3)
            prejoin_candidate = False
            target_uid = select_dynamic_owner_uid(
                own_inflight_seq=member.dynamic_inflight_seq,
                dynamic_owner_uid=g.dynamic_owner_uid,
                incoming_uid=uid,
                allow_multi_user=allow_multi_user,
            )
            if target_uid and (not allow_multi_user) and str(target_uid) != str(uid):
                target_uid = None
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
                    recompute_decision = await mark_dynamic_soft_recompute(
                        target_member,
                        dynamic_snap,
                        merge_delay=merge_cfg.delay_sec,
                        merge_max_count=merge_cfg.max_count,
                    )
                    if (not recompute_decision.accepted) and (
                        recompute_decision.reason in {
                            "deadline_reached",
                            "max_count_reached",
                        }
                    ):
                        return
                    inflight_seq = int(recompute_decision.inflight_seq or 0)
                    if recompute_decision.accepted and inflight_seq > 0:
                        stop_requested = request_dynamic_recompute_stop(
                            event,
                            inflight_seq=inflight_seq,
                            owner_uid=str(target_uid),
                        )
                        if stop_requested > 0:
                            requeued = schedule_dynamic_recompute_requeue(
                                event,
                                event_queue=self.context.get_event_queue(),
                                owner_uid=str(target_uid),
                                inflight_seq=inflight_seq,
                            )
                            if requeued:
                                logger.debug(
                                    "[LLMEnhancement] 动态软重算已中断旧流程并停止当前事件，等待重排消息触发新请求："
                                    f"group={gid or 'private'}, owner_uid={target_uid}, incoming_uid={uid}, "
                                    f"inflight_seq={inflight_seq}"
                                )
                                event.stop_event()
                                return
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
                            f"require_wake={followup_require_wake}, stop_requested={stop_requested}"
                        )
                    elif (
                        prejoin_candidate
                        and str(target_uid) == str(uid)
                        and str(recompute_decision.reason or "") == "no_inflight"
                    ):
                        async with target_member.lock:
                            upsert_dynamic_unresolved_snapshot(target_member, dynamic_snap)
                        logger.debug(
                            "[LLMEnhancement] 动态合并预请求窗口缓存消息（未启动合并）："
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
                get_history_msg=_get_wake_context_messages,
                similarity_fn=self.similarity.similarity,
                find_provider=lambda provider_id: resolve_provider(self.context, provider_id),
            )
            if wake and wake_reason:
                reason = wake_reason

        # 话题相关性唤醒 (仅群聊)
        if gid and not wake:
            relevant_wake = self._get_cfg("relevant_wake")
            if relevant_wake:
                relevant_ctx_count = self.similarity.context_count_for_query(msg)
                if bmsgs := await self._effective_dialog_history.get_history_messages(
                    event,
                    relevant_ctx_count,
                ):
                    simi = await self.similarity.similarity(gid, msg, bmsgs)
                    repeat_factor, _hit_count = compute_relevant_context_substring_downweight(
                        msg,
                        bmsgs,
                    )
                    adjusted_simi = simi * relevant_wake_factor * repeat_factor
                    if adjusted_simi >= relevant_wake:
                        wake = True
                        reason = f"话题相关性{adjusted_simi:.2f}>={relevant_wake:.2f}"

        # 答疑唤醒 (仅群聊)
        if gid and not wake:
            ask_wake = self._get_cfg("ask_wake")
            if ask_wake:  
                ask_score = await self.sent.ask(msg)
                adjusted_ask_score = ask_score * active_wake_factor
                if adjusted_ask_score >= ask_wake:
                    wake = True
                    reason = f"答疑唤醒{adjusted_ask_score:.2f}>={ask_wake:.2f}"

        # 无聊唤醒 (仅群聊)
        if gid and not wake:
            bored_wake = self._get_cfg("bored_wake")
            if bored_wake:
                bored_score = await self.sent.bored(msg)
                adjusted_bored_score = bored_score * active_wake_factor
                if adjusted_bored_score >= bored_wake:
                    wake = True
                    reason = f"无聊唤醒{adjusted_bored_score:.2f}>={bored_wake:.2f}"

        # 概率唤醒 (仅群聊)
        if gid and not wake:
            prob_wake = self._get_cfg("prob_wake")
            if prob_wake:  
                prob_roll = random.random()
                adjusted_prob_wake = float(prob_wake) * active_wake_factor
                if prob_roll < adjusted_prob_wake:
                    wake = True
                    reason = f"概率唤醒{prob_roll:.4f}<{adjusted_prob_wake:.4f}"

        event.set_extra("_llme_direct_wake", direct_wake)
        event.set_extra("_llme_wake_reason", reason)
        event.set_extra("_llme_command_trigger_event", command_trigger_event)
        event.set_extra("_llme_force_dynamic_followup", force_dynamic_followup)

        if dynamic_merge_mode and (not force_dynamic_followup) and (not command_trigger_event):
            target_uid = select_dynamic_owner_uid(
                own_inflight_seq=member.dynamic_inflight_seq,
                dynamic_owner_uid=g.dynamic_owner_uid,
                incoming_uid=uid,
                allow_multi_user=allow_multi_user,
            )
            if target_uid and (not allow_multi_user) and str(target_uid) != str(uid):
                target_uid = None
            if target_uid:
                target_member = g.members.get(target_uid)
                if target_member and wake:
                    dynamic_snap = build_event_snapshot(event, gid, uid)
                    ensure_snapshot_merge_key(dynamic_snap)
                    recompute_decision = await mark_dynamic_soft_recompute(
                        target_member,
                        dynamic_snap,
                        merge_delay=merge_cfg.delay_sec,
                        merge_max_count=merge_cfg.max_count,
                    )
                    if (not recompute_decision.accepted) and (
                        recompute_decision.reason in {
                            "deadline_reached",
                            "max_count_reached",
                        }
                    ):
                        return
                    inflight_seq = int(recompute_decision.inflight_seq or 0)
                    if recompute_decision.accepted and inflight_seq > 0:
                        stop_requested = request_dynamic_recompute_stop(
                            event,
                            inflight_seq=inflight_seq,
                            owner_uid=str(target_uid),
                        )
                        if stop_requested > 0:
                            requeued = schedule_dynamic_recompute_requeue(
                                event,
                                event_queue=self.context.get_event_queue(),
                                owner_uid=str(target_uid),
                                inflight_seq=inflight_seq,
                            )
                            if requeued:
                                logger.debug(
                                    "[LLMEnhancement] 动态软重算已中断旧流程并停止当前事件，等待重排消息触发新请求："
                                    f"group={gid or 'private'}, owner_uid={target_uid}, incoming_uid={uid}, "
                                    f"inflight_seq={inflight_seq}"
                                )
                                event.stop_event()
                                return
                        dynamic_state_uid = str(target_uid)
                        event.set_extra("_llme_dynamic_followup", True)
                        event.set_extra("_llme_dynamic_state_uid", dynamic_state_uid)
                        logger.debug(
                            "[LLMEnhancement] 动态合并检测到新消息，已标记丢弃旧响应并等待下一次软重算："
                            f"group={gid or 'private'}, owner_uid={target_uid}, incoming_uid={uid}, "
                            f"inflight_seq={inflight_seq}, msg_id={str(dynamic_snap.get('msg_id') or 'unknown')}, "
                            f"require_wake={followup_require_wake}, stop_requested={stop_requested}"
                        )
        elif dynamic_merge_mode and command_trigger_event and member.dynamic_inflight_seq > 0:
            logger.debug(
                "[LLMEnhancement] 跳过命令消息的动态软重算，保留正在进行中的 LLM 响应："
                f"group={gid or 'private'}, uid={uid}, msg={msg[:50]}, inflight_seq={member.dynamic_inflight_seq}"
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
            async with member.lock:
                member.last_wake_ts = now
            keep_sec = max(max(merge_cfg.delay_sec, 10.0) * 6, 60.0)
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
        merge_cfg = load_merge_runtime_config(self._get_cfg)
        merge_delay = merge_cfg.delay_sec
        allow_multi_user = merge_cfg.allow_multi_user
        followup_require_wake = merge_cfg.followup_require_wake
        merge_max_count = merge_cfg.max_count
        ttl_base = max(merge_cfg.delay_sec, 10.0)
        cache_keep_sec = max(ttl_base * 6, 60.0)
        merged_skip_ttl = max(ttl_base * 6, 120.0)
        wait_timeout_sec = max(0.1, merge_cfg.delay_sec)
        merged_window_tolerance = 0.3
        merge_start_ts = time.time()
        merge_deadline_ts = merge_start_ts + merge_delay

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
                merge_delay=wait_timeout_sec,
                merged_window_tolerance=merged_window_tolerance,
                merged_skip_ttl=merged_skip_ttl,
                merge_max_count=merge_max_count,
                add_pending_msg_id=lambda msg_id: add_pending_msg_id(group_state, member, msg_id),
            )

        # buffer 结构: List[Tuple[msg_id, sender_name, message_str]]
        message_buffer = build_message_buffer_from_snapshots(
            preselected_snapshots,
            default_sender_name=event.get_sender_name(),
        )
        additional_components: List[Any] = collect_additional_components_from_snapshots(preselected_snapshots)

        @session_waiter(timeout=wait_timeout_sec, record_history_chains=False)
        async def collect_messages(controller: SessionController, ev: AstrMessageEvent):
            nonlocal message_buffer, additional_components
            if member.cancel_merge:
                controller.stop()
                return
            if time.time() > merge_deadline_ts:
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
                        remaining_sec = merge_deadline_ts - time.time()
                        if remaining_sec <= 0:
                            controller.stop()
                        else:
                            controller.keep(timeout=min(wait_timeout_sec, remaining_sec), reset_timeout=True)
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
            if len(message_buffer) >= merge_max_count:
                controller.stop()
                return
            remaining_sec = merge_deadline_ts - time.time()
            if remaining_sec <= 0:
                controller.stop()
                return
            controller.keep(timeout=min(wait_timeout_sec, remaining_sec), reset_timeout=True)
        
        try:
            if len(message_buffer) < merge_max_count:
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
                member.merge_start_ts = 0.0
            
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
        merge_cfg = load_merge_runtime_config(self._get_cfg)
        allow_multi_user = merge_cfg.allow_multi_user
        merge_delay = merge_cfg.delay_sec
        merge_max_count = merge_cfg.max_count
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
    async def get_user_avatar(self, event: AstrMessageEvent, user_id: str = "") -> Any:
        """
        获取指定 QQ 用户的头像并将其作为图片附件注入到当前对话中。
        当你需要识别、描述某个人头像特征，或者用户明确要求“看看某人的头像”时使用。

        Args:
            user_id (str, optional): 目标用户的 QQ 号。若目标就是当前对话者，可不提供；留空时默认使用当前消息发送者的 ID。
        """
        return await process_user_avatar(event=event, user_id=user_id)

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
        user_id: str = "",
        duration: int = 0,
        group_id: str = None,
    ) -> str:
        """
        在群聊中禁言或解除禁言某位成员。
        支持在群聊中直接使用，或在私聊中指定 group_id 使用，由你根据请求者身份与上下文判断是否应执行。
        
        Args:
            user_id (str, optional): 目标用户的 QQ 号。若操作对象就是当前对话者，可不提供；留空时默认使用当前消息发送者的 ID。
            duration (int): 禁言时长（秒）。0 为解禁；60-600 为警告级；3600-86400 为惩罚级；最大为 2592000 (30天)。请根据违规严重程度灵活选择。
            group_id (str, optional): 目标 QQ 群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
        """
        return await set_group_ban_logic(
            event,
            user_id,
            duration,
            group_id,
            admin_required_tools=self._get_cfg("tool_admin_required_tools", []),
            enabled_dangerous_tools=self._get_cfg("enabled_dangerous_tools", []),
        )

    @filter.llm_tool(name="kick_group_member")
    async def kick_group_member(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        group_id: str = None,
        reject_add_request: bool = False,
        confirm_token: str = "",
    ) -> str:
        """
        将指定成员踢出群聊。

        Args:
            user_id (str, optional): 目标用户 ID。若操作对象就是当前对话者，可不提供；留空时默认使用当前消息发送者的 ID。
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
        user_id: str = "",
        enable: bool = True,
        group_id: str = None,
        confirm_token: str = "",
    ) -> str:
        """
        设置或取消群管理员。

        Args:
            user_id (str, optional): 目标用户 ID。若操作对象就是当前对话者，可不提供；留空时默认使用当前消息发送者的 ID。
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
    async def set_group_card(self, event: AstrMessageEvent, user_id: str = "", card: str = "", group_id: str = None) -> str:
        """
        设置群成员名片即群昵称。

        Args:
            user_id (str, optional): 目标用户 ID。若操作对象就是当前对话者，可不提供；留空时默认使用当前消息发送者的 ID。
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
        user_id: str = "",
        special_title: str = "",
        group_id: str = None,
    ) -> str:
        """
        设置群成员专属头衔。

        Args:
            user_id (str, optional): 目标用户 ID。若操作对象就是当前对话者，可不提供；留空时默认使用当前消息发送者的 ID。
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
            user_id (str, optional): 目标用户 ID。若拉黑对象就是当前对话者，可不提供；留空时会默认使用当前消息发送者的 ID。
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
    async def get_blacklist_status(self, event: AstrMessageEvent, user_id: str) -> str:
        """
        查询用户是否在黑名单中，即是否被拉黑。
        若返回 expire_time，表示黑名单失效时间，失效后会自动移出黑名单，可重新与其进行对话。
        这是黑名单语义，不是封禁语义。

        Args:
            user_id (str): 目标用户 ID。
        """
        return await self.blacklist.tool_get_blacklist_status(event=event, user_id=user_id)

    # ==================== LLM 请求级别逻辑 ====================

    @filter.on_waiting_llm_request(priority=15)
    async def on_waiting_llm_request(self, event: AstrMessageEvent):
        """在框架会话锁之前做并发预占位，超限时直接拦截。"""
        if event.is_stopped():
            return

        self._refresh_config()
        uid: str = event.get_sender_id()
        if not uid:
            return
        gid: str = event.get_group_id()

        if bool(event.get_extra("_llme_concurrency_acquired", default=False)) and (
            not bool(event.get_extra("_llme_concurrency_released", default=False))
        ):
            return

        merge_cfg = load_merge_runtime_config(self._get_cfg)
        dynamic_merge_mode = bool(merge_cfg.dynamic_mode and merge_cfg.delay_sec > 0)
        state_uid = (
            str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip()
            if dynamic_merge_mode
            else uid
        )
        current_msg_id = get_event_msg_id(event) or ""

        user_limit = normalize_concurrency_limit(self._get_cfg("max_user_concurrent_requests", 0))
        group_limit = normalize_concurrency_limit(self._get_cfg("max_group_concurrent_requests", 0))
        async with self._request_counter_lock:
            accepted_slot, slot_key, slot_detail = try_acquire_request_concurrency_slot(
                active_request_refs=self._active_request_refs,
                active_request_meta=self._active_request_meta,
                uid=uid,
                gid=gid,
                state_uid=state_uid,
                dynamic_merge_mode=dynamic_merge_mode,
                current_msg_id=current_msg_id,
                event_identity=id(event),
                now_ns=time.time_ns(),
                user_limit=user_limit,
                group_limit=group_limit,
            )
        if not accepted_slot:
            logger.debug(f"[LLMEnhancement] on_waiting_llm_request 并发拦截：{slot_detail}")
            event.stop_event()
            return

        event.set_extra("_llme_concurrency_key", slot_key)
        event.set_extra("_llme_concurrency_acquired", True)
        event.set_extra("_llme_concurrency_released", False)
        event.set_extra("_llme_concurrency_preacquired", True)

    @filter.on_llm_request(priority=15)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在请求发送给 LLM 前执行，处理防护、合并及多媒体注入。"""
        setattr(event, "_provider_req", req)
        if event.is_stopped():
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_skip_stopped"
            )
            return
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not uid:
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_missing_uid"
            )
            return

        # 任意拦截等级都会拦截 LLM 请求。
        if await self.blacklist.intercept_llm_request(event):
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_blacklist_intercept"
            )
            return
        
        merge_cfg = load_merge_runtime_config(self._get_cfg)
        dynamic_merge_mode = bool(merge_cfg.dynamic_mode and merge_cfg.delay_sec > 0)
        g = StateManager.get_group(gid or f"private_{uid}")
        async def _get_wake_context_messages(_event, count: int) -> list[str]:
            msgs = get_wake_history_messages(
                group_state=g,
                get_cfg=self._get_cfg,
                count=count,
            )
            if msgs:
                return msgs
            # 兜底：当上下文注入关闭或暂无记录时，回退到旧的有效对话历史。
            return await self._effective_dialog_history.get_history_messages(_event, count)
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        sender_member = g.members[uid]
        state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip() if dynamic_merge_mode else uid
        if state_uid not in g.members:
            g.members[state_uid] = MemberState(uid=state_uid)
        merge_member = g.members[state_uid]
        requeue_from_seq = int(event.get_extra("_llme_dynamic_requeue_from_seq", default=0) or 0)
        is_requeued_dynamic_followup = bool(event.get_extra("_llme_dynamic_requeued", default=False))
        if dynamic_merge_mode and is_requeued_dynamic_followup and requeue_from_seq > 0:
            async with merge_member.lock:
                current_req_seq = int(merge_member.dynamic_request_seq or 0)
            if current_req_seq > requeue_from_seq:
                logger.debug(
                    "[LLMEnhancement] 跳过过期动态重排请求："
                    f"group={gid or 'private'}, sender_uid={uid}, state_uid={state_uid}, "
                    f"requeue_from_seq={requeue_from_seq}, current_req_seq={current_req_seq}"
                )
                event.stop_event()
                await self._release_concurrency_slot_if_needed(
                    event, reason="on_llm_request_stale_dynamic_requeue"
                )
                return
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
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_message_recalled"
            )
            return
        merge_delay = merge_cfg.delay_sec
        cache_keep_sec = max(max(merge_cfg.delay_sec, 10.0) * 6, 60.0)
        async with merge_member.lock:
            prune_member_msg_cache(merge_member, keep_sec=cache_keep_sec)
            if merge_delay > 0:
                raw_message = (
                    event.message_obj.raw_message
                    if (event.message_obj and hasattr(event.message_obj, "raw_message"))
                    else {}
                )
                if not raw_message and hasattr(event, "event"):
                    raw_message = event.event
                raw_msg_ts = _raw_get(raw_message, "time", None)
                if raw_msg_ts in (None, ""):
                    raw_msg_ts = _raw_get(raw_message, "date", None)
                event_ts = normalize_event_ts(raw_msg_ts, now)
                if dynamic_merge_mode:
                    if merge_member.merge_start_ts <= 0.0:
                        if merge_member.dynamic_unresolved_msgs:
                            merge_member.merge_start_ts = min(
                                float(item.get("ts", event_ts) or event_ts)
                                for item in merge_member.dynamic_unresolved_msgs
                            )
                        else:
                            merge_member.merge_start_ts = event_ts
                else:
                    merge_member.merge_start_ts = event_ts
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
                await self._release_concurrency_slot_if_needed(
                    event, reason="on_llm_request_skip_merged_dynamic"
                )
                return
            if not dynamic_merge_mode and current_msg_id and current_msg_id in merge_member.merged_msg_ids:
                event.stop_event()
                await self._release_concurrency_slot_if_needed(
                    event, reason="on_llm_request_skip_merged_static"
                )
                return

        message_chain = []
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            message_chain = event.message_obj.message or []
        if not msg and not message_chain:
            logger.debug(f"[LLMEnhancement] 忽略空消息事件: gid={gid or 'private'}, uid={uid}")
            event.stop_event()
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_empty_message"
            )
            return
        
        if (not dynamic_merge_mode) and merge_member.in_merging:
            logger.debug(f"[LLMEnhancement] 当前存在进行中的合并会话，跳过重复请求: gid={gid or 'private'}, uid={uid}")
            event.stop_event()
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_merge_in_progress"
            )
            return

        if gid and g.shutup_until > now:
            event.stop_event()
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_group_shutup"
            )
            return
        if not event.is_admin() and sender_member.silence_until > now:
            event.stop_event()
            await self._release_concurrency_slot_if_needed(
                event, reason="on_llm_request_user_silenced"
            )
            return

        has_preacquired_slot = bool(event.get_extra("_llme_concurrency_preacquired", default=False)) and bool(
            event.get_extra("_llme_concurrency_acquired", default=False)
        ) and (not bool(event.get_extra("_llme_concurrency_released", default=False)))
        if not has_preacquired_slot:
            logger.debug(
                "[LLMEnhancement] on_llm_request 并发判定跳过：未检测到 on_waiting_llm_request 预占位。"
            )

        request_forwarded = False
        try:
            # ==================== 1. 消息合并（前置，避免异步防护导致并发请求拆分） ====================
            if merge_delay <= 0:
                all_components = extract_merge_components(event)
                logger.debug(
                    "[LLMEnhancement] 消息合并已关闭："
                    f"gid={gid or 'private'}, uid={uid}, merge_delay={merge_delay}, dynamic_mode={merge_cfg.dynamic_mode}"
                )
            elif dynamic_merge_mode:
                all_components = await self._handle_message_merge_dynamic(event, req, gid, uid, merge_member)
            else:
                all_components = await self._handle_message_merge(event, req, gid, uid, merge_member)
            if merge_member.cancel_merge:
                event.stop_event()
                return

            direct_wake = bool(event.get_extra("_llme_direct_wake", default=False))
            wake_reason = str(event.get_extra("_llme_wake_reason", default="") or "")
            command_trigger_event = bool(event.get_extra("_llme_command_trigger_event", default=False))
            force_dynamic_followup = bool(event.get_extra("_llme_force_dynamic_followup", default=False))
            require_at_for_wake_prefix = bool(self._get_cfg("require_at_for_wake_prefix", False))
            prefix_wake_triggered = bool(event.get_extra("_llme_prefix_wake_triggered", default=False))
            addressed_to_bot = bool(event.get_extra("_llme_addressed_to_bot", default=False))
            if (
                require_at_for_wake_prefix
                and gid
                and direct_wake
                and prefix_wake_triggered
                and (not addressed_to_bot)
            ):
                logger.debug(
                    "[LLMEnhancement] on_llm_request 拦截：已启用唤醒前缀需@Bot，"
                    f"检测到仅前缀唤醒。group={gid or 'private'}, uid={uid}, reason={wake_reason}"
                )
                event.stop_event()
                return
            wake, wake_judge_detail = await apply_post_wake_judge_gate(
                event=event,
                msg=event.message_str,
                gid=gid,
                wake=bool(event.is_at_or_wake_command),
                wake_reason=wake_reason,
                command_trigger_event=command_trigger_event,
                direct_wake=direct_wake,
                force_dynamic_followup=force_dynamic_followup,
                get_cfg=self._get_cfg,
                get_history_msg=_get_wake_context_messages,
                find_provider=lambda provider_id: resolve_provider(self.context, provider_id),
            )
            if wake_judge_detail.startswith("model:") or wake_judge_detail.startswith("fallback_allow:"):
                logger.debug(
                    "[LLMEnhancement] on_llm_request 唤醒判定："
                    f"group={gid or 'private'}, uid={uid}, reason={wake_reason}, detail={wake_judge_detail}, wake={wake}"
                )
            if direct_wake and (not wake):
                logger.debug(
                    "[LLMEnhancement] on_llm_request 拦截：显式唤醒判定未通过，已取消本次 LLM 请求。"
                    f"group={gid or 'private'}, uid={uid}, reason={wake_reason}, detail={wake_judge_detail}"
                )
                event.stop_event()
                return

            # ==================== 2. 防护机制（使用合并后的文本） ====================
            msg = event.message_str
            event.set_extra(
                "_llme_effective_user_text",
                self._effective_dialog_history.build_user_text(msg, all_components),
            )
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

            empty_mention_ctx = str(event.get_extra("_llme_empty_mention_context", default="") or "").strip()
            if empty_mention_ctx:
                req.prompt = f"{(req.prompt or '').strip()}\n\n{empty_mention_ctx}".strip()

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

            injected_context, inject_detail, _injected_block = inject_context_into_request(
                req=req,
                group_state=g,
                get_cfg=self._get_cfg,
                direct_wake=direct_wake,
                wake_reason=wake_reason,
            )
            if injected_context:
                logger.debug(
                    "[LLMEnhancement][ContextInjection] 已注入上下文："
                    f"group={gid or 'private'}, uid={uid}, reason={wake_reason}, detail={inject_detail}"
                )

            injection_summary = {
                "perception": perception_injected,
                "member_info": sender_member_injected,
                "bot_member_info": bot_member_injected,
                "json": False,
                "file": False,
                "url": False,
                "forward": False,
                "video": False,
            }

            # 注册清理路径容器
            req._cleanup_paths = []

            dynamic_batch_msg_ids = event.get_extra("_llme_dynamic_batch_msg_ids", default=[]) or []
            if not dynamic_batch_msg_ids:
                await inject_current_message_image_context(
                    event=event,
                    req=req,
                )
            await inject_current_message_forward_origin_context(
                event=event,
                req=req,
            )
             
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
            injection_summary["url"] = bool(getattr(ref_result, "injected_url", False))
            if ref_result.blocked:
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
            request_forwarded = True
        
        finally:
            if not request_forwarded:
                await self._release_concurrency_slot_if_needed(
                    event, reason="request_blocked_before_provider"
                )
            if dynamic_merge_mode and event.is_stopped():
                dynamic_req_seq = int(event.get_extra("_llme_dynamic_request_seq") or 0)
                dynamic_batch_keys = event.get_extra("_llme_dynamic_batch_keys", default=[]) or []
                if dynamic_req_seq > 0:
                    async with merge_member.lock:
                        drop_dynamic_batch_from_unresolved(merge_member, dynamic_batch_keys)
                        if merge_member.dynamic_inflight_seq == dynamic_req_seq:
                            merge_member.dynamic_inflight_seq = 0
                            reset_dynamic_capture_session(merge_member)
                        if g.dynamic_owner_uid == state_uid:
                            g.dynamic_owner_uid = None
                    clear_pending_msg_ids(g, merge_member)
                    merge_member.cancel_merge = False

            # 6. 统一清理文件
            if hasattr(req, "_cleanup_paths"):
                await cleanup_paths_later(req._cleanup_paths)


    @filter.on_llm_response(priority=20)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """在 LLM 返回结果后执行，用于更新会话状态并处理动态丢弃回退。"""
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not uid:
            await self._release_concurrency_slot_if_needed(
                event, reason="llm_response_missing_uid"
            )
            return

        try:
            target_id = gid or f"private_{uid}"
            g = StateManager.get_group(target_id)
            context_injection_enabled = bool(gid) and is_context_injection_enabled(self._get_cfg)
            context_injection_max_messages = (
                get_context_injection_max_messages(self._get_cfg)
                if context_injection_enabled
                else 0
            )
            state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip()
            member = g.members.get(state_uid)

            if member:
                merge_cfg = load_merge_runtime_config(self._get_cfg)
                dynamic_merge_mode = bool(merge_cfg.dynamic_mode and merge_cfg.delay_sec > 0)
                cache_ttl_sec = get_discarded_response_ttl_sec(self._get_cfg, DYNAMIC_DISCARDED_RESPONSE_TTL_SEC)
                dynamic_req_seq = int(event.get_extra("_llme_dynamic_request_seq") or 0)
                is_dynamic_request = dynamic_merge_mode and dynamic_req_seq > 0
                dynamic_batch_keys = event.get_extra("_llme_dynamic_batch_keys", default=[]) or []
                if dynamic_merge_mode and dynamic_req_seq > 0:
                    should_drop = False
                    cache_stored = False
                    next_inflight_seq = 0
                    async with member.lock:
                        if dynamic_req_seq <= member.dynamic_discard_before_seq:
                            should_drop = True
                            cache_stored = store_discarded_response_cache(member, dynamic_req_seq, resp)
                        else:
                            drop_dynamic_batch_from_unresolved(member, dynamic_batch_keys)
                        if member.dynamic_inflight_seq == dynamic_req_seq:
                            member.dynamic_inflight_seq = 0
                            reset_dynamic_capture_session(member)
                        if g.dynamic_owner_uid == state_uid:
                            g.dynamic_owner_uid = None
                        next_inflight_seq = int(member.dynamic_inflight_seq or 0)

                    if should_drop:
                        has_newer_inflight = next_inflight_seq > dynamic_req_seq
                        if has_newer_inflight:
                            logger.debug(
                                "[LLMEnhancement] 动态合并放弃上次请求响应："
                                f"uid={state_uid}, sender_uid={uid}, group={gid or 'private'}, response_seq={dynamic_req_seq}, "
                                f"discard_before_seq={member.dynamic_discard_before_seq}, cache_stored={cache_stored}, "
                                f"next_inflight_seq={next_inflight_seq}"
                            )
                            clear_pending_msg_ids(g, member)
                            member.cancel_merge = False
                            event.set_extra("_llme_pending_last_response_update", False)
                            event.stop_event()
                            await self._release_concurrency_slot_if_needed(
                                event, reason="llm_response_discarded"
                            )
                            return
                        logger.warning(
                            "[LLMEnhancement] 动态丢弃保护触发：未检测到更新中的后续请求，保留当前响应以避免整轮无回复。"
                            f"uid={state_uid}, sender_uid={uid}, group={gid or 'private'}, response_seq={dynamic_req_seq}, "
                            f"discard_before_seq={member.dynamic_discard_before_seq}, cache_stored={cache_stored}"
                        )

                # 检查是否在此期间发生了撤回
                if member.cancel_merge:
                    logger.info(
                        " [LLMEnhancement] LLM 响应生成完成，但检测到消息已撤回，拦截回复 "
                        f"(state_uid: {state_uid}, sender_uid: {uid})。"
                    )
                    member.cancel_merge = False
                    clear_pending_msg_ids(g, member)
                    event.set_extra("_llme_pending_last_response_update", False)
                    event.set_extra("_llme_effective_assistant_text_candidate", "")
                    event.stop_event()
                    await self._release_concurrency_slot_if_needed(
                        event, reason="llm_response_cancelled_by_recall"
                    )
                    return

                event.set_extra("_llme_pending_last_response_update", True)
                event.set_extra("_llme_discard_cache_clear_after_sent", False)
                event.set_extra(
                    "_llme_effective_assistant_text_candidate",
                    self._effective_dialog_history.extract_assistant_text_from_response(resp),
                )
                # 清理待处理消息 ID
                clear_pending_msg_ids(g, member)
                member.cancel_merge = False

                is_err_resp = str(resp.role or "").lower() == "err"
                is_empty_resp = is_llm_response_empty_without_tool(resp)
                if is_dynamic_request and (is_err_resp or is_empty_resp):
                    await apply_discarded_response_fallback(
                        event=event,
                        member=member,
                        gid=gid,
                        uid=uid,
                        reason="llm_response_err" if is_err_resp else "llm_response_empty",
                        ttl_sec=cache_ttl_sec,
                    )
                else:
                    # 正常可发送内容，发送成功后清理缓存。
                    event.set_extra("_llme_discard_cache_clear_after_sent", True)

                if context_injection_enabled:
                    assistant_text = self._effective_dialog_history.extract_assistant_text_from_response(resp)
                    if not assistant_text:
                        assistant_text = str(resp.completion_text or "").strip()
                    if assistant_text and str(resp.role or "").lower() != "err":
                        append_group_context_message(
                            g,
                            uid=str(event.get_self_id() or "bot"),
                            sender_name="Bot",
                            message_text=assistant_text,
                            max_messages=context_injection_max_messages,
                            msg_id="",
                            is_bot=True,
                            source="assistant",
                            at_targets=[],
                            at_bot=False,
                            at_all=False,
                            reply_to_id=state_uid,
                            reply_msg_id="",
                            reply_preview="",
                            now_ts=time.time(),
                            get_cfg=self._get_cfg,
                        )
                        sender_name = event.get_sender_name()
                        g.context_bot_last_replied_to_uid = state_uid

        finally:
            await self._release_concurrency_slot_if_needed(
                event, reason="llm_response_finished"
            )

    @filter.on_decorating_result(priority=95)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """发送前兜底：若结果为空/报错，尝试回退到动态丢弃缓存。"""
        if not bool(event.get_extra("_llme_pending_last_response_update", default=False)):
            return

        uid: str = event.get_sender_id()
        if not uid:
            return
        gid: str = event.get_group_id()
        target_id = gid or f"private_{uid}"
        g = StateManager.get_group(target_id)
        state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip()
        member = g.members.get(state_uid)
        if not member:
            return
        dynamic_req_seq = int(event.get_extra("_llme_dynamic_request_seq") or 0)
        if dynamic_req_seq <= 0:
            return

        result = event.get_result()
        if result is None:
            return

        chain = list(result.chain or [])
        need_fallback = is_chain_effectively_empty(chain) or looks_like_error_result(chain)
        if not need_fallback:
            event.set_extra("_llme_discard_cache_clear_after_sent", True)
            return

        cache_ttl_sec = get_discarded_response_ttl_sec(self._get_cfg, DYNAMIC_DISCARDED_RESPONSE_TTL_SEC)
        used = await apply_discarded_response_fallback(
            event=event,
            member=member,
            gid=gid,
            uid=uid,
            reason="decorating_error_or_empty",
            ttl_sec=cache_ttl_sec,
        )
        if not used:
            event.set_extra("_llme_discard_cache_clear_after_sent", False)

    @filter.after_message_sent(priority=100)
    async def after_message_sent(self, event: AstrMessageEvent):
        """消息实际发送后再更新唤醒延长时间锚点。"""
        if not bool(event.get_extra("_llme_pending_last_response_update", default=False)):
            return

        uid: str = event.get_sender_id()
        if not uid:
            return
        state_uid = str(event.get_extra("_llme_dynamic_state_uid", default=uid) or uid).strip()
        should_clear_discard_cache = bool(event.get_extra("_llme_discard_cache_clear_after_sent", default=False))

        if not getattr(event, "_has_send_oper", False):
            logger.debug(
                "[LLMEnhancement] 跳过唤醒延长锚点更新：本次事件未实际发送消息。"
                f"uid={uid}, state_uid={state_uid}, group={event.get_group_id() or 'private'}"
            )
            event.set_extra("_llme_pending_last_response_update", False)
            event.set_extra("_llme_discard_cache_clear_after_sent", False)
            event.set_extra("_llme_effective_user_text", "")
            event.set_extra("_llme_effective_assistant_text_candidate", "")
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

        cleared_recent_wake_count = 0
        async with member.lock:
            cleared_recent_wake_count = len(member.recent_wake_msgs)
            member.recent_wake_msgs = []
            if should_clear_discard_cache:
                clear_discarded_response_cache(member)

        user_text = str(event.get_extra("_llme_effective_user_text", default="") or "")
        assistant_text = ""
        if not bool(event.get_extra("_llme_discard_cache_used", default=False)):
            assistant_text = str(event.get_extra("_llme_effective_assistant_text_candidate", default="") or "")
        if not assistant_text:
            assistant_text = self._effective_dialog_history.extract_assistant_text(event)
        if user_text and assistant_text:
            assistant_name = "bot"
            self._effective_dialog_history.append_turn(
                event.unified_msg_origin,
                user_text=user_text,
                assistant_text=assistant_text,
                user_name=event.get_sender_name(),
                assistant_name=assistant_name,
            )

        event.set_extra("_llme_pending_last_response_update", False)
        event.set_extra("_llme_discard_cache_clear_after_sent", False)
        event.set_extra("_llme_effective_user_text", "")
        event.set_extra("_llme_effective_assistant_text_candidate", "")
        logger.debug(
            "[LLMEnhancement] 已更新唤醒延长锚点："
            f"uid={uid}, state_uid={state_uid}, group={gid or 'private'}, ts={resp_ts:.3f}, "
            f"clear_discard_cache={should_clear_discard_cache}, cleared_recent_wake_count={cleared_recent_wake_count}"
        )

    async def terminate(self):
        async with self._request_counter_lock:
            self._active_request_refs.clear()
            self._active_request_meta.clear()
        await self.blacklist.terminate()
        logger.info("[LLMEnhancement] 插件已终止")
