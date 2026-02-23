import asyncio
import json
import time
import random
from typing import List, Any, Optional
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig 
import astrbot.api.message_components as Comp 
from astrbot.api.provider import LLMResponse, ProviderRequest 
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from .modules.sentiment import Sentiment
from .modules.similarity import Similarity
from .modules.state_manager import StateManager, MemberState
from .modules.forward_parser import process_reference_and_forward_content
from .modules.media_content_processor import process_media_content
from .modules.video_parser import (
    download_video_to_temp,
)
import os
import shutil
from .modules.info_utils import process_group_members_info, set_group_ban_logic
from .modules.provider_utils import find_provider
from .modules.blacklist_bridge import is_user_blacklisted_via_blacklist_plugin
from .modules.wake_extend import evaluate_wake_extend
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

@register("llm_enhancement", "Foolllll", "增强LLM的综合表现", "1.0.0") 
class LLMEnhancement(Star): 
    def __init__(self, context: Context, config: AstrBotConfig): 
        super().__init__(context) 
        self.config = config
        self.cfg = {}
        self._refresh_config()
        self.sent = Sentiment()
        self.similarity = Similarity()
        logger.info(f"[LLMEnhancement] 插件初始化完成。IS_AIOCQHTTP: {IS_AIOCQHTTP}")

    def _refresh_config(self):
        """将 object 格式的配置平铺到 self.cfg 中"""
        self.cfg = {}
        # 1. 获取顶级配置项
        for k in ["group_whitelist", "group_blacklist", "user_blacklist"]:
            self.cfg[k] = self.config.get(k)
        
        # 2. 平铺对象配置
        for section in [
            "intelligent_wake",
            "parse_switches",
            "video_injection",
            "forward_parsing",
            "file_parsing",
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

    # ==================== 唤醒消息级别 ====================
    
    @filter.event_message_type(filter.EventMessageType.ALL, priority=1)
    async def on_group_msg(self, event: AstrMessageEvent):
        """处理消息的初步过滤、黑白名单检查及唤醒逻辑。支持群聊和私聊。"""
        self._refresh_config()
        raw_message = event.message_obj.raw_message if (event.message_obj and hasattr(event.message_obj, "raw_message")) else {}
        if not raw_message and hasattr(event, "event") and isinstance(event.event, dict):
            raw_message = event.event
        if isinstance(raw_message, dict) and raw_message.get("post_type") not in (None, "", "message"):
            return
        bid: str = event.get_self_id()
        gid: str = event.get_group_id() # 私聊下为 None
        uid: str = event.get_sender_id()
        msg: str = event.message_str.strip() if event.message_str else ""
        
        g = StateManager.get_group(gid or f"private_{uid}")

        # 1. 全局屏蔽检查
        if uid == bid:
            return
        
        # 1.0 黑名单插件拦截
        if await is_user_blacklisted_via_blacklist_plugin(uid):
            event.stop_event()
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
            
        u_blacklist = self._get_cfg("user_blacklist")
        if u_blacklist and uid in u_blacklist:
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

        # 特殊处理：空 @ Bot 消息（仅群聊）
        if wake and not msg and gid:
            msg = f"{event.get_sender_name()}@了你"
            event.message_str = msg
            reason = "空@唤醒"

        if not msg:
            if not wake:
                return

        # 提及唤醒 (仅群聊)
        if gid and not wake:
            mention_wake = self._get_cfg("mention_wake")
            if mention_wake:
                names = [n for n in mention_wake if n]
                for n in names:
                    if n and n in msg:
                        wake = True
                        reason = f"提及唤醒({n})"
                        break

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
        forbidden_words = self._get_cfg("wake_forbidden_words")
        if forbidden_words:  
            for word in forbidden_words:
                if not event.is_admin() and word in event.message_str:
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
        additional_components: List[Any] = []
        for item in preselected_snapshots:
            for seg in item.get("components", []) or []:
                if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File)):
                    additional_components.append(seg)

        @session_waiter(timeout=merge_delay, record_history_chains=False)
        async def collect_messages(controller: SessionController, ev: AstrMessageEvent):
            nonlocal message_buffer, additional_components
            # 撤回事件实时监听已停用，统一在最终阶段做 get_msg 校验。

            if member.cancel_merge:
                controller.stop()
                return

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
            if not ev.message_str and not any(isinstance(seg, (Comp.Image, Comp.Video, Comp.File, Comp.Forward)) for seg in ev.message_obj.message):
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
                if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File)):
                    additional_components.append(seg)
            controller.keep(timeout=merge_delay, reset_timeout=True)
        
        try:
            await collect_messages(event)
        except TimeoutError:
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

    @filter.llm_tool(name="get_user_avatar")
    async def get_user_avatar(self, event: AstrMessageEvent, user_id: str) -> str:
        """
        获取指定 QQ 用户的头像并将其作为图片附件注入到当前对话中。
        当你需要识别、描述某个人头像特征，或者用户明确要求“看看某人的头像”时使用。

        Args:
            user_id (str): 目标用户的 QQ 号。必须是纯数字字符串。
        """
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
        支持在群聊中直接使用，或在私聊中指定 group_id 使用。
        
        Args:
            user_id (str): 目标用户的 QQ 号。必须是纯数字字符串。
            duration (int): 禁言时长（秒）。0 为解禁；60-600 为警告级；3600-86400 为惩罚级；最大为 2592000 (30天)。请根据违规严重程度灵活选择。
            user_name (str): 目标用户的昵称或称呼，用于回复确认。
            group_id (str, optional): 目标 QQ 群号。在私聊使用时必填，在群聊使用时可选（默认当前群）。
        """
        return await set_group_ban_logic(event, user_id, duration, user_name, group_id)

    # ==================== LLM 请求级别逻辑 ====================

    @filter.on_llm_request(priority=15)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在请求发送给 LLM 前执行，处理防护、合并及多媒体注入。"""
        setattr(event, "_provider_req", req)
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not uid:
            return
        
        g = StateManager.get_group(gid or f"private_{uid}")
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        member = g.members[uid]
        now = time.time()
        msg = event.message_str
        current_msg_id = get_event_msg_id(event)
        merge_delay = float(self._get_cfg("merge_delay", 10.0) or 10.0)
        cache_keep_sec = max(merge_delay * 6, 60.0)
        async with member.lock:
            prune_member_msg_cache(member, keep_sec=cache_keep_sec)
            if current_msg_id and current_msg_id in member.merged_msg_ids:
                event.stop_event()
                return

        # 协议端偶发空事件（常见于 notice/回执）不应进入 LLM 请求流程
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
            
            # ==================== 3. 引用消息/转发消息内容提取 ====================
            ref_result = await process_reference_and_forward_content(
                event=event,
                req=req,
                all_components=all_components,
                get_cfg=self._get_cfg,
                get_stt_provider=lambda ev: self._get_stt_provider(event=ev),
                cleanup_paths=self._cleanup_paths,
                download_media=download_video_to_temp,
            )
            if ref_result.blocked:
                event.stop_event()
                return
            reply_seg = ref_result.reply_seg
            if ref_result.handled_forward:
                return
            
            # ==================== 4. 媒体场景检测与处理 ====================
            await process_media_content(
                context=self.context,
                event=event,
                req=req,
                all_components=all_components,
                reply_seg=reply_seg,
                get_cfg=self._get_cfg,
            )
        
        finally:
            # 5. 统一清理文件
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
        logger.info("[LLMEnhancement] 插件已终止")
