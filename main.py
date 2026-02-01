import asyncio
import json
import time
import random
from typing import List, Dict, Any, Optional, Tuple 
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.core.star.filter.platform_adapter_type import PlatformAdapterType
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig 
import astrbot.api.message_components as Comp 
from astrbot.api.provider import LLMResponse, ProviderRequest 
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from .modules.sentiment import Sentiment
from .modules.similarity import Similarity
from .modules.state_manager import StateManager, MemberState, GroupState
from .modules.forward_parser import extract_forward_content
from .modules.video_parser import (
    extract_videos_from_chain, 
    extract_forward_video_keyframes, 
    probe_duration_sec,
    download_video_to_temp,
    extract_audio_wav,
    MediaScenario,
    MediaContext,
    VideoFrameProcessor,
    is_gif_file
)
import os
import shutil
import aiosqlite
from datetime import datetime
from pathlib import Path
from .modules.info_utils import process_group_members_info, set_group_ban_logic
try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False

# ==================== 黑名单插件桥接 ====================

def _get_blacklist_db_path() -> Optional[Path]:
    """获取黑名单插件数据库路径"""
    try:
        target_data_dir = StarTools.get_data_dir("astrbot_plugin_blacklist_tools")
        db_path = Path(target_data_dir) / "blacklist.db"
        
        if db_path.exists():
            return db_path
        else:
            logger.debug(f"[LLMEnhancement] 未找到黑名单数据库。目标路径: {db_path}")
    except Exception as e:
        logger.error(f"[LLMEnhancement] 获取黑名单插件数据目录失败: {e}")
    return None

async def is_user_blacklisted_via_blacklist_plugin(user_id: str) -> bool:
    """通过黑名单插件查询用户是否在黑名单中"""
    db_path = _get_blacklist_db_path()
    if db_path is None:
        return False
        
    try:
        async with aiosqlite.connect(str(db_path)) as db:
            cursor = await db.execute(
                "SELECT * FROM blacklist WHERE user_id = ?", 
                (user_id,)
            )
            user = await cursor.fetchone()
            if user:
                # user[2] 是 expire_time
                expire_time_str = user[2]
                if expire_time_str:
                    try:
                        expire_datetime = datetime.fromisoformat(expire_time_str)
                        if datetime.now() > expire_datetime:
                            # 已过期，逻辑上不视为黑名单（由原插件负责删除）
                            return False
                        else:
                            logger.info(f"[LLMEnhancement] 用户 {user_id} 在黑名单中 (到期时间: {expire_time_str})")
                            return True
                    except ValueError:
                        return True
                else:
                    # expire_time 为空表示永久黑名单
                    logger.info(f"[LLMEnhancement] 用户 {user_id} 在永久黑名单中")
                    return True
            return False
    except Exception as e:
        logger.error(f"[LLMEnhancement] 查询 blacklist 插件数据库失败: {e}")
        return False

# ==================== 常量定义 ====================

# AstrBot 内置指令列表
BUILT_CMDS = [
    "llm", "t2i", "tts", "sid", "op", "wl",
    "dashboard_update", "alter_cmd", "provider", "model",
    "plugin", "plugin ls", "new", "switch", "rename",
    "del", "reset", "history", "persona", "tool ls",
    "key", "websearch", "help",
]

MAX_MERGE_MESSAGES = 15 # 最大合并消息数


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
        """将 template_list 格式的配置平铺到 self.cfg 中"""
        for k in ["group_whitelist", "group_blacklist", "user_blacklist"]:
            self.cfg[k] = self.config.get(k)
        
        modules = self.config.get("modules", [])
        if isinstance(modules, list):
            for item in modules:
                template_key = item.get("__template_key")
                if not template_key:
                    continue
                
                for k, v in item.items():
                    if k != "__template_key":
                        self.cfg[k] = v

    def _get_cfg(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        if key in self.cfg:
            return self.cfg[key]
        return self.config.get(key, default)

    # ==================== 唤醒消息级别 ====================
    
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1)
    async def on_group_msg(self, event: AstrMessageEvent):
        """处理群消息的初步过滤、黑白名单检查及唤醒逻辑。"""
        self._refresh_config()
        bid: str = event.get_self_id()
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        msg: str = event.message_str.strip() if event.message_str else ""
        g = StateManager.get_group(gid)

        # 1. 全局屏蔽检查
        if uid == bid: return
        
        # 1.0 黑名单插件拦截
        if await is_user_blacklisted_via_blacklist_plugin(uid):
            event.stop_event()
            return
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

        # 特殊处理：空 @ 机器人消息
        empty_pt = self._get_cfg("empty_mention_pt")
        if wake and not msg and empty_pt:
            prompt = empty_pt.format(username=event.get_sender_name())
            yield event.request_llm(prompt=prompt)
            return

        if not msg:
            if not wake:
                return

        # 提及唤醒
        mention_wake = self._get_cfg("mention_wake")
        if not wake and mention_wake:
            names = [n for n in mention_wake if n]
            for n in names:
                if n and n in msg:
                    wake = True
                    reason = f"提及唤醒({n})"
                    break

        # 唤醒延长
        wake_extend = self._get_cfg("wake_extend")
        if (not wake and wake_extend and 
            (now - member.last_response) <= int(wake_extend or 0)):
            if bmsgs := await self._get_history_msg(event, count=3):
                simi = self.similarity.similarity(gid, msg, bmsgs)
                if simi > 0.3:
                    wake = True
                    reason = f"唤醒延长(相关性{simi:.2f})"

        # 话题相关性唤醒
        relevant_wake = self._get_cfg("relevant_wake")
        if not wake and relevant_wake:
            if bmsgs := await self._get_history_msg(event, count=5):
                simi = self.similarity.similarity(gid, msg, bmsgs)
                if simi > relevant_wake:
                    wake = True
                    reason = f"话题相关性{simi:.2f}>{relevant_wake}"

        # 答疑唤醒
        ask_wake = self._get_cfg("ask_wake")
        if not wake and ask_wake:  
            if self.sent.ask(msg) > ask_wake:
                wake = True
                reason = "答疑唤醒"

        # 概率唤醒
        prob_wake = self._get_cfg("prob_wake")
        if not wake and prob_wake:  
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
            logger.info(f"群({gid})用户({uid}) {reason}: {msg[:50]}")

    @filter.platform_adapter_type(PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def on_notice(self, event: AstrMessageEvent):
        """处理通知事件，如消息撤回。"""
        raw_message = event.message_obj.raw_message if (event.message_obj and hasattr(event.message_obj, "raw_message")) else {}
        if not raw_message and hasattr(event, "event") and isinstance(event.event, dict):
            raw_message = event.event

        if not raw_message:
            return
            
        post_type = raw_message.get("post_type")
        notice_type = raw_message.get("notice_type")
        
        if post_type == "notice" and notice_type == "group_recall":
            gid = str(raw_message.get("group_id", ""))
            msg_id = str(raw_message.get("message_id", ""))
            
            if gid and msg_id:
                g = StateManager.get_group(gid)
                # 遍历该群所有成员
                found = False
                for member in g.members.values():
                    # 增加调试日志
                    if member.pending_msg_ids:
                        logger.debug(f" [LLMEnhancement] 正在检查用户 {member.uid} 的待处理消息: {member.pending_msg_ids}")
                    
                    # 只有当被撤回的是“触发消息”（第一条）时，才取消整个合并/请求流程
                    trigger_id = getattr(member, "trigger_msg_id", None)
                    if trigger_id and msg_id == trigger_id:
                        member.cancel_merge = True
                        found = True
                        logger.info(f" [LLMEnhancement] 检测到【触发消息】撤回 (ID: {msg_id})，正在取消用户 {member.uid} 的合并/请求流程。")
                    
                    # 如果是后续消息被撤回，on_notice 不做处理，交给 collect_messages 在合并窗口期处理
                    # 如果已经过了合并窗口期（即在 LLM 生成中），后续消息撤回无法影响已发送的 prompt，故忽略
                
                if not found:
                    logger.debug(f" [LLMEnhancement] 收到撤回事件 (ID: {msg_id})，但未匹配到任何正在进行的请求（或非触发消息）。")
    
    # ==================== 消息合并处理 ====================
    
    async def _handle_message_merge(self, event: AstrMessageEvent, req: ProviderRequest, gid: str, uid: str, member: MemberState) -> List[Any]:
        """执行消息合并逻辑，根据配置决定是否收集多用户消息并格式化。"""
        # 初始化状态
        member.pending_msg_ids.clear()
        member.cancel_merge = False
        member.trigger_msg_id = None
        
        # 获取初始消息 ID
        current_msg_id = None
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message_id"):
            current_msg_id = str(event.message_obj.message_id)
            member.pending_msg_ids.add(current_msg_id)
            member.trigger_msg_id = current_msg_id # 记录触发消息 ID

        # buffer 结构: List[Tuple[msg_id, sender_name, message_str]]
        # 注意：这里改为存储三元组，以便后续按 ID 删除
        message_buffer = [(current_msg_id, event.get_sender_name(), event.message_str)]
        additional_components = [seg for seg in event.message_obj.message if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File))]
        
        merge_delay = self._get_cfg("merge_delay")
        allow_multi_user = self._get_cfg("merge_multi_user")

        @session_waiter(timeout=merge_delay, record_history_chains=False)
        async def collect_messages(controller: SessionController, ev: AstrMessageEvent):
            nonlocal message_buffer, additional_components
            
            # 识别撤回事件
            raw = ev.message_obj.raw_message if (ev.message_obj and hasattr(ev.message_obj, "raw_message")) else {}
            if not raw and hasattr(ev, "event") and isinstance(ev.event, dict): raw = ev.event
            
            if raw.get("post_type") == "notice" and raw.get("notice_type") == "group_recall":
                recalled_msg_id = str(raw.get("message_id"))
                
                # 情况 1: 撤回的是第一条消息（触发消息）
                if recalled_msg_id == member.trigger_msg_id:
                    member.cancel_merge = True
                    logger.info(f" [LLMEnhancement] 合并期间检测到【触发消息】撤回 (ID: {recalled_msg_id})，取消合并。")
                    controller.stop()
                    return

                # 情况 2: 撤回的是后续消息
                # 从 buffer 中查找并移除
                for i, (mid, name, content) in enumerate(message_buffer):
                    if mid == recalled_msg_id:
                        logger.info(f" [LLMEnhancement] 合并期间检测到后续消息撤回 (ID: {recalled_msg_id})，已从合并列表中移除。")
                        message_buffer.pop(i)
                        # 同时也从 pending_msg_ids 中移除
                        if recalled_msg_id in member.pending_msg_ids:
                            member.pending_msg_ids.remove(recalled_msg_id)
                        # 刷新等待时间，继续收集
                        controller.keep(timeout=merge_delay, reset_timeout=True)
                        return
                
                return # 其他消息撤回，忽略

            # 检查是否已取消
            if member.cancel_merge:
                controller.stop()
                return

            # 基础检查：必须是同个群
            if ev.get_group_id() != gid:
                return

            # 如果不允许跨用户合并，则检查发送者是否一致
            if not allow_multi_user and ev.get_sender_id() != uid:
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

            if len(message_buffer) >= MAX_MERGE_MESSAGES:
                controller.stop()
                return
            
            # 记录消息 ID
            if new_msg_id:
                member.pending_msg_ids.add(new_msg_id)

            message_buffer.append((new_msg_id, ev.get_sender_name(), ev.message_str))
            for seg in ev.message_obj.message:
                if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File)):
                    additional_components.append(seg)

            controller.keep(timeout=merge_delay, reset_timeout=True)
        
        try:
            await collect_messages(event)
        except TimeoutError:
            pass # 继续后续处理
            
        # 无论是否超时，如果已取消，直接返回
        if member.cancel_merge:
            logger.info(f" [LLMEnhancement] 合并流程已因消息撤回而取消 (用户: {uid})")
            event.stop_event()
            return []

        if len(message_buffer) > 0: # 改为 > 0，因为如果只有一条但没被撤回也需要处理；如果全被撤回了 len=0
            # 检查是否涉及多个用户
            senders = set(name for _, name, _ in message_buffer)
            
            if len(senders) > 1:
                # 多人模式：[用户A]: 消息1 \n [用户B]: 消息2
                merged_msg = "\n".join([f"[{name}]: {msg}" for _, name, msg in message_buffer])
            else:
                # 单人模式：直接空格连接
                merged_msg = " ".join([msg for _, _, msg in message_buffer])

            event.message_str = merged_msg
            req.prompt = merged_msg
            logger.info(f"合并：用户({uid})触发，共合并了{len(message_buffer)}条消息 (涉及{len(senders)}人)")
        else:
            # 如果所有消息都被撤回了
            logger.info(f" [LLMEnhancement] 所有消息均已被撤回，取消请求。")
            event.stop_event()
            return []
        
        # 注意：finally 块已移除，因为状态清理已移动到 on_llm_response
        return additional_components

    # ==================== LLM 工具注册 ====================

    @filter.llm_tool(name="get_user_avatar")
    async def get_user_avatar(self, event: AstrMessageEvent, user_id: str) -> str:
        """
        获取指定 QQ 号的头像并注入到当前对话中。
        当用户要求查看某个人的头像，或者需要根据头像信息进行识别/描述时调用。

        Args:
            user_id (str): 用户的 QQ 号。
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
    async def get_group_members(self, event: AstrMessageEvent) -> str:
        """
        获取QQ群成员信息的LLM工具。
        需要判断是否为群聊时，以及当需要知道群里有哪些人，或者需要获取他们的昵称和用户ID，
        或者需要知道群里是否有特定成员时，调用此工具。其中display_name是“群昵称”，username是用户“QQ名”
        获取数据之后需要联系上下文，用符合prompt的方式回答用户的问题。
        """
        return await process_group_members_info(event)

    @filter.llm_tool(name="set_group_ban")
    async def set_group_ban(self, event: AstrMessageEvent, user_id: str, duration: int, user_name: str) -> str:
        """
        在群聊中禁言某用户。被禁言的用户在禁言期间将无法发送消息。
        
        Args:
            user_id (str): 被禁言用户的 QQ 号。
            duration (int): 禁言时长（秒），0 表示取消禁言。
            user_name (str): 被禁言用户的昵称。
        """
        return await set_group_ban_logic(event, user_id, duration, user_name)

    # ==================== LLM 请求级别逻辑 ====================
    
    async def _detect_media_scenario(self, req: ProviderRequest, video_sources: List[str] = None, reply_seg: Optional[Comp.Reply] = None) -> MediaContext:
        """
        【前置判断】检测当前是什么媒体处理场景
        返回 MediaContext，包含场景标记和相关信息
        """
        ctx = MediaContext()
        image_urls:  Any = getattr(req, "image_urls", None)
        
        # 优先判断是否有视频来源
        if video_sources and len(video_sources) > 0:
            ctx.media_path = video_sources[0]
            # 后续会进行时长探测和进一步分类
        elif image_urls and isinstance(image_urls, list) and len(image_urls) > 0:
            ctx.media_path = image_urls[0]
        else:
            ctx.scenario = MediaScenario.NONE
            return ctx
            
        first_path = ctx.media_path
        if not first_path or not isinstance(first_path, str):
            ctx.scenario = MediaScenario.NONE
            logger.warning("[场景判断] → 无效媒体（非字符串）")
            return ctx
        
        # 远程 URL → 尝试下载到本地
        if first_path.startswith(("http://", "https://")):
            logger.debug(f"[场景判断] 检测到远程 URL，尝试下载: {first_path}")
            try:
                local_path = await download_video_to_temp(first_path, 20)
                if local_path: 
                    ctx.media_path = local_path
                    ctx.cleanup_paths.append(local_path)
                    first_path = local_path
                    logger.info(f"[场景判断] 下载成功: {local_path}")
                else:
                    ctx.scenario = MediaScenario.NONE
                    # 具体原因由 download_video_to_temp 内部打印
                    return ctx
            except Exception as e:  
                logger.warning(f"[场景判断] 下载过程抛出异常: {e}")
                ctx.scenario = MediaScenario.NONE
                return ctx
        
        ctx.media_path = first_path
        
        # ========== 场景分类 ==========
        # 1. 首先通过 is_gif_file 判断是否为纯 GIF 流程
        if is_gif_file(first_path):
            ctx.duration = await self._probe_duration_helper(first_path)
            if ctx.duration <= 0:
                ctx.scenario = MediaScenario.NONE
                logger.info(f"[场景判断] → GIF 探测时长为 0s，视为静态图片，跳过增强处理")
                return ctx
            ctx.scenario = MediaScenario.GIF_ANIMATED
            logger.info(f"[场景判断] → GIF 动图 (时长: {ctx.duration:.2f}s)")
            return ctx
            
        # 2. 判断是否为视频流程 (含 MP4 格式的动图)
        is_from_video_source = video_sources and len(video_sources) > 0 and ctx.media_path == video_sources[0]
        suffix = Path(first_path).suffix.lower()
        
        # 检查是否为视频后缀或魔数
        is_video_format = suffix in [".mp4", ".mov", ".avi", ".wmv", ".flv", ".m4v"] or is_from_video_source
        if not is_video_format:
            try:
                with open(first_path, "rb") as f:
                    header = f.read(32)
                    if b"ftyp" in header or b"matroska" in header or b"fLaC" in header:
                        is_video_format = True
            except: pass

        if is_video_format:
            ctx.duration = await self._probe_duration_helper(first_path)
            
            # 关键修复：只要时长探测为 0，就判定为普通图片，不论其来源
            if ctx.duration <= 0:
                ctx.scenario = MediaScenario.NONE
                logger.info(f"[场景判断] → 探测时长为 0s，判定为普通图片，跳过增强处理")
                return ctx
                
            ctx.scenario = MediaScenario.VIDEO
            logger.info(f"[场景判断] → 视频 (统一抽帧流程, 时长: {ctx.duration:.2f}s)")
        else:
            # 3. 兜底进入 NONE 流程（静态图片不走增强解析）
            ctx.scenario = MediaScenario.NONE
            logger.info(f"[场景判断] → 静态图片或未知格式 (后缀: {suffix})，跳过增强处理")
        
        return ctx
    
    async def _probe_duration_helper(self, media_path: str) -> float:
        """异步版本的时长探测"""
        try: 
            duration = await asyncio.to_thread(
                probe_duration_sec,
                self._get_cfg("ffmpeg_path", ""),
                media_path
            ) or 0
            return duration
        except Exception as e:  
            logger.warning(f"[探测时长] 异常:  {e}")
            return 0

    # ==================== LLM 请求级别逻辑 ====================

    @filter.on_llm_request(priority=15)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在请求发送给 LLM 前执行，处理防护、合并及多媒体注入。"""
        setattr(event, "_provider_req", req)
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not gid or not uid:
            return
        
        g = StateManager.get_group(gid)
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        member = g.members[uid]
        now = time.time()
        msg = event.message_str
        
        if member.in_merging:
            return

        # ==================== 1. 防护机制 ====================
        shutup_th_cfg = self._get_cfg("shutup")
        if shutup_th_cfg:
            shut_th = self.sent.shut(msg)
            if shut_th > shutup_th_cfg:
                silence_sec = shut_th * self._get_cfg("silence_multiple", 500)
                g.shutup_until = now + silence_sec
                logger.info(f"群({gid})触发闭嘴，沉默{silence_sec:.1f}秒")
                event.stop_event()
                return

        insult_th_cfg = self._get_cfg("insult")
        if insult_th_cfg:
            insult_th = self.sent.insult(msg)
            if insult_th > insult_th_cfg:
                silence_sec = insult_th * self._get_cfg("silence_multiple", 500)
                member.silence_until = now + silence_sec
                logger.info(f"用户({uid})触发辱骂沉默{silence_sec:.1f}秒(下次生效)")

        if g.shutup_until > now:
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
            # ==================== 2. 消息合并 ====================
            all_components = await self._handle_message_merge(event, req, gid, uid, member)
            if member.cancel_merge:
                event.stop_event()
                return

            # 注册清理路径容器
            req._cleanup_paths = []
            
            # ==================== 3. 引用消息/转发消息内容提取 ====================
            forward_id = None 
            reply_seg = None 
            json_extracted_texts = []

            if IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent):
                for seg in all_components:
                    if isinstance(seg, Comp.Forward):
                        forward_id = seg.id 
                        break
                    elif isinstance(seg, Comp.Reply):
                        reply_seg = seg 
            
                if event.is_at_or_wake_command:  
                    if not forward_id and reply_seg:
                        try:
                            client = event.bot
                            original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
                            if original_msg and 'message' in original_msg:
                                # 提取发送者信息
                                sender_info = original_msg.get("sender", {})
                                original_sender = str(sender_info.get("user_id", ""))
                                original_sender_name = sender_info.get("nickname", "未知用户")
                                self_id = str(event.get_self_id())
                                block_prob = self._get_cfg("quote_self_video_block_prob", 0)
                                
                                if original_sender == self_id and block_prob > 0:
                                    has_video = any(s.get("type") == "video" for s in original_msg['message'])
                                    if has_video:
                                        if random.random() < block_prob:
                                            logger.info(f"[LLMEnhancement] 检测到引用了 Bot 自身发送的视频，且命中屏蔽概率 ({block_prob})，已拦截请求。")
                                            event.stop_event()
                                            return

                                # 记录引用消息的发送者，用于后续注入
                                setattr(req, "_quoted_sender", original_sender_name)

                                original_message_chain = original_msg['message']
                                if isinstance(original_message_chain, list):
                                    has_extracted_media = False
                                    for segment in original_message_chain:  
                                        seg_type = segment.get("type")
                                        if seg_type == "forward":
                                            forward_id = segment.get("data", {}).get("id")
                                            if forward_id:
                                                break
                                        elif seg_type == "image":
                                            url = segment.get("data", {}).get("url")
                                            if url:
                                                has_extracted_media = True
                                                if not hasattr(req, "image_urls") or req.image_urls is None:
                                                    req.image_urls = []
                                                
                                                if url.startswith(("http://", "https://")):
                                                    try:
                                                        local_path = await download_video_to_temp(url, 20)
                                                        if local_path:  
                                                            if local_path not in req.image_urls:
                                                                req.image_urls.append(local_path)
                                                                logger.debug(f"[LLMEnhancement] 从引用消息中提取并下载媒体: {url[:50]}... -> {local_path}")
                                                                req._cleanup_paths.append(local_path)
                                                            continue
                                                    except Exception as e:
                                                        logger.warning(f"下载引用消息图片失败: {e}")
                                                
                                                if url not in req.image_urls:
                                                    req.image_urls.append(url)
                                                    logger.debug(f"[LLMEnhancement] 从引用消息中提取到媒体 URL: {url}")
                                        elif seg_type == "json":
                                            try:  
                                                inner_data_str = segment.get("data", {}).get("data")
                                                if inner_data_str:
                                                    inner_data_str = inner_data_str.replace("&#44;", ",")
                                                    inner_json = json.loads(inner_data_str)
                                                    if inner_json.get("app") == "com.tencent.multimsg" and inner_json.get("config", {}).get("forward") == 1:
                                                        news_items = inner_json.get("meta", {}).get("detail", {}).get("news", [])
                                                        for item in news_items:
                                                            text_content = item.get("text")
                                                            if text_content:
                                                                clean_text = text_content.strip().replace("[图片]", "").strip()
                                                                if clean_text:
                                                                    json_extracted_texts.append(clean_text)
                                                        if json_extracted_texts:
                                                            break
                                            except:  
                                                continue
                                    
                                    if has_extracted_media:
                                        quoted_sender = getattr(req, "_quoted_sender", "未知用户")
                                        # 注入到 prompt 中，告知 LLM 这些图片是谁发的
                                        context_desc = f"\n\n[补充信息] 以上图片提取自用户 {quoted_sender} 的引用消息，请在回复时参考该上下文。"
                                        req.prompt += context_desc
                                        logger.debug(f"[LLMEnhancement] 已为引用图片注入发送者信息: {quoted_sender}")
                                    
                                    # 如果有视频
                                    has_video_in_quote = any(s.get("type") == "video" for s in original_message_chain)
                                    if has_video_in_quote:
                                        quoted_sender = getattr(req, "_quoted_sender", "未知用户")
                                        logger.debug(f"[LLMEnhancement] 检测到引用视频，已记录发送者信息: {quoted_sender}")
                        except Exception as e:
                            logger.warning(f"获取被回复消息详情失败: {e}")
                    
                    # 【转发消息处理】
                    if forward_id or json_extracted_texts:
                        image_urls = []
                        forward_video_sources = []
                        try:
                            if forward_id:
                                extracted_texts, image_urls, forward_video_sources = await extract_forward_content(event.bot, forward_id)
                            else:
                                extracted_texts = json_extracted_texts
                            
                            if extracted_texts or image_urls or forward_video_sources:
                                # 处理转发消息中的视频关键帧
                                if forward_video_sources and self._get_cfg("forward_video_keyframe_enable", True):
                                    f_frames, f_cleanup, f_local_videos = await extract_forward_video_keyframes(
                                        event,
                                        forward_video_sources,
                                        max_count=self._get_cfg("forward_video_max_count", 2),
                                        ffmpeg_path=self._get_cfg("ffmpeg_path", ""),
                                        max_mb=self._get_cfg("video_max_size_mb", 50),
                                        max_duration=7200, # 硬编码安全限制 120 分钟
                                        timeout_sec=10
                                    )
                                    if f_frames:
                                        image_urls.extend(f_frames)
                                        
                                        # 处理转发视频的 ASR
                                        if self._get_cfg("video_asr_enable", True) and f_local_videos:  
                                            for idx, lv_path in enumerate(f_local_videos):
                                                try:
                                                    wav_path = await extract_audio_wav(self._get_cfg("ffmpeg_path", ""), lv_path)
                                                    if wav_path and os.path.exists(wav_path):
                                                        stt = self._get_stt_provider()
                                                        
                                                        if stt:  
                                                            f_asr_text = None
                                                            try:
                                                                if hasattr(stt, "get_text"):
                                                                    f_asr_text = await stt.get_text(wav_path)
                                                                elif hasattr(stt, "speech_to_text"):
                                                                    res = await stt.speech_to_text(wav_path)
                                                                    f_asr_text = res.get("text", "") if isinstance(res, dict) else str(res)
                                                                
                                                                if f_asr_text:
                                                                    extracted_texts.append(f" [视频语音转写 {idx+1}] {f_asr_text}")
                                                                    logger.info(f"转发视频 {idx+1} ASR 成功")
                                                            except:  
                                                                pass
                                                        
                                                        try:
                                                            os.remove(wav_path)
                                                        except: 
                                                            pass
                                                except:  
                                                    pass

                                        if f_cleanup:
                                            async def cleanup_f():
                                                await asyncio.sleep(60)
                                                for p in f_cleanup:
                                                    try:
                                                        if os.path.isdir(p):
                                                            shutil.rmtree(p)
                                                        elif os.path.isfile(p):
                                                            os.remove(p)
                                                    except:  
                                                        pass
                                            asyncio.create_task(cleanup_f())

                                chat_records = "\n".join(extracted_texts)
                                user_question = req.prompt.strip() or "请总结一下这个聊天记录"
                                
                                # 获取转发来源信息
                                quoted_sender = getattr(req, "_quoted_sender", "未知用户")
                                
                                context_prompt = (
                                    f"\n\n用户提供了由 {quoted_sender} 发送/转发的聊天记录，请根据这些记录内容来响应。聊天记录如下：\n"
                                    f"--- 聊天记录开始 ---\n"
                                    f"{chat_records}\n"
                                    f"--- 聊天记录结束 ---"
                                )
                                req.prompt = user_question + context_prompt
                                req.image_urls.extend(image_urls)
                                logger.info(f"[转发消息] 成功注入转发内容 ({len(extracted_texts)} 条文本, {len(image_urls)} 张图片)")
                                
                                # ✅ 转发消息已处理完毕，跳过后续所有视频/GIF 处理
                                await self._cleanup_paths(req._cleanup_paths)
                                return
                        except Exception as e:
                            logger.warning(f"内容提取失败: {e}")
            
            # ==================== 4. 【分支】媒体场景检测与处理 ====================
            if self._get_cfg("video_detect_enable", True):
                # 尝试从当前消息或引用消息获取视频
                video_sources = extract_videos_from_chain(all_components)
                
                if not video_sources and reply_seg:
                    try:  
                        client = event.bot
                        original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
                        if original_msg and 'message' in original_msg:  
                            video_sources = extract_videos_from_chain(original_msg['message'])
                    except Exception as e:
                        logger.warning(f"从引用消息提取视频失败: {e}")
                
                # 【关键】前置判断：当前是什么媒体场景
                media_ctx = await self._detect_media_scenario(req, video_sources, reply_seg)
                req._cleanup_paths.extend(media_ctx.cleanup_paths)
                
                logger.info(f"[LLMEnhancement] 媒体场景: {media_ctx.scenario.value}")
                
                # ========== 分支处理 ==========
                processor = VideoFrameProcessor(
                    self.context,
                    event,
                    self._get_cfg
                )
                
                # 如果是视频/GIF 场景，为了避免 LLM 报错（1210 图片格式错误），先从 image_urls 中移除原始路径/URL
                if media_ctx.scenario in [MediaScenario.VIDEO, MediaScenario.GIF_ANIMATED]:
                    if media_ctx.media_path in req.image_urls:
                        req.image_urls.remove(media_ctx.media_path)
                    # 同时尝试移除可能存在的原始 URL
                    if len(req.image_urls) > 0:
                        req.image_urls = [url for url in req.image_urls if not any(url.lower().endswith(s) for s in [".mp4", ".mov", ".avi", ".wmv", ".flv", ".m4v"])]
                
                # 【场景 1】视频 → 统一抽帧流程
                if media_ctx.scenario == MediaScenario.VIDEO:  
                    logger.debug("[分支] 执行视频处理")
                    quoted_sender = getattr(req, "_quoted_sender", None) if reply_seg else None
                    success = await processor.process_long_video(req, media_ctx.media_path, media_ctx.duration, sender_name=quoted_sender)
                    if not success:  
                        logger.warning("[分支] 视频处理失败")
                
                # 【场景 2】GIF 动图 → 固定参数
                elif media_ctx.scenario == MediaScenario.GIF_ANIMATED:
                    logger.debug("[分支] 执行 GIF 处理")
                    quoted_sender = getattr(req, "_quoted_sender", None) if reply_seg else None
                    success = await processor.process_gif(req, media_ctx.media_path, sender_name=quoted_sender)
                    if not success: 
                        logger.warning("[分支] GIF 处理失败")
                
                # 【场景 3】无媒体或无效场景 → 无处理
                elif media_ctx.scenario == MediaScenario.NONE:
                    logger.debug("[分支] 无媒体内容")
        
        finally:
            # 5. 统一清理文件
            if hasattr(req, "_cleanup_paths"):
                await self._cleanup_paths(req._cleanup_paths)

    async def _cleanup_paths(self, cleanup_paths: List[str]):
        """延迟清理临时文件"""
        if not cleanup_paths:
            return
        
        async def final_cleanup():
            await asyncio.sleep(120)
            for p in cleanup_paths: 
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                    elif os.path.isdir(p):
                        shutil.rmtree(p)
                except: 
                    pass
        
        asyncio.create_task(final_cleanup())

    def _find_provider(self, provider_id: str):
        """通用方法：从所有 Provider（包含 LLM 和 STT）中查找匹配的 ID/Name"""
        if not provider_id: return None
        
        all_lists = []
        try: all_lists.append(self.context.get_all_providers())
        except: pass
        try: all_lists.append(self.context.get_all_stt_providers())
        except: pass
        
        for p_list in all_lists:
            if not p_list: continue
            for p in p_list:
                candidates = set()
                for attr in ("id", "provider_id", "name"):
                    val = getattr(p, attr, None)
                    if val: candidates.add(str(val))
                
                pcfg = getattr(p, "provider_config", None)
                if isinstance(pcfg, dict):
                    for k in ("id", "provider_id", "name"):
                        v = pcfg.get(k)
                        if v: candidates.add(str(v))
                
                if provider_id in candidates:
                    return p
        return None

    def _get_stt_provider(self):
        """获取 STT Provider"""
        asr_pid = self._get_cfg("asr_provider_id")
        p = self._find_provider(asr_pid)
        if p:
            logger.info(f"[LLMEnhancement] 成功匹配到指定的 STT Provider: {asr_pid}")
            return p
            
        if asr_pid:
            logger.warning(f"[LLMEnhancement] 未找到指定的 STT Provider: {asr_pid}")
        
        try:
            return self.context.get_using_stt_provider(umo=self.event.unified_msg_origin)
        except:
            pass
        
        return None

    @filter.on_llm_response(priority=20)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """在 LLM 返回结果后执行，用于更新会话状态并处理撤回拦截。"""
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not gid or not uid:
            return
            
        g = StateManager.get_group(gid)
        member = g.members.get(uid)
        
        if member:
            # 检查是否在此期间发生了撤回
            if member.cancel_merge:
                logger.info(f" [LLMEnhancement] LLM 响应生成完成，但检测到消息已撤回，拦截回复 (用户: {uid})。")
                member.cancel_merge = False
                member.pending_msg_ids.clear()
                event.stop_event() # 拦截响应
                return

            member.last_response = time.time()
            # 清理待处理消息 ID
            member.pending_msg_ids.clear()
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