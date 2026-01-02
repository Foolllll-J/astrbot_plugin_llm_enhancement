import asyncio
import json
import time
import random
from typing import List, Dict, Any, Optional, Tuple 
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig 
import astrbot.api.message_components as Comp 
from astrbot.api.provider import LLMResponse, ProviderRequest 
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from .modules.sentiment import Sentiment
from .modules.similarity import Similarity
from .modules.state_manager import StateManager, MemberState, GroupState
from .modules.forward_parser import extract_forward_content
from .modules.video_parser import extract_videos_from_chain, prepare_video_context, get_video_summary, extract_forward_video_keyframes, probe_duration_sec
from .modules.info_utils import process_group_members_info
import os
import shutil

# 检查是否为 aiocqhttp 平台，因为合并转发是其特性 
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

# 合并延迟期间最多合并的消息数量（防止轰炸攻击）
MAX_MERGE_MESSAGES = 10


# ==================== 数据模型 ====================




@register("llm_enhancement", "Foolllll", "增强LLM的综合表现。", "0.1.0") 
class LLMEnhancement(Star): 
    def __init__(self, context: Context, config: AstrBotConfig): 
        super().__init__(context) 
        self.config = config
        self.cfg = {}
        self._refresh_config()
        self.sent = Sentiment()
        self.similarity = Similarity()

    def _refresh_config(self):
        """将 template_list 格式的配置平铺到 self.cfg 中"""
        # 基础配置直接复制
        for k in ["group_whitelist", "group_blacklist", "user_blacklist"]:
            self.cfg[k] = self.config.get(k)
        
        # 处理 modules 字段
        modules = self.config.get("modules", [])
        if isinstance(modules, list):
            for item in modules:
                # 获取模板 key
                template_key = item.get("__template_key")
                if not template_key: continue
                
                # 将该项下的所有属性平铺到 self.cfg
                for k, v in item.items():
                    if k != "__template_key":
                        self.cfg[k] = v
        

    def _get_cfg(self, key: str, default: Any = None) -> Any:
        """获取配置项，优先从平铺后的 self.cfg 获取，否则从 self.config 获取"""
        if key in self.cfg:
            return self.cfg[key]
        return self.config.get(key, default)

    # ==================== WakePro 消息级别逻辑 ====================

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1)
    async def on_group_msg(self, event: AstrMessageEvent):
        """
        【第一层: 消息级别 - 基础检查和唤醒判断】
        """
        self._refresh_config() # 每次处理前刷新，确保配置实时生效
        bid: str = event.get_self_id()
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        msg: str = event.message_str.strip() if event.message_str else ""
        g = StateManager.get_group(gid)

        # 1. 全局屏蔽检查
        if uid == bid: return
        whitelist = self._get_cfg("group_whitelist")
        if whitelist and gid not in whitelist: return
        
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
            if not wake: return # 既没说话也没被唤醒，则返回

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
                for bmsg in bmsgs:
                    simi = self.similarity.cosine(msg, bmsg, gid)
                    if simi > 0.3:
                        wake = True
                        reason = f"唤醒延长(相关性{simi:.2f})"
                        break

        # 话题相关性唤醒
        relevant_wake = self._get_cfg("relevant_wake")
        if not wake and relevant_wake:
            if bmsgs := await self._get_history_msg(event, count=5):
                for bmsg in bmsgs:
                    simi = self.similarity.cosine(msg, bmsg, gid)
                    if simi > relevant_wake:
                        wake = True
                        reason = f"话题相关性{simi:.2f}>{relevant_wake}"
                        break

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
            logger.info(f"群({gid})用户({uid}) {reason}：{msg[:50]}")

    # ==================== 消息合并处理 ====================
    
    async def _handle_message_merge(self, event: AstrMessageEvent, req: ProviderRequest, gid: str, uid: str, member: MemberState) -> List[Any]:
        message_buffer = [event.message_str]
        # 收集第一条消息中的组件
        additional_components = [seg for seg in event.message_obj.message if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File))]
        
        merge_delay = self._get_cfg("merge_delay")
        @session_waiter(timeout=merge_delay, record_history_chains=False)
        async def collect_messages(controller: SessionController, ev: AstrMessageEvent):
            nonlocal message_buffer, additional_components
            if ev.get_sender_id() != uid or ev.get_group_id() != gid: return
            
            if len(message_buffer) == 1 and ev.message_str == message_buffer[0]:
                ev.stop_event()
                return
            
            ev.stop_event()

            if len(message_buffer) >= MAX_MERGE_MESSAGES:
                controller.stop()
                return
            
            message_buffer.append(ev.message_str)
            # 收集后续消息中的组件
            for seg in ev.message_obj.message:
                if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File)):
                    additional_components.append(seg)

            controller.keep(timeout=merge_delay, reset_timeout=True)
        
        try:
            await collect_messages(event)
        except TimeoutError:
            if len(message_buffer) > 1:
                merged_msg = " ".join(message_buffer)
                event.message_str = merged_msg
                req.prompt = merged_msg
                logger.info(f"合并：用户({uid})合并了{len(message_buffer)}条消息")
        
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
        # 直接使用 QQ 头像 URL
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        
        try:
            req: ProviderRequest = getattr(event, "_provider_req", None)
            if not req:
                req = getattr(event, "request", None)
            
            if not req:
                logger.error("无法获取当前请求对象 ProviderRequest，注入失败。")
                return f"获取头像成功，但内部错误导致无法注入到当前请求中。"

            # 1. 合并转发的 Prompt 注入逻辑
            user_question = req.prompt.strip()
            context_prompt = (
                f"\n\n以下是系统为你获取到的用户 {user_id} 的头像信息，请根据该头像内容来响应用户的要求。信息如下：\n"
                f"--- 注入内容开始 ---\n"
                f"[图片] 用户 {user_id} 的头像已作为图片附件注入到本次请求的 image_urls 中。\n"
                f"--- 注入内容结束 ---"
            )
            req.prompt = user_question + context_prompt

            # 2. 合并转发的图片注入逻辑
            req.image_urls.append(avatar_url)
            
            logger.info(f"成功将用户 {user_id} 的头像注入到 LLM 请求中 (Prompt + URL)。")
            logger.info(f"当前 LLM 请求 Prompt: {req.prompt}")
            logger.info(f"当前 LLM 请求 Image URLs: {req.image_urls}")

            return f"已成功获取用户 {user_id} 的头像并注入到请求上下文中。"
        except Exception as e:
            logger.error(f"注入头像到请求时发生错误: {e}")
            return f"注入头像失败: {e}"

    @filter.llm_tool(name="get_group_members_info")
    async def get_group_members(self, event: AstrMessageEvent) -> str:
        """
        获取QQ群成员信息的LLM工具。
        需要判断是否为群聊时，以及当需要知道群里有哪些人，或者需要获取他们的昵称和用户ID，
        或者需要知道群里是否有特定成员时，调用此工具。其中display_name是“群昵称”，username是用户“QQ名”
        获取数据之后需要联系上下文，用符合prompt的方式回答用户的问题。
        """
        return await process_group_members_info(event)

    # ==================== LLM 请求级别逻辑 ====================

    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        整合了 WakePro 的防护/合并与 ForwardReader 的内容提取。
        """
        setattr(event, "_provider_req", req)
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not gid or not uid: return
        
        g = StateManager.get_group(gid)
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        member = g.members[uid]
        now = time.time()
        msg = event.message_str
        
        if member.in_merging: return

        # 1. WakePro 沉默触发检测
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

        # 2. WakePro 防护机制检查
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
        
        # 3. WakePro 消息合并与组件收集
        all_components: List[Any] = []
        merge_delay = self._get_cfg("merge_delay")
        if merge_delay and merge_delay > 0:
            if not member.in_merging:
                member.in_merging = True
                try:
                    all_components = await self._handle_message_merge(event, req, gid, uid, member)
                finally:
                    member.in_merging = False
        else:
            # 不合并时，只拿当前消息的组件
            all_components = [seg for seg in event.message_obj.message if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File))]

        # 4. ForwardReader 内容提取与注入
        if IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent):
            # 只有在机器人被唤醒且存在转发内容时才进行解析
            if event.is_at_or_wake_command:
                forward_id: Optional[str] = None 
                reply_seg: Optional[Comp.Reply] = None 
                json_extracted_texts: List[str] = [] 
                
                # 从收集到的所有组件中寻找转发内容
                for seg in all_components: 
                    if isinstance(seg, Comp.Forward): 
                        forward_id = seg.id 
                        break
                    elif isinstance(seg, Comp.Reply): 
                        reply_seg = seg 
                

                if not forward_id and reply_seg:
                    try:
                        client = event.bot
                        original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
                        if original_msg and 'message' in original_msg: 
                            original_message_chain = original_msg['message'] 
                            if isinstance(original_message_chain, list): 
                                for segment in original_message_chain: 
                                    seg_type = segment.get("type")
                                    if seg_type == "forward": 
                                        forward_id = segment.get("data", {}).get("id") 
                                        if forward_id: break
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
                                                            if clean_text: json_extracted_texts.append(clean_text)
                                                    if json_extracted_texts: break
                                        except: continue
                    except Exception as e: 
                        logger.warning(f"获取被回复消息详情失败: {e}") 
                
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
                                    max_duration=self._get_cfg("video_max_duration_sec", 120),
                                    timeout_sec=10
                                )
                                if f_frames:
                                    image_urls.extend(f_frames)
                                    
                                    # 处理转发视频的 ASR
                                    if self._get_cfg("video_asr_enable", True) and f_local_videos:
                                        for idx, lv_path in enumerate(f_local_videos):
                                            try:
                                                from .modules.video_parser import extract_audio_wav
                                                wav_path = await extract_audio_wav(self._get_cfg("ffmpeg_path", ""), lv_path)
                                                if wav_path and os.path.exists(wav_path):
                                                    stt = None
                                                    asr_pid = self._get_cfg("asr_provider_id")
                                                    if asr_pid:
                                                        try:
                                                            all_stt = self.context.get_all_stt_providers()
                                                            asr_pid_l = asr_pid.lower()
                                                            for p in all_stt:
                                                                candidates = []
                                                                for attr in ("id", "provider_id", "name"):
                                                                    val = getattr(p, attr, None)
                                                                    if isinstance(val, str) and val: candidates.append(val)
                                                                if any(str(c).lower() == asr_pid_l for c in candidates):
                                                                    stt = p
                                                                    break
                                                        except: pass
                                                    if not stt:
                                                        try: stt = self.context.get_using_stt_provider(umo=event.unified_msg_origin)
                                                        except: pass
                                                    
                                                    if stt:
                                                        f_asr_text = None
                                                        try:
                                                            if hasattr(stt, "get_text"): f_asr_text = await stt.get_text(wav_path)
                                                            elif hasattr(stt, "speech_to_text"):
                                                                res = await stt.speech_to_text(wav_path)
                                                                f_asr_text = res.get("text", "") if isinstance(res, dict) else str(res)
                                                            
                                                            if f_asr_text:
                                                                extracted_texts.append(f" [视频语音转写 {idx+1}] {f_asr_text}")
                                                                logger.info(f"转发视频 {idx+1} ASR 成功: {f_asr_text[:50]}...")
                                                        except: pass
                                                    
                                                    try: os.remove(wav_path)
                                                    except: pass
                                            except: pass

                                    if f_cleanup:
                                        async def cleanup_f():
                                            await asyncio.sleep(60)
                                            for p in f_cleanup:
                                                try:
                                                    if os.path.isdir(p): shutil.rmtree(p)
                                                    elif os.path.isfile(p): os.remove(p)
                                                except: pass
                                        asyncio.create_task(cleanup_f())

                            chat_records = "\n".join(extracted_texts)
                            user_question = req.prompt.strip() or "请总结一下这个聊天记录"
                            context_prompt = (
                                f"\n\n用户是在吐槽以下聊天记录中的内容，请根据以下聊天记录内容来响应用户的吐槽。聊天记录如下：\n"
                                f"--- 聊天记录开始 ---\n"
                                f"{chat_records}\n"
                                f"--- 聊天记录结束 ---"
                            )
                            req.prompt = user_question + context_prompt
                            req.image_urls.extend(image_urls)
                            logger.info(f"成功注入转发内容 ({len(extracted_texts)} 条文本, {len(image_urls)} 张图片) 到 LLM 请求末尾。")
                    except Exception as e:  
                         logger.warning(f"内容提取失败: {e}") 

                # 5. 视频解析增强
                if self._get_cfg("video_detect_enable", True):
                    video_sources = extract_videos_from_chain(all_components)
                    
                    # 如果当前消息没视频，尝试从被回复的消息中获取
                    if not video_sources and reply_seg:
                        try:
                            client = event.bot
                            original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
                            if original_msg and 'message' in original_msg:
                                video_sources = extract_videos_from_chain(original_msg['message'])
                        except Exception as e:
                            logger.warning(f"video_parser: 从引用消息提取视频失败: {e}")
                    
                    if video_sources:
                        try:
                            # 探测时长（取第一个视频的）
                            duration = 0
                            try:
                                duration = probe_duration_sec(self._get_cfg("ffmpeg_path", ""), video_sources[0]) or 0
                            except: pass

                            # 计算抽帧数量
                            interval = self._get_cfg("video_frame_interval_sec", 6)
                            if duration > 0:
                                sample_count = max(1, int(duration / interval))
                            else:
                                sample_count = self._get_cfg("video_sample_count", 3)

                            frames, cleanup_paths, local_video_path = await prepare_video_context(
                                event,
                                video_sources,
                                max_mb=self._get_cfg("video_max_size_mb", 50),
                                max_duration=self._get_cfg("video_max_duration_sec", 120),
                                sample_count=sample_count,
                                ffmpeg_path=self._get_cfg("ffmpeg_path", ""),
                                process_timeout=30
                            )
                            
                            if frames:
                                # 可选 ASR：由 video_asr_enable + asr_provider_id 控制
                                asr_text = None
                                if self._get_cfg("video_asr_enable", True) and local_video_path:
                                    logger.info(f"video_parser: ASR enabled, starting processing for {local_video_path}")
                                    try:
                                        from .modules.video_parser import extract_audio_wav
                                        wav_path = await extract_audio_wav(self._get_cfg("ffmpeg_path", ""), local_video_path)
                                        if wav_path and os.path.exists(wav_path):
                                            # 选择 ASR Provider
                                            stt = None
                                            asr_pid = self._get_cfg("asr_provider_id")
                                            if asr_pid:
                                                try:
                                                    all_p = self.context.get_all_stt_providers()
                                                    asr_pid_l = asr_pid.lower()
                                                    for p in all_p:
                                                        candidates = []
                                                        for attr in ("id", "provider_id", "name"):
                                                            val = getattr(p, attr, None)
                                                            if isinstance(val, str) and val: candidates.append(val)
                                                        if any(str(c).lower() == asr_pid_l for c in candidates):
                                                            stt = p
                                                            break
                                                except: pass
                                            
                                            if not stt:
                                                try:
                                                    stt = self.context.get_using_stt_provider(umo=event.unified_msg_origin)
                                                except: pass
                                            
                                            if stt:
                                                try:
                                                    if hasattr(stt, "get_text"):
                                                        asr_text = await stt.get_text(wav_path)
                                                    elif hasattr(stt, "speech_to_text"):
                                                        res = await stt.speech_to_text(wav_path)
                                                        asr_text = res.get("text", "") if isinstance(res, dict) else str(res)
                                                    
                                                    if asr_text:
                                                        logger.info(f"video_parser: ASR 成功: {asr_text[:50]}...")
                                                except Exception as e:
                                                    logger.warning(f"video_parser: ASR 调用失败: {e}")
                                            else:
                                                logger.warning("video_parser: ASR enabled but no STT provider found.")
                                            
                                            try: os.remove(wav_path)
                                            except: pass
                                        else:
                                            logger.warning("video_parser: ASR failed to extract audio wav.")
                                    except Exception as e:
                                        logger.warning(f"video_parser: ASR 处理失败: {e}")
                                else:
                                    logger.info("video_parser: ASR disabled or no local video path available.")

                                # 逐帧描述 -> 汇总文本
                                summary = await get_video_summary(
                                    self.context,
                                    event,
                                    frames,
                                    duration=duration,
                                    video_name="用户发送的视频",
                                    image_provider_id=self._get_cfg("image_provider_id"),
                                    asr_text=asr_text
                                )
                                
                                if summary:
                                    logger.info(f"video_parser: 视频总结完成: {summary}")
                                    user_question = req.prompt.strip()
                                    context_prompt = (
                                        f"\n\n以下是系统为你分析的视频内容总结，请结合此总结来响应用户的要求。信息如下：\n"
                                        f"--- 注入内容开始 ---\n"
                                        f"[视频总结] {summary}\n"
                                        f"--- 注入内容结束 ---"
                                    )
                                    req.prompt = user_question + context_prompt
                                    logger.info(f"成功注入视频总结到 LLM 请求。")
                                else:
                                    logger.warning("video_parser: 视频总结为空")
                                    # 回退到直接注入图片帧（或者报错）
                                    req.image_urls.extend(frames)
                                    logger.info(f"视频总结失败，回退到直接注入视频帧 ({len(frames)} 张图片)。")
                                
                                # 注册清理任务
                                if cleanup_paths:
                                    async def cleanup():
                                        await asyncio.sleep(60) # 延迟清理，给 LLM 留出读取时间
                                        for p in cleanup_paths:
                                            try:
                                                if os.path.isdir(p): shutil.rmtree(p)
                                                elif os.path.isfile(p): os.remove(p)
                                            except: pass
                                    asyncio.create_task(cleanup())
                        except Exception as e:
                            logger.warning(f"视频解析失败: {e}")


    @filter.on_llm_response(priority=20)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        if not gid or not uid: return
        g = StateManager.get_group(gid)
        member = g.members.get(uid)
        if member:
            member.last_response = time.time()

    async def _get_history_msg(self, event: AstrMessageEvent, role: str = "assistant", count: int | None = 0) -> List[str]:
        try:
            umo = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if not curr_cid: return []
            conversation = await self.context.conversation_manager.get_conversation(umo, curr_cid)
            if not conversation: return []
            history = json.loads(conversation.history or "[]")
            contexts = [record["content"] for record in history if record.get("role") == role and record.get("content")]
            return contexts[-count:] if count else contexts
        except Exception as e:
            logger.error(f"获取历史消息失败：{e}")
            return []

    async def terminate(self): 
        pass
