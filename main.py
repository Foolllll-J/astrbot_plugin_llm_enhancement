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
        self.sent = Sentiment()
        self.similarity = Similarity()

    # ==================== WakePro 消息级别逻辑 ====================

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1)
    async def on_group_msg(self, event: AstrMessageEvent):
        """
        【第一层: 消息级别 - 基础检查和唤醒判断】
        """
        bid: str = event.get_self_id()
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        msg: str = event.message_str.strip() if event.message_str else ""
        g = StateManager.get_group(gid)

        # 1. 全局屏蔽检查
        if uid == bid: return
        if self.config["group_whitelist"] and gid not in self.config["group_whitelist"]: return
        if gid in self.config["group_blacklist"] and not event.is_admin():
            event.stop_event()
            return
        if uid in self.config.get("user_blacklist", []):
            event.stop_event()
            return

        # 2. 内置指令屏蔽
        if self.config["block_builtin"]:
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
        if wake and not msg and self.config.get("empty_mention_pt"):
            prompt = self.config["empty_mention_pt"].format(username=event.get_sender_name())
            yield event.request_llm(prompt=prompt)
            return

        if not msg:
            if not wake: return # 既没说话也没被唤醒，则返回

        # 提及唤醒
        if not wake and self.config["mention_wake"]:
            names = [n for n in self.config["mention_wake"] if n]
            for n in names:
                if n and n in msg:
                    wake = True
                    reason = f"提及唤醒({n})"
                    break

        # 唤醒延长
        if (not wake and self.config["wake_extend"] and 
            (now - member.last_response) <= int(self.config["wake_extend"] or 0)):
            if bmsgs := await self._get_history_msg(event, count=3):
                for bmsg in bmsgs:
                    simi = self.similarity.cosine(msg, bmsg, gid)
                    if simi > 0.3:
                        wake = True
                        reason = f"唤醒延长(相关性{simi:.2f})"
                        break

        # 话题相关性唤醒
        if not wake and self.config["relevant_wake"]:
            if bmsgs := await self._get_history_msg(event, count=5):
                for bmsg in bmsgs:
                    simi = self.similarity.cosine(msg, bmsg, gid)
                    if simi > self.config["relevant_wake"]:
                        wake = True
                        reason = f"话题相关性{simi:.2f}>{self.config['relevant_wake']}"
                        break

        # 答疑唤醒
        if not wake and self.config["ask_wake"]:
            if self.sent.ask(msg) > self.config["ask_wake"]:
                wake = True
                reason = "答疑唤醒"

        # 概率唤醒
        if not wake and self.config["prob_wake"]:
            if random.random() < self.config["prob_wake"]:
                wake = True
                reason = "概率唤醒"

        # 违禁词检查
        if self.config["wake_forbidden_words"]:
            for word in self.config["wake_forbidden_words"]:
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
        additional_components = [seg for seg in event.message_obj.message if isinstance(seg, (Comp.Forward, Comp.Reply))]
        
        @session_waiter(timeout=self.config["merge_delay"], record_history_chains=False)
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
                if isinstance(seg, (Comp.Forward, Comp.Reply)):
                    additional_components.append(seg)

            controller.keep(timeout=self.config["merge_delay"], reset_timeout=True)
        
        try:
            await collect_messages(event)
        except TimeoutError:
            if len(message_buffer) > 1:
                merged_msg = " ".join(message_buffer)
                event.message_str = merged_msg
                req.prompt = merged_msg
                logger.info(f"合并：用户({uid})合并了{len(message_buffer)}条消息")
        
        return additional_components

    # ==================== LLM 请求级别逻辑 ====================

    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        整合了 WakePro 的防护/合并与 ForwardReader 的内容提取。
        """
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
        if self.config["shutup"]:
            shut_th = self.sent.shut(msg)
            if shut_th > self.config["shutup"]:
                silence_sec = shut_th * self.config["silence_multiple"]
                g.shutup_until = now + silence_sec
                logger.info(f"群({gid})触发闭嘴，沉默{silence_sec:.1f}秒")
                event.stop_event()
                return

        if self.config["insult"]:
            insult_th = self.sent.insult(msg)
            if insult_th > self.config["insult"]:
                silence_sec = insult_th * self.config["silence_multiple"]
                member.silence_until = now + silence_sec
                logger.info(f"用户({uid})触发辱骂沉默{silence_sec:.1f}秒(下次生效)")

        # 2. WakePro 防护机制检查
        if g.shutup_until > now:
            event.stop_event()
            return
        if not event.is_admin() and member.silence_until > now:
            event.stop_event()
            return
        
        request_cd_value = self.config.get("request_cd", 0)
        if request_cd_value > 0:
            if now - member.last_request < request_cd_value:
                event.stop_event()
                return
        
        member.last_request = now
        
        # 3. WakePro 消息合并与组件收集
        all_components: List[Any] = []
        if self.config["merge_delay"] and self.config["merge_delay"] > 0:
            if not member.in_merging:
                member.in_merging = True
                try:
                    all_components = await self._handle_message_merge(event, req, gid, uid, member)
                finally:
                    member.in_merging = False
        else:
            # 不合并时，只拿当前消息的组件
            all_components = [seg for seg in event.message_obj.message if isinstance(seg, (Comp.Forward, Comp.Reply))]

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
                     try:
                         if forward_id:
                             extracted_texts, image_urls = await extract_forward_content(event.bot, forward_id)
                         else:
                             extracted_texts = json_extracted_texts
                         
                         if extracted_texts or image_urls:
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
                             logger.info(f"成功注入转发内容 ({len(extracted_texts)} 条文本) 到 LLM 请求末尾。")
                     except Exception as e:
                         logger.warning(f"内容提取失败: {e}") 
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
