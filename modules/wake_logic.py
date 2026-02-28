import re
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp

from .state_manager import GroupState, MemberState

_MENTION_WAKE_REGEX_CACHE: dict[str, re.Pattern[str]] = {}
_MENTION_WAKE_REGEX_INVALID: set[str] = set()


def _parse_regex_flags(flags_text: str) -> int:
    flags = 0
    for ch in flags_text:
        if ch == "i":
            flags |= re.IGNORECASE
        elif ch == "m":
            flags |= re.MULTILINE
        elif ch == "s":
            flags |= re.DOTALL
        elif ch == "x":
            flags |= re.VERBOSE
        elif ch == "a":
            flags |= re.ASCII
        elif ch == "u":
            flags |= re.UNICODE
        elif ch == "L":
            flags |= re.LOCALE
        else:
            raise ValueError(f"unsupported regex flag: {ch}")
    return flags


def match_mention_wake_rule(rule: str, text: str) -> bool:
    """
    mention_wake 支持三种规则：
    1) 普通字符串：直接做子串匹配
    2) re:pattern：按正则表达式匹配
    3) /pattern/flags：按正则表达式匹配（flags 支持 i,m,s,x,a,u,L）
    """
    raw = str(rule or "").strip()
    if not raw or not text:
        return False

    pattern = ""
    flags = 0
    use_regex = False

    if raw.startswith("re:"):
        pattern = raw[3:].strip()
        use_regex = bool(pattern)
    elif raw.startswith("/") and len(raw) > 1:
        tail = raw.rfind("/")
        if tail > 0:
            flag_text = raw[tail + 1 :].strip()
            try:
                flags = _parse_regex_flags(flag_text)
                pattern = raw[1:tail]
                use_regex = bool(pattern)
            except ValueError:
                use_regex = False

    if not use_regex:
        return raw in text

    cache_key = f"{pattern}\n{flags}"
    compiled = _MENTION_WAKE_REGEX_CACHE.get(cache_key)
    if compiled is None:
        if cache_key in _MENTION_WAKE_REGEX_INVALID:
            return False
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            _MENTION_WAKE_REGEX_INVALID.add(cache_key)
            logger.warning(f"[LLMEnhancement] mention_wake 正则无效: rule={raw!r}, error={e}")
            return False
        _MENTION_WAKE_REGEX_CACHE[cache_key] = compiled

    return bool(compiled.search(text))


def detect_wake_media_components(message_chain: Any) -> tuple[bool, bool, bool, bool, str]:
    """识别消息链中的视频/文件/转发/JSON 组件，并返回文件名。"""
    has_video_component = False
    has_file_component = False
    has_forward_component = False
    has_json_component = False
    file_name = ""
    try:
        for seg in (message_chain or []):
            if isinstance(seg, Comp.Video):
                has_video_component = True
            elif isinstance(seg, Comp.File):
                has_file_component = True
                file_name = str(getattr(seg, "name", "") or getattr(seg, "file", "") or "").strip()
            elif isinstance(seg, Comp.Forward):
                has_forward_component = True
            elif isinstance(seg, Comp.Json):
                has_json_component = True
            elif isinstance(seg, dict):
                seg_type = seg.get("type")
                data = seg.get("data") or {}
                if seg_type == "video":
                    has_video_component = True
                elif seg_type == "file":
                    has_file_component = True
                    if not file_name:
                        file_name = str(data.get("name") or data.get("file") or "").strip()
                elif seg_type == "forward":
                    has_forward_component = True
                elif seg_type == "json":
                    has_json_component = True
    except Exception:
        return False, False, False, False, ""
    return (
        has_video_component,
        has_file_component,
        has_forward_component,
        has_json_component,
        file_name,
    )


def normalize_wake_trigger_message(
    wake: bool,
    msg: str,
    gid: Optional[str],
    sender_name: str,
    has_video_component: bool,
    has_file_component: bool,
    has_forward_component: bool,
    has_json_component: bool,
    file_name: str,
) -> tuple[str, Optional[str]]:
    """
    对“已触发唤醒但无文本”的场景补齐文本，返回 (normalized_msg, reason)。
    reason 为 None 表示不做改动。
    """
    if not wake or msg:
        return msg, None
    if gid:
        return f"{sender_name}@了你", "空@唤醒"
    if has_video_component:
        return f"{sender_name}发送了一个视频", "视频消息唤醒"
    if has_file_component:
        suffix = f"：{file_name}" if file_name else ""
        return f"{sender_name}发送了一个文件{suffix}", "文件消息唤醒"
    if has_forward_component:
        return f"{sender_name}发送了一条转发消息", "转发消息唤醒"
    if has_json_component:
        return f"{sender_name}发送了一条分享卡片", "JSON卡片唤醒"
    return msg, None


def evaluate_mention_wake(msg: str, mention_wake: Any) -> Optional[str]:
    """提及唤醒命中时返回命中的规则文本。"""
    if not msg or not mention_wake:
        return None
    if isinstance(mention_wake, (list, tuple, set)):
        rules = mention_wake
    else:
        rules = [mention_wake]
    for rule in rules:
        if not rule:
            continue
        raw_rule = str(rule)
        if match_mention_wake_rule(raw_rule, msg):
            return raw_rule
    return None


def contains_forbidden_wake_word(message_text: str, forbidden_words: Any) -> Optional[str]:
    """违禁词命中时返回命中的词。"""
    if not message_text or not forbidden_words:
        return None
    if isinstance(forbidden_words, (list, tuple, set)):
        words = forbidden_words
    else:
        words = [forbidden_words]
    for word in words:
        if not word:
            continue
        rule = str(word)
        if rule and match_mention_wake_rule(rule, message_text):
            return rule
    return None


def _extract_provider_text(response: Any) -> str:
    """尽可能从 Provider 响应中提取文本。"""
    try:
        if response is None:
            return ""
        if hasattr(response, "completion_text"):
            return str(response.completion_text or "").strip()
        if isinstance(response, dict):
            for key in ("completion_text", "text", "content"):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(response).strip()
    except Exception:
        return ""


async def wake_extend_llm_decision(
    event: AstrMessageEvent,
    msg: str,
    history_msgs: List[str],
    threshold: float,
    provider_id: str,
    find_provider: Callable[[str], Any],
) -> Optional[bool]:
    """使用配置的 Provider 做唤醒延长判定。返回 True/False，失败返回 None。"""
    provider = find_provider(provider_id)
    if not provider:
        logger.warning(f"[LLMEnhancement] 未找到唤醒延长判定 Provider: {provider_id}")
        return None

    history_text = "\n".join([f"{idx + 1}. {h}" for idx, h in enumerate(history_msgs[-3:])]) or "<empty>"
    prompt = (
        "请判断当前消息是否与上文足够相关，是否需要继续响应。\n"
        f"相关性阈值参考: {threshold:.2f}（0~1，越高越严格）\n"
        "你只能输出一个大写字母：T 或 F。\n"
        "T=需要响应，F=不需要响应。\n\n"
        f"[上文]\n{history_text}\n\n"
        f"[当前消息]\n{msg}"
    )
    system_prompt = "你是唤醒判定器。只输出一个大写字母：T 或 F。禁止输出其他内容。"

    try:
        resp = await provider.text_chat(
            prompt=prompt,
            system_prompt=system_prompt,
            image_urls=[],
            context=[],
        )
        text = _extract_provider_text(resp).upper()
        if text.startswith("T"):
            logger.info("[LLMEnhancement] 唤醒延长模型判定结果: T")
            return True
        if text.startswith("F"):
            logger.info("[LLMEnhancement] 唤醒延长模型判定结果: F")
            return False
        logger.warning(f"[LLMEnhancement] 唤醒延长模型判定返回无效结果: {text[:20]}")
        return None
    except Exception as e:
        logger.warning(f"[LLMEnhancement] 唤醒延长模型判定失败，回退本地相似度: {e}")
        return None


async def evaluate_wake_extend(
    event: AstrMessageEvent,
    msg: str,
    gid: str,
    uid: str,
    now: float,
    group_state: GroupState,
    member: MemberState,
    get_cfg: Callable[[str, Any], Any],
    get_history_msg: Callable[[AstrMessageEvent, int], Awaitable[List[str]]],
    similarity_fn: Callable[[str, str, List[str]], Awaitable[float]],
    find_provider: Callable[[str], Any],
) -> Tuple[bool, Optional[str]]:
    """
    评估唤醒延长是否成立。
    返回 (should_wake, reason)。
    """
    wake_extend = float(get_cfg("wake_extend", 0) or 0)
    if wake_extend <= 0:
        return False, None

    same_user_only = bool(get_cfg("wake_extend_same_user_only", True))

    # 以群级“最近一次请求用户/时间”为主，历史数据缺失时回退旧状态字段
    ref_uid = group_state.last_response_uid
    ref_ts = float(group_state.last_response_ts or 0.0)
    if ref_ts <= 0:
        if same_user_only:
            ref_uid = uid
            ref_ts = float(member.last_response or 0.0)
        else:
            latest_ts = 0.0
            latest_uid = None
            for m_uid, m_state in group_state.members.items():
                m_ts = float(m_state.last_response or 0.0)
                if m_ts > latest_ts:
                    latest_ts = m_ts
                    latest_uid = m_uid
            ref_uid = latest_uid
            ref_ts = latest_ts

    in_window = bool(ref_ts > 0 and (now - ref_ts) <= wake_extend)
    if in_window and same_user_only and ref_uid and uid != ref_uid:
        in_window = False
    if not in_window:
        return False, None

    threshold = float(get_cfg("wake_extend_similarity", 0.1) or 0.0)
    if threshold <= 0:
        return True, "唤醒延长(阈值0)"

    history_msgs = await get_history_msg(event, 3)
    if not history_msgs:
        return False, None

    provider_id = str(get_cfg("wake_extend_provider_id", "") or "").strip()
    if provider_id:
        llm_decision = await wake_extend_llm_decision(
            event=event,
            msg=msg,
            history_msgs=history_msgs,
            threshold=threshold,
            provider_id=provider_id,
            find_provider=find_provider,
        )
        if llm_decision is True:
            return True, "唤醒延长(模型判定T)"
        if llm_decision is False:
            return False, None

        simi = await similarity_fn(gid, msg, history_msgs)
        logger.debug(
            f" [LLMEnhancement] 唤醒延长检查(本地回退): 相关系数={simi:.4f}, "
            f"阈值={threshold:.4f}, 历史参考={len(history_msgs)}条"
        )
        if simi >= threshold:
            return True, f"唤醒延长(相关性{simi:.2f})"
        return False, None

    simi = await similarity_fn(gid, msg, history_msgs)
    logger.debug(
        f" [LLMEnhancement] 唤醒延长检查: 相关系数={simi:.4f}, "
        f"阈值={threshold:.4f}, 历史参考={len(history_msgs)}条"
    )
    if simi >= threshold:
        return True, f"唤醒延长(相关性{simi:.2f})"
    return False, None
