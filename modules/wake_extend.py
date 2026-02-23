from typing import Any, Awaitable, Callable, List, Optional, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .state_manager import GroupState, MemberState


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
