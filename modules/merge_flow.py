import time
from typing import Any, Callable, Dict, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .state_manager import GroupState, MemberState

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent

    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


def get_event_msg_id(event: AstrMessageEvent) -> Optional[str]:
    if hasattr(event, "message_obj") and hasattr(event.message_obj, "message_id"):
        raw_msg_id = getattr(event.message_obj, "message_id", None)
        if raw_msg_id is not None:
            return str(raw_msg_id)
    return None


def prune_member_msg_cache(member: MemberState, keep_sec: float) -> None:
    now_ts = time.time()
    member.recent_wake_msgs = [
        item for item in member.recent_wake_msgs
        if now_ts - float(item.get("ts", 0.0)) <= keep_sec
    ]
    expired_ids = [mid for mid, exp_ts in member.merged_msg_ids.items() if exp_ts <= now_ts]
    for mid in expired_ids:
        member.merged_msg_ids.pop(mid, None)


def build_event_snapshot(event: AstrMessageEvent, gid: str, uid: str) -> Dict[str, Any]:
    chain = []
    if hasattr(event, "message_obj") and hasattr(event.message_obj, "message") and event.message_obj.message:
        chain = event.message_obj.message
    def _is_merge_component(seg: Any) -> bool:
        if isinstance(seg, (Comp.Forward, Comp.Reply, Comp.Video, Comp.File, Comp.Json)):
            return True
        if isinstance(seg, dict):
            seg_type = str(seg.get("type") or "").lower()
            return seg_type in {"forward", "reply", "video", "file", "json"}
        return False

    components = [seg for seg in chain if _is_merge_component(seg)]
    return {
        "msg_id": get_event_msg_id(event),
        "ts": time.time(),
        "gid": gid or f"private_{uid}",
        "uid": uid,
        "sender_name": event.get_sender_name(),
        "text": event.message_str,
        "components": components,
    }


def upsert_recent_wake_snapshot(member: MemberState, snapshot: Dict[str, Any]) -> None:
    msg_id = str(snapshot.get("msg_id") or "")
    if not msg_id:
        return
    for idx, item in enumerate(member.recent_wake_msgs):
        if str(item.get("msg_id") or "") == msg_id:
            # 保留更早的时间戳，避免并发事件重排导致窗口错位
            old_ts = float(item.get("ts", snapshot["ts"]))
            snapshot["ts"] = min(old_ts, float(snapshot["ts"]))
            member.recent_wake_msgs[idx] = snapshot
            return
    member.recent_wake_msgs.append(snapshot)


def prepare_initial_merge_snapshots(
    event: AstrMessageEvent,
    gid: str,
    uid: str,
    member: MemberState,
    merge_delay: float,
    merged_window_tolerance: float,
    merged_skip_ttl: float,
    add_pending_msg_id: Callable[[str], None],
) -> tuple[list[Dict[str, Any]], Optional[str]]:
    """预选本次合并窗口内的消息快照，并更新 member 的已并入索引。"""
    current_snapshot = build_event_snapshot(event, gid, uid)
    current_msg_id = str(current_snapshot.get("msg_id") or "")
    if current_msg_id:
        upsert_recent_wake_snapshot(member, current_snapshot)

    current_ts = float(current_snapshot.get("ts", time.time()))
    if current_msg_id:
        for item in member.recent_wake_msgs:
            if str(item.get("msg_id") or "") == current_msg_id:
                current_ts = float(item.get("ts", current_ts))
                break

    nearby_ts_list = [
        float(item.get("ts", current_ts))
        for item in member.recent_wake_msgs
        if abs(float(item.get("ts", current_ts)) - current_ts) <= merge_delay
    ]
    window_start_ts = min(nearby_ts_list) if nearby_ts_list else current_ts
    window_end_ts = window_start_ts + merge_delay + merged_window_tolerance

    preselected_snapshots: list[Dict[str, Any]] = []
    for item in sorted(member.recent_wake_msgs, key=lambda x: float(x.get("ts", 0.0))):
        msg_id = str(item.get("msg_id") or "")
        if not msg_id:
            continue
        if msg_id in member.merged_msg_ids and msg_id != current_msg_id:
            continue
        ts = float(item.get("ts", 0.0))
        if ts < window_start_ts:
            continue
        if ts > window_end_ts:
            continue
        preselected_snapshots.append(item)

    if not preselected_snapshots:
        preselected_snapshots = [current_snapshot]

    if current_msg_id and all(str(item.get("msg_id") or "") != current_msg_id for item in preselected_snapshots):
        preselected_snapshots.insert(0, current_snapshot)

    deduped_snapshots: list[Dict[str, Any]] = []
    seen_msg_ids: set[str] = set()
    for item in preselected_snapshots:
        msg_id = str(item.get("msg_id") or "")
        key = msg_id or f"noid_{id(item)}"
        if key in seen_msg_ids:
            continue
        seen_msg_ids.add(key)
        deduped_snapshots.append(item)
    preselected_snapshots = deduped_snapshots

    for item in preselected_snapshots:
        msg_id = str(item.get("msg_id") or "")
        if msg_id:
            add_pending_msg_id(msg_id)
            if current_msg_id and msg_id != current_msg_id:
                member.merged_msg_ids[msg_id] = time.time() + merged_skip_ttl

    trigger_msg_id = current_msg_id or (str(preselected_snapshots[0].get("msg_id") or "") if preselected_snapshots else None)
    return preselected_snapshots, trigger_msg_id


def add_pending_msg_id(group_state: GroupState, member: MemberState, msg_id: Optional[str]) -> None:
    """在成员和组反向索引中同时登记待处理消息 ID。"""
    if not msg_id:
        return
    member.pending_msg_ids.add(msg_id)
    group_state.pending_msg_index[msg_id] = member.uid


def remove_pending_msg_id(group_state: GroupState, member: MemberState, msg_id: Optional[str]) -> None:
    """在成员和组反向索引中同时移除消息 ID。"""
    if not msg_id:
        return
    member.pending_msg_ids.discard(msg_id)
    group_state.pending_msg_index.pop(msg_id, None)


def clear_pending_msg_ids(group_state: GroupState, member: MemberState) -> None:
    """清空成员待处理消息，并维护组反向索引一致性。"""
    if not member.pending_msg_ids:
        return
    for msg_id in list(member.pending_msg_ids):
        group_state.pending_msg_index.pop(msg_id, None)
    member.pending_msg_ids.clear()


async def is_msg_still_available(event: AstrMessageEvent, msg_id: str) -> bool:
    """最终阶段通过协议端校验消息是否仍可获取（仅 AIOCQHTTP）。"""
    if not msg_id:
        return True
    if not (IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent)):
        return True

    client = getattr(event, "bot", None)
    api = getattr(client, "api", None) if client else None
    if api is None:
        logger.debug("[LLMEnhancement] get_msg 校验跳过：event.bot.api 不可用")
        return True

    try:
        message_id = int(msg_id)
    except (TypeError, ValueError):
        logger.debug(f"[LLMEnhancement] get_msg 校验跳过：非数字 message_id，msg_id={msg_id}")
        return True

    try:
        detail = await api.call_action("get_msg", message_id=message_id)
    except Exception as e:
        err_text = str(e).lower()
        if any(k in err_text for k in ("not found", "not exist", "不存在", "撤回", "invalid")):
            logger.info(f"[LLMEnhancement] get_msg 指示消息不可用/已撤回: msg_id={msg_id}")
            return False
        logger.debug(f"[LLMEnhancement] get_msg 调用异常但按可用处理: msg_id={msg_id}, err={e}")
        return True

    if isinstance(detail, dict) and isinstance(detail.get("data"), dict):
        detail = detail["data"]

    if not isinstance(detail, dict):
        logger.info(f"[LLMEnhancement] get_msg 返回结构异常，判定不可用: msg_id={msg_id}")
        return False

    msg_content = detail.get("message")
    if msg_content is None:
        logger.info(f"[LLMEnhancement] get_msg 未返回 message 字段，判定不可用: msg_id={msg_id}")
        return False
    if isinstance(msg_content, (list, str)) and len(msg_content) == 0:
        logger.debug(f"[LLMEnhancement] get_msg 返回空内容，判定不可用: msg_id={msg_id}")
        return False

    return True
