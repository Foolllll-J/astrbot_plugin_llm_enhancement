import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .qq_face import build_qq_face_text
from .state_manager import GroupState, MemberState

from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
try:
    from astrbot.core.utils.active_event_registry import active_event_registry
except Exception:
    active_event_registry = None

try:
    from astrbot.core.pipeline.process_stage import follow_up as follow_up_stage
except Exception:
    follow_up_stage = None


@dataclass(frozen=True)
class MergeRuntimeConfig:
    delay_sec: float
    dynamic_mode: bool
    followup_require_wake: bool
    max_count: int
    allow_multi_user: bool
    premerge_window_sec: float


@dataclass(frozen=True)
class DynamicSoftRecomputeDecision:
    inflight_seq: int
    accepted: bool
    reason: str


def load_merge_runtime_config(get_cfg: Callable[[str, Any], Any]) -> MergeRuntimeConfig:
    """Load merge settings from `request_orchestration` object only."""
    raw_merge_obj = get_cfg("request_orchestration", {})
    merge_obj = raw_merge_obj if isinstance(raw_merge_obj, dict) else {}

    delay_sec = max(0.0, float(merge_obj.get("merge_delay", 0.0)))
    raw_dynamic_mode = str(
        merge_obj.get("merge_dynamic_mode", "dynamic") or "dynamic"
    ).strip().lower()
    dynamic_mode = raw_dynamic_mode == "dynamic"
    followup_require_wake = bool(merge_obj.get("merge_followup_require_wake", False))
    max_count = max(1, int(merge_obj.get("merge_max_count", 3)))
    allow_multi_user = bool(merge_obj.get("merge_multi_user", False))
    premerge_window_sec = max(0.0, float(merge_obj.get("merge_premerge_window", 0.0)))

    return MergeRuntimeConfig(
        delay_sec=delay_sec,
        dynamic_mode=dynamic_mode,
        followup_require_wake=followup_require_wake,
        max_count=max_count,
        allow_multi_user=allow_multi_user,
        premerge_window_sec=premerge_window_sec,
    )


def normalize_event_ts(raw_ts: Any, fallback_ts: float) -> float:
    """归一化事件时间戳，兼容秒/毫秒并避免异常未来时间影响合并窗口。"""
    try:
        ts = float(raw_ts or 0.0)
    except Exception:
        return float(fallback_ts)
    if ts <= 0:
        return float(fallback_ts)
    if ts > 1e12:
        ts = ts / 1000.0
    if ts > (float(fallback_ts) + 86400.0):
        return float(fallback_ts)
    return ts


def get_event_msg_id(event: AstrMessageEvent) -> Optional[str]:
    if hasattr(event, "message_obj") and hasattr(event.message_obj, "message_id"):
        raw_msg_id = getattr(event.message_obj, "message_id", None)
        if raw_msg_id is not None:
            return str(raw_msg_id)
    return None


def is_merge_component(seg: Any) -> bool:
    if isinstance(seg, (Comp.Image, Comp.Forward, Comp.Reply, Comp.Video, Comp.File, Comp.Json, Comp.Record, Comp.Face)):
        return True
    if isinstance(seg, dict):
        seg_type = str(seg.get("type") or "").lower()
        return seg_type in {"image", "forward", "reply", "video", "file", "json", "record", "face"}
    return False


def extract_merge_components(event: AstrMessageEvent) -> list[Any]:
    chain = []
    if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
        chain = event.message_obj.message or []
    return [seg for seg in chain if is_merge_component(seg)]


def is_message_payload_component(seg: Any) -> bool:
    """是否属于可视为消息载荷的非文本组件。"""
    if isinstance(seg, (Comp.Image, Comp.Video, Comp.File, Comp.Forward, Comp.Json, Comp.Record, Comp.Face)):
        return True
    if isinstance(seg, dict):
        seg_type = str(seg.get("type") or "").lower()
        if seg_type in {"image", "video", "file", "forward", "json", "record", "face"}:
            return True
        qq_face_text = build_qq_face_text(seg)
        return bool(qq_face_text)
    return False


def build_message_buffer_from_snapshots(
    snapshots: list[Dict[str, Any]],
    default_sender_name: str,
) -> list[tuple[Optional[str], str, str]]:
    return [
        (
            str(item.get("msg_id") or "") or None,
            str(item.get("sender_name") or default_sender_name),
            str(item.get("text") or ""),
        )
        for item in snapshots
    ]


def collect_additional_components_from_snapshots(
    snapshots: list[Dict[str, Any]],
) -> list[Any]:
    additional_components: list[Any] = []
    for item in snapshots:
        for seg in item.get("components", []) or []:
            if is_merge_component(seg):
                additional_components.append(seg)
    return additional_components


async def filter_unavailable_message_buffer(
    event: AstrMessageEvent,
    message_buffer: list[tuple[Optional[str], str, str]],
) -> tuple[list[tuple[Optional[str], str, str]], list[str]]:
    filtered_buffer: list[tuple[Optional[str], str, str]] = []
    removed_msg_ids: list[str] = []
    for mid, name, content in message_buffer:
        if mid and not await is_msg_still_available(event, mid):
            removed_msg_ids.append(mid)
            continue
        filtered_buffer.append((mid, name, content))
    return filtered_buffer, removed_msg_ids


def _compress_identical_messages(
    message_buffer: list[tuple[Optional[str], str, str]],
    merged_msg: str,
) -> str:
    """若单人消息 buffer 中所有消息内容完全相同，压缩为 "{sender}发送了N..."。"""
    if len(message_buffer) <= 1:
        return merged_msg
    msgs = [msg for _, _, msg in message_buffer]
    first = msgs[0]
    if not all(m == first for m in msgs[1:]):
        return merged_msg
    count = len(msgs)
    compressed = first.replace("发送了1", f"发送了{count}", 1)
    return compressed if compressed != first else merged_msg


def apply_merged_message_to_request(
    event: AstrMessageEvent,
    req: Any,
    message_buffer: list[tuple[Optional[str], str, str]],
) -> int:
    senders = {name for _, name, _ in message_buffer}
    if len(senders) > 1:
        merged_msg = "\n".join([f"[{name}]: {msg}" for _, name, msg in message_buffer])
    else:
        merged_msg = " ".join([msg for _, _, msg in message_buffer])
        merged_msg = _compress_identical_messages(message_buffer, merged_msg)

    merged_anchor_msg_id = next(
        (str(mid).strip() for mid, _name, _content in reversed(message_buffer) if str(mid or "").strip()),
        "",
    )
    if merged_anchor_msg_id:
        anchor_value: Any = (
            int(merged_anchor_msg_id) if merged_anchor_msg_id.isdigit() else merged_anchor_msg_id
        )
        event.set_extra("_llme_merged_anchor_msg_id", merged_anchor_msg_id)
        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            try:
                setattr(message_obj, "message_id", anchor_value)
            except Exception:
                pass
            raw_message = getattr(message_obj, "raw_message", None)
            if isinstance(raw_message, dict):
                raw_message["message_id"] = anchor_value
        raw_event = getattr(event, "event", None)
        if isinstance(raw_event, dict):
            raw_event["message_id"] = anchor_value
        elif raw_event is not None:
            try:
                setattr(raw_event, "message_id", anchor_value)
            except Exception:
                pass

    event.message_str = merged_msg
    req.prompt = merged_msg
    return len(senders)


def apply_recall_to_message_buffer_state(
    group_state: GroupState,
    member: MemberState,
    message_buffer: list[tuple[Optional[str], str, str]],
    recalled_msg_id: str,
) -> dict[str, Any]:
    """在持锁前提下处理实时撤回对硬等待合并缓冲区的影响。"""
    in_pending = recalled_msg_id in member.pending_msg_ids
    in_buffer = any(str(mid or "") == recalled_msg_id for mid, _n, _c in message_buffer)
    if not (in_pending or in_buffer):
        return {"hit": False}

    before_count = len(message_buffer)
    new_buffer = [
        (mid, name, content)
        for mid, name, content in message_buffer
        if str(mid or "") != recalled_msg_id
    ]
    remove_pending_msg_id(group_state, member, recalled_msg_id)
    member.merged_msg_ids.pop(recalled_msg_id, None)
    after_count = len(new_buffer)

    is_trigger_recalled = str(member.trigger_msg_id or "") == recalled_msg_id
    if is_trigger_recalled and after_count > 0:
        member.trigger_msg_id = str(new_buffer[0][0] or "")
    should_stop = after_count <= 0
    if should_stop:
        member.cancel_merge = True
        clear_pending_msg_ids(group_state, member)

    return {
        "hit": True,
        "in_pending": in_pending,
        "in_buffer": in_buffer,
        "before_count": before_count,
        "after_count": after_count,
        "is_trigger_recalled": is_trigger_recalled,
        "should_stop": should_stop,
        "new_trigger_msg_id": member.trigger_msg_id,
        "message_buffer": new_buffer,
    }


def evaluate_followup_collectability(
    ev: AstrMessageEvent,
    gid: str,
    uid: str,
    allow_multi_user: bool,
    followup_require_wake: bool,
) -> tuple[bool, str]:
    """判断后续事件是否应进入硬等待合并窗口。"""
    if gid:
        if ev.get_group_id() != gid:
            return False, "group_mismatch"
        if not allow_multi_user and ev.get_sender_id() != uid:
            return False, "sender_mismatch"
    else:
        if ev.get_sender_id() != uid:
            return False, "private_sender_mismatch"

    chain = ev.message_obj.message if (hasattr(ev, "message_obj") and hasattr(ev.message_obj, "message")) else []
    if not ev.message_str and not any(is_message_payload_component(seg) for seg in chain):
        return False, "empty_payload"

    if followup_require_wake and (not ev.is_at_or_wake_command):
        return False, "wake_required"

    return True, "ok"


def is_duplicate_followup_message(
    message_buffer: list[tuple[Optional[str], str, str]],
    ev: AstrMessageEvent,
    uid: str,
) -> bool:
    """防止单条重复触发导致重复并入。"""
    return (
        len(message_buffer) == 1
        and ev.message_str == message_buffer[0][2]
        and ev.get_sender_id() == uid
    )


async def append_followup_to_merge_buffer(
    group_state: GroupState,
    member: MemberState,
    message_buffer: list[tuple[Optional[str], str, str]],
    additional_components: list[Any],
    ev: AstrMessageEvent,
    merged_skip_ttl: float,
) -> tuple[list[tuple[Optional[str], str, str]], list[Any], Optional[str]]:
    """把后续事件追加到硬等待合并缓冲，并同步 pending/merged 索引。"""
    new_msg_id = None
    if hasattr(ev, "message_obj") and hasattr(ev.message_obj, "message_id"):
        new_msg_id = str(ev.message_obj.message_id)

    if new_msg_id:
        async with member.lock:
            add_pending_msg_id(group_state, member, new_msg_id)
            member.merged_msg_ids[new_msg_id] = time.time() + merged_skip_ttl

    new_message_buffer = list(message_buffer)
    new_message_buffer.append((new_msg_id, ev.get_sender_name(), ev.message_str))

    new_additional_components = list(additional_components)
    chain = ev.message_obj.message if (hasattr(ev, "message_obj") and hasattr(ev.message_obj, "message")) else []
    for seg in chain:
        if is_merge_component(seg):
            new_additional_components.append(seg)

    return new_message_buffer, new_additional_components, new_msg_id


def has_recent_wake_in_window(
    member: MemberState,
    now_ts: float,
    window_sec: float,
) -> bool:
    """是否存在窗口内的最近唤醒消息（用于预请求阶段的动态跟进捕获）。"""
    if window_sec <= 0:
        return False

    latest_ts = 0.0
    for item in member.recent_wake_msgs:
        ts = float(item.get("ts", 0.0) or 0.0)
        if ts > latest_ts:
            latest_ts = ts
    if member.last_wake_ts > latest_ts:
        latest_ts = member.last_wake_ts
    if latest_ts <= 0:
        return False
    return (now_ts - latest_ts) <= window_sec


def ensure_snapshot_merge_key(snapshot: Dict[str, Any]) -> str:
    existing_key = str(snapshot.get("_merge_key") or "").strip()
    if existing_key:
        return existing_key
    msg_id = str(snapshot.get("msg_id") or "").strip()
    if msg_id:
        key = f"id:{msg_id}"
    else:
        uid = str(snapshot.get("uid") or "")
        ts = int(float(snapshot.get("ts") or 0.0) * 1000)
        text_len = len(str(snapshot.get("text") or ""))
        key = f"noid:{uid}:{ts}:{text_len}"
    snapshot["_merge_key"] = key
    return key


def upsert_dynamic_unresolved_snapshot(
    member: MemberState,
    snapshot: Dict[str, Any],
    max_keep: int = 50,
) -> bool:
    key = ensure_snapshot_merge_key(snapshot)
    for idx, item in enumerate(member.dynamic_unresolved_msgs):
        if ensure_snapshot_merge_key(item) == key:
            old_ts = float(item.get("ts") or 0.0)
            snapshot["ts"] = min(old_ts, float(snapshot.get("ts") or old_ts))
            member.dynamic_unresolved_msgs[idx] = snapshot
            inserted_new = False
            break
    else:
        member.dynamic_unresolved_msgs.append(snapshot)
        inserted_new = True

    member.dynamic_unresolved_msgs.sort(key=lambda x: float(x.get("ts") or 0.0))
    if len(member.dynamic_unresolved_msgs) > max_keep:
        member.dynamic_unresolved_msgs = member.dynamic_unresolved_msgs[-max_keep:]
    return inserted_new


def reset_dynamic_capture_session(member: MemberState) -> None:
    member.merge_start_ts = 0.0
    member.dynamic_capture_count = 0


def reset_member_state(member: MemberState) -> None:
    """重置成员的所有动态合并/唤醒状态，相当于该成员刚加入群时的状态"""
    member.in_merging = False
    member.pending_msg_ids.clear()
    member.cancel_merge = False
    member.trigger_msg_id = None
    member.recent_wake_msgs.clear()
    member.premerge_msgs.clear()
    member.dynamic_attached_premerge_msgs.clear()
    member.last_wake_ts = 0.0
    member.merge_start_ts = 0.0
    member.merged_msg_ids.clear()
    member.dynamic_unresolved_msgs.clear()
    member.dynamic_request_seq = 0
    member.dynamic_inflight_seq = 0
    member.dynamic_discard_before_seq = 0
    member.dynamic_discarded_response_cache.clear()
    member.dynamic_capture_count = 0
    member.dynamic_source_event = None
    member.dynamic_source_event_seq = 0
    member.dynamic_requeue_pending_seq = 0
    member.last_response = 0.0


def reset_group_state(group_state: GroupState) -> None:
    """重置群组的所有状态"""
    group_state.dynamic_owner_uid = None
    group_state.wake_extend_consumed_ref_ts = 0.0
    group_state.active_wake_new_msg_count = 0
    group_state.prob_wake_pending_count = 0
    group_state.prob_wake_no_reply_count = 0
    group_state.prob_wake_last_check_ts = 0.0
    group_state.wake_extend_batch_count = 0
    group_state.last_response_uid = None
    group_state.last_response_ts = 0.0
    group_state.pending_msg_index.clear()
    group_state.last_user_interaction.clear()
    group_state.context_bot_last_replied_to_uid = ""


def prune_member_msg_cache(member: MemberState, keep_sec: float, ref_ts: float = 0.0) -> None:
    """清理过期的消息缓存。ref_ts 用于判断消息列表的相对过期，merged_msg_ids 使用真实时间判断。"""
    # 消息列表用 ref_ts（消息发送时间）判断，避免处理延迟导致历史消息被误删
    member.recent_wake_msgs = [
        item for item in member.recent_wake_msgs
        if ref_ts - float(item.get("ts") or 0.0) <= keep_sec
    ]
    member.premerge_msgs = [
        item for item in member.premerge_msgs
        if ref_ts - float(item.get("ts") or 0.0) <= keep_sec
    ]
    member.dynamic_attached_premerge_msgs = [
        item
        for item in member.dynamic_attached_premerge_msgs
        if ref_ts - float(item.get("ts") or 0.0) <= keep_sec
    ]
    member.dynamic_unresolved_msgs = [
        item
        for item in member.dynamic_unresolved_msgs
        if ref_ts - float(item.get("ts") or 0.0) <= keep_sec
    ]
    # merged_msg_ids 存储的是绝对过期时间，必须用真实时间判断
    now_ts = time.time()
    expired_ids = [mid for mid, exp_ts in member.merged_msg_ids.items() if exp_ts <= now_ts]
    for mid in expired_ids:
        member.merged_msg_ids.pop(mid, None)


def build_event_snapshot(event: AstrMessageEvent, gid: str, uid: str, ts: Optional[float] = None) -> Dict[str, Any]:
    """构建事件快照。ts 参数用于传入消息的实际发送时间，避免用处理时间替代。"""
    chain = []
    if hasattr(event, "message_obj") and hasattr(event.message_obj, "message") and event.message_obj.message:
        chain = event.message_obj.message
    components = [seg for seg in chain if is_merge_component(seg)]
    return {
        "msg_id": get_event_msg_id(event),
        "ts": ts if ts is not None else time.time(),
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
            old_ts = float(item.get("ts") or 0.0)
            snapshot["ts"] = min(old_ts, float(snapshot.get("ts") or 0.0))
            member.recent_wake_msgs[idx] = snapshot
            return
    member.recent_wake_msgs.append(snapshot)


def upsert_premerge_snapshot(
    member: MemberState,
    snapshot: Dict[str, Any],
    max_keep: int,
) -> bool:
    """记录唤醒前的非唤醒消息，按时间排序并保留最近候选。"""
    if max_keep <= 0:
        return False
    key = ensure_snapshot_merge_key(snapshot)
    for idx, item in enumerate(member.premerge_msgs):
        if ensure_snapshot_merge_key(item) == key:
            old_ts = float(item.get("ts") or 0.0)
            snapshot["ts"] = min(old_ts, float(snapshot.get("ts") or old_ts))
            member.premerge_msgs[idx] = snapshot
            inserted_new = False
            break
    else:
        member.premerge_msgs.append(snapshot)
        inserted_new = True

    member.premerge_msgs.sort(key=lambda x: float(x.get("ts") or 0.0))
    if len(member.premerge_msgs) > max_keep:
        member.premerge_msgs = member.premerge_msgs[-max_keep:]
    return inserted_new


def consume_premerge_snapshots_for_trigger(
    member: MemberState,
    *,
    trigger_ts: float,
    window_sec: float,
    max_items: int,
) -> list[Dict[str, Any]]:
    """取出触发消息之前、窗口内、距离最近的预合并消息。"""
    if max_items <= 0 or window_sec <= 0:
        return []

    eligible = [
        item
        for item in member.premerge_msgs
        if 0.0 <= (trigger_ts - float(item.get("ts") or 0.0)) <= window_sec
    ]
    if not eligible:
        return []

    selected = eligible[-max_items:]
    selected_keys = {ensure_snapshot_merge_key(item) for item in selected}
    member.premerge_msgs = [
        item
        for item in member.premerge_msgs
        if ensure_snapshot_merge_key(item) not in selected_keys
    ]
    return selected


def attach_dynamic_premerge_snapshots(
    member: MemberState,
    snapshots: Optional[list[Dict[str, Any]]],
) -> int:
    """将本轮命中的预合并消息附着到当前动态会话，供后续软重算/重排复用。"""
    if not snapshots:
        return 0

    existing: dict[str, Dict[str, Any]] = {}
    for item in member.dynamic_attached_premerge_msgs:
        existing[ensure_snapshot_merge_key(item)] = item

    attached_new = 0
    for raw_item in snapshots:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        key = ensure_snapshot_merge_key(item)
        old_item = existing.get(key)
        if old_item is None:
            existing[key] = item
            attached_new += 1
            continue
        old_ts = float(old_item.get("ts") or 0.0)
        item["ts"] = min(old_ts, float(item.get("ts") or old_ts))
        existing[key] = item

    member.dynamic_attached_premerge_msgs = sorted(
        existing.values(),
        key=lambda x: float(x.get("ts") or 0.0),
    )
    return attached_new


def prepare_initial_merge_snapshots(
    event: AstrMessageEvent,
    gid: str,
    uid: str,
    member: MemberState,
    merge_delay: float,
    merged_window_tolerance: float,
    merged_skip_ttl: float,
    merge_max_count: int,
    add_pending_msg_id: Callable[[str], None],
    event_ts: float = 0.0,
) -> tuple[list[Dict[str, Any]], Optional[str]]:
    """预选本次合并窗口内的消息快照，并更新 member 的已并入索引。event_ts 用于传入消息发送时间。"""
    current_snapshot = build_event_snapshot(event, gid, uid, ts=event_ts or None)
    current_msg_id = str(current_snapshot.get("msg_id") or "")
    if current_msg_id:
        upsert_recent_wake_snapshot(member, current_snapshot)

    merge_start_ts = float(getattr(member, "merge_start_ts", 0.0) or 0.0)
    current_ts = float(current_snapshot.get("ts") or 0.0)
    if current_msg_id:
        for item in member.recent_wake_msgs:
            if str(item.get("msg_id") or "") == current_msg_id:
                current_ts = float(item.get("ts") or 0.0) or current_ts
                break

    nearby_ts_list = [
        float(item.get("ts") or 0.0)
        for item in member.recent_wake_msgs
        if (
            (merge_start_ts <= 0.0 or float(item.get("ts") or 0.0) >= merge_start_ts)
            and abs(float(item.get("ts") or 0.0) - current_ts) <= merge_delay
        )
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
        if merge_start_ts > 0.0 and ts < merge_start_ts:
            continue
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
    if len(preselected_snapshots) > merge_max_count:
        preselected_snapshots = preselected_snapshots[:merge_max_count]

    for item in preselected_snapshots:
        msg_id = str(item.get("msg_id") or "")
        if msg_id:
            add_pending_msg_id(msg_id)
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


def member_contains_msg_id(member: MemberState, msg_id: str) -> bool:
    if not msg_id:
        return False
    if msg_id in member.pending_msg_ids:
        return True
    if msg_id in member.merged_msg_ids:
        return True
    if any(str(item.get("msg_id") or "") == msg_id for item in member.recent_wake_msgs):
        return True
    if any(str(item.get("msg_id") or "") == msg_id for item in member.premerge_msgs):
        return True
    if any(str(item.get("msg_id") or "") == msg_id for item in member.dynamic_attached_premerge_msgs):
        return True
    if any(str(item.get("msg_id") or "") == msg_id for item in member.dynamic_unresolved_msgs):
        return True
    return False


def remove_recalled_msg_from_member(
    group_state: GroupState,
    member: MemberState,
    msg_id: str,
) -> dict[str, Any]:
    """将撤回消息从成员相关状态中移除，并返回影响摘要。"""
    summary: dict[str, Any] = {
        "removed_pending": False,
        "removed_merged": False,
        "removed_recent": 0,
        "removed_premerge": 0,
        "removed_dynamic_attached_premerge": 0,
        "removed_dynamic_unresolved": 0,
        "trigger_recalled": False,
        "trigger_replaced": None,
        "marked_cancel": False,
        "inflight_seq": int(member.dynamic_inflight_seq or 0),
        "remaining_pending": 0,
        "remaining_dynamic_unresolved": 0,
    }
    if not msg_id:
        return summary

    if msg_id in member.pending_msg_ids:
        remove_pending_msg_id(group_state, member, msg_id)
        summary["removed_pending"] = True

    if msg_id in member.merged_msg_ids:
        member.merged_msg_ids.pop(msg_id, None)
        summary["removed_merged"] = True

    before_recent = len(member.recent_wake_msgs)
    member.recent_wake_msgs = [
        item for item in member.recent_wake_msgs if str(item.get("msg_id") or "") != msg_id
    ]
    summary["removed_recent"] = max(0, before_recent - len(member.recent_wake_msgs))

    before_premerge = len(member.premerge_msgs)
    member.premerge_msgs = [
        item for item in member.premerge_msgs if str(item.get("msg_id") or "") != msg_id
    ]
    summary["removed_premerge"] = max(0, before_premerge - len(member.premerge_msgs))

    before_attached_premerge = len(member.dynamic_attached_premerge_msgs)
    member.dynamic_attached_premerge_msgs = [
        item for item in member.dynamic_attached_premerge_msgs if str(item.get("msg_id") or "") != msg_id
    ]
    summary["removed_dynamic_attached_premerge"] = max(
        0,
        before_attached_premerge - len(member.dynamic_attached_premerge_msgs),
    )

    before_dynamic = len(member.dynamic_unresolved_msgs)
    member.dynamic_unresolved_msgs = [
        item
        for item in member.dynamic_unresolved_msgs
        if str(item.get("msg_id") or "") != msg_id
    ]
    summary["removed_dynamic_unresolved"] = max(
        0,
        before_dynamic - len(member.dynamic_unresolved_msgs),
    )
    summary["remaining_pending"] = len(member.pending_msg_ids)
    summary["remaining_dynamic_unresolved"] = len(member.dynamic_unresolved_msgs)

    summary["trigger_recalled"] = str(member.trigger_msg_id or "") == msg_id
    if summary["trigger_recalled"]:
        new_trigger = None
        for item in member.dynamic_unresolved_msgs:
            candidate = str(item.get("msg_id") or "").strip()
            if candidate:
                new_trigger = candidate
                break
        member.trigger_msg_id = new_trigger
        summary["trigger_replaced"] = new_trigger

    if (
        member.dynamic_inflight_seq > 0
        and (
            summary["remaining_pending"] <= 0
            and summary["remaining_dynamic_unresolved"] <= 0
        )
    ):
        member.cancel_merge = True
        summary["marked_cancel"] = True

    return summary


def prepare_dynamic_merge_batch(
    member: MemberState,
    group_state: GroupState,
    current_snapshot: Dict[str, Any],
    premerge_snapshots: Optional[list[Dict[str, Any]]],
    uid: str,
    allow_multi_user: bool,
    merge_max_count: int,
    merge_delay: float,
    merged_skip_ttl: float,
) -> tuple[list[Dict[str, Any]], list[str], int, int, Optional[str]]:
    had_inflight = member.dynamic_inflight_seq > 0
    attach_dynamic_premerge_snapshots(member, premerge_snapshots)
    upsert_dynamic_unresolved_snapshot(member, current_snapshot)

    member.dynamic_request_seq += 1
    request_seq = member.dynamic_request_seq
    member.dynamic_inflight_seq = request_seq
    if not group_state.dynamic_owner_uid:
        group_state.dynamic_owner_uid = member.uid

    merge_start_ts = float(getattr(member, "merge_start_ts", 0.0) or 0.0)
    if merge_start_ts > 0.0:
        member.dynamic_unresolved_msgs = [
            item
            for item in member.dynamic_unresolved_msgs
            if float(item.get("ts") or 0.0) >= merge_start_ts
        ]

    # 判定当前触发消息是否已经跨越了时间断层
    current_ts = float(current_snapshot.get("ts") or 0.0)
    if merge_start_ts > 0.0 and (current_ts - merge_start_ts) > merge_delay:
        # 既然已经超过 merge_delay，说明旧池子里的全是"僵尸消息"
        logger.debug(
            f"[LLMEnhancement] 检测到时间断层 ({(current_ts - merge_start_ts):.1f}s > {merge_delay}s)，重置动态池"
        )
        member.dynamic_unresolved_msgs = [current_snapshot]  # 只保留当前消息
        member.merge_start_ts = current_ts  # 重置起点
        member.dynamic_attached_premerge_msgs = []
        attach_dynamic_premerge_snapshots(member, premerge_snapshots)
        member.dynamic_capture_count = 1
        # 同步清理全局索引
        clear_pending_msg_ids(group_state, member)

    attached_snapshots: list[Dict[str, Any]] = list(member.dynamic_attached_premerge_msgs)
    selected_unresolved_snapshots: list[Dict[str, Any]] = []
    current_key = ensure_snapshot_merge_key(current_snapshot)
    for item in member.dynamic_unresolved_msgs:
        item_uid = str(item.get("uid") or "")
        if not allow_multi_user and item_uid and item_uid != uid:
            continue
        ts = float(item.get("ts") or 0.0)
        if merge_start_ts > 0.0 and ts < merge_start_ts:
            continue
        is_current_trigger = ensure_snapshot_merge_key(item) == current_key
        if not is_current_trigger and merge_delay > 0:
            if (ts - merge_start_ts) > merge_delay:
                continue
        elif ensure_snapshot_merge_key(item) != current_key:
            continue
        text = str(item.get("text") or "")
        components = item.get("components", []) or []
        # 允许当前触发快照兜底进入，避免动态合并在部分平台上因空文本快照被全量过滤后误取消请求。
        if not text and not components and ensure_snapshot_merge_key(item) != current_key:
            continue
        selected_unresolved_snapshots.append(item)

    if not attached_snapshots and not selected_unresolved_snapshots:
        # 兜底：至少保留当前触发快照，避免动态合并把整批请求误判为空而取消。
        selected_unresolved_snapshots = [current_snapshot]

    selected_unresolved_snapshots.sort(key=lambda x: float(x.get("ts") or 0.0))
    if len(selected_unresolved_snapshots) > merge_max_count:
        selected_unresolved_snapshots = selected_unresolved_snapshots[-merge_max_count:]

    selected_snapshots: list[Dict[str, Any]] = list(attached_snapshots) + selected_unresolved_snapshots
    selected_snapshots.sort(key=lambda x: float(x.get("ts") or 0.0))

    deduped_snapshots: list[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in selected_snapshots:
        key = ensure_snapshot_merge_key(item)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_snapshots.append(item)
    selected_snapshots = deduped_snapshots

    clear_pending_msg_ids(group_state, member)
    selected_keys: list[str] = []
    for item in selected_snapshots:
        msg_id = str(item.get("msg_id") or "").strip()
        if msg_id:
            add_pending_msg_id(group_state, member, msg_id)
            member.merged_msg_ids[msg_id] = time.time() + merged_skip_ttl
        selected_keys.append(ensure_snapshot_merge_key(item))

    trigger_msg_id = str(current_snapshot.get("msg_id") or "").strip() or (
        str(selected_snapshots[-1].get("msg_id") or "").strip() if selected_snapshots else None
    )

    if not had_inflight:
        # 首轮触发：用触发消息的 ts 作为合并起点
        member.merge_start_ts = float(current_snapshot.get("ts") or 0.0) or 0.0
        member.dynamic_capture_count = 1

    unresolved_count = len(member.dynamic_unresolved_msgs)
    return selected_snapshots, selected_keys, request_seq, unresolved_count, trigger_msg_id


def select_dynamic_owner_uid(
    own_inflight_seq: int,
    dynamic_owner_uid: Optional[str],
    incoming_uid: str,
    allow_multi_user: bool,
) -> Optional[str]:
    if own_inflight_seq > 0:
        return incoming_uid
    if allow_multi_user and dynamic_owner_uid and dynamic_owner_uid != incoming_uid:
        return dynamic_owner_uid
    return None


async def mark_dynamic_soft_recompute(
    target_member: MemberState,
    snapshot: Dict[str, Any],
    *,
    merge_delay: float,
    merge_max_count: int,
) -> DynamicSoftRecomputeDecision:
    async with target_member.lock:
        inflight_seq = target_member.dynamic_inflight_seq
        if inflight_seq <= 0:
            return DynamicSoftRecomputeDecision(
                inflight_seq=0,
                accepted=False,
                reason="no_inflight",
            )
        snapshot_ts = float(snapshot.get("ts") or 0.0)
        if target_member.merge_start_ts <= 0.0:
            if target_member.dynamic_unresolved_msgs:
                target_member.merge_start_ts = min(
                    float(item.get("ts") or 0.0)
                    for item in target_member.dynamic_unresolved_msgs
                )
            else:
                # 【修改】：直接用快照时间初始化 merge_start_ts，避免依赖系统时间
                target_member.merge_start_ts = snapshot_ts
        if target_member.dynamic_capture_count <= 0:
            target_member.dynamic_capture_count = len(
                {
                    ensure_snapshot_merge_key(item)
                    for item in target_member.dynamic_unresolved_msgs
                }
            )

        if merge_delay > 0.0 and (
            snapshot_ts - float(target_member.merge_start_ts)
        ) >= float(merge_delay):
            return DynamicSoftRecomputeDecision(
                inflight_seq=inflight_seq,
                accepted=False,
                reason="deadline_reached",
            )
        current_unresolved_count = len(target_member.dynamic_unresolved_msgs)
        if merge_max_count > 0 and current_unresolved_count >= merge_max_count:
            return DynamicSoftRecomputeDecision(
                inflight_seq=inflight_seq,
                accepted=False,
                reason="max_count_reached",
            )
        inserted_new = upsert_dynamic_unresolved_snapshot(target_member, snapshot)
        if not inserted_new:
            return DynamicSoftRecomputeDecision(
                inflight_seq=inflight_seq,
                accepted=False,
                reason="duplicate_snapshot",
            )
        target_member.dynamic_capture_count += 1
        target_member.dynamic_discard_before_seq = max(
            target_member.dynamic_discard_before_seq,
            inflight_seq,
        )
        return DynamicSoftRecomputeDecision(
            inflight_seq=inflight_seq,
            accepted=True,
            reason="accepted",
        )


def request_dynamic_recompute_stop(
    event: AstrMessageEvent,
    *,
    inflight_seq: int,
    owner_uid: str,
) -> int:
    """Stop active events for this UMO (hard stop), excluding current event."""
    if active_event_registry is None:
        return 0
    try:
        stop_fn = getattr(active_event_registry, "stop_all", None)
        if callable(stop_fn):
            stopped_count = int(
                stop_fn(
                    event.unified_msg_origin,
                    exclude=event,
                )
                or 0
            )
            stop_mode = "hard_stop_all"
        else:
            stopped_count = int(
                active_event_registry.request_agent_stop_all(
                    event.unified_msg_origin,
                    exclude=event,
                )
                or 0
            )
            stop_mode = "request_agent_stop_all"
        if stopped_count > 0:
            logger.debug(
                "[LLMEnhancement] 动态软重算已停止进行中的任务："
                f"umo={event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}, "
                f"stopped_count={stopped_count}, mode={stop_mode}"
            )
        else:
            logger.debug(
                "[LLMEnhancement] 动态软重算未找到可停止的进行中任务："
                f"umo={event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}, mode={stop_mode}"
            )
        return stopped_count
    except Exception as e:
        logger.warning(
            "[LLMEnhancement] 动态软重算请求停止任务失败："
            f"umo={event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}, error={e}"
        )
        return 0


def _has_active_follow_up_runner(umo: str) -> bool:
    if follow_up_stage is None:
        return False
    try:
        runners = getattr(follow_up_stage, "_ACTIVE_AGENT_RUNNERS", None)
        if isinstance(runners, dict):
            return runners.get(umo) is not None
    except Exception:
        return False
    return False


def schedule_dynamic_recompute_requeue(
    source_event: AstrMessageEvent,
    *,
    event_queue: Any,
    owner_uid: str,
    inflight_seq: int,
    delay_sec: float = 0.35,
    max_attempts: int = 2,
) -> bool:
    """Requeue a saved message event as a fresh event after stop request."""
    try:
        attempt = int(source_event.get_extra("_llme_dynamic_requeue_attempt") or 0)
    except Exception:
        attempt = 0
    if attempt >= max_attempts:
        logger.warning(
            "[LLMEnhancement] 动态软重算重排已达上限，放弃再次入队："
            f"umo={source_event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}, "
            f"attempt={attempt}, max_attempts={max_attempts}"
        )
        return False

    if event_queue is None or not hasattr(event_queue, "put_nowait"):
        logger.warning(
            "[LLMEnhancement] 动态软重算重排入队失败：事件队列不可用，"
            f"umo={source_event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}"
        )
        return False

    try:
        replay_event = copy.copy(source_event)
        # copy.copy(event) 会共享 _result / _extras 等运行态；若原事件随后 stop_event，
        # 重排事件会继承已停止状态，导致框架跳过默认 LLM 请求。
        replay_event._result = None
        replay_event._extras = dict(source_event.get_extra(None) or {})
        replay_event._has_send_oper = False
        replay_event.call_llm = False
        replay_event.is_wake = False
        replay_event.is_at_or_wake_command = False
        replay_event.set_extra("agent_stop_requested", False)
        replay_event.set_extra("_llme_dynamic_requeue_attempt", attempt + 1)
        replay_event.set_extra("_llme_dynamic_requeued", True)
        replay_event.set_extra("_llme_dynamic_requeue_from_seq", int(inflight_seq))
        replay_event.set_extra("_llme_dynamic_followup", True)
        replay_event.set_extra("_llme_dynamic_state_uid", str(owner_uid))

        async def _enqueue_later() -> None:
            await asyncio.sleep(max(0.0, float(delay_sec)))
            wait_start = time.time()
            waited_sec = 0.0
            while _has_active_follow_up_runner(source_event.unified_msg_origin):
                waited_sec = time.time() - wait_start
                if waited_sec >= 2.0:
                    logger.warning(
                        "[LLMEnhancement] 动态软重算重排等待 active runner 退出超时，继续强制入队："
                        f"umo={source_event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}, "
                        f"attempt={attempt + 1}, waited_sec={waited_sec:.2f}"
                    )
                    break
                await asyncio.sleep(0.05)
            event_queue.put_nowait(replay_event)
            logger.debug(
                "[LLMEnhancement] 动态软重算消息已重排入队："
                f"umo={source_event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}, "
                f"attempt={attempt + 1}, delay_sec={delay_sec:.2f}, waited_active_runner={waited_sec:.2f}"
            )

        asyncio.create_task(_enqueue_later())
        return True
    except Exception as e:
        logger.warning(
            "[LLMEnhancement] 动态软重算重排入队失败："
            f"umo={event.unified_msg_origin}, owner_uid={owner_uid}, inflight_seq={inflight_seq}, error={e}"
        )
        return False


def drop_dynamic_batch_from_unresolved(member: MemberState, batch_keys: list[str]) -> int:
    key_set = {str(k) for k in batch_keys if str(k or "").strip()}
    if not key_set:
        return 0
    before_count = len(member.dynamic_unresolved_msgs)
    member.dynamic_unresolved_msgs = [
        item
        for item in member.dynamic_unresolved_msgs
        if ensure_snapshot_merge_key(item) not in key_set
    ]
    return max(0, before_count - len(member.dynamic_unresolved_msgs))


async def execute_dynamic_merge(
    event: AstrMessageEvent,
    req: Any,
    gid: str,
    uid: str,
    member: MemberState,
    group_state: GroupState,
    allow_multi_user: bool,
    merge_delay: float,
    merge_max_count: int,
    is_recent_recalled: Optional[Callable[[str], bool]] = None,
    event_ts: Optional[float] = None,
) -> dict[str, Any]:
    ttl_base = max(merge_delay, 10.0)
    cache_keep_sec = max(ttl_base * 6, 60.0)
    merged_skip_ttl = max(ttl_base * 6, 120.0)
    current_snapshot = build_event_snapshot(event, gid, uid, ts=event_ts)
    ensure_snapshot_merge_key(current_snapshot)
    raw_premerge_snapshots = event.get_extra("_llme_premerge_snapshots", default=[]) or []
    premerge_snapshots = [
        item for item in raw_premerge_snapshots
        if isinstance(item, dict)
    ]
    event.set_extra("_llme_premerge_snapshots", [])

    selected_snapshots: list[Dict[str, Any]] = []
    selected_keys: list[str] = []
    removed_recalled_ids: list[str] = []
    removed_unavailable_ids: list[str] = []
    request_seq = 0
    unresolved_count = 0
    try:
        async with member.lock:
            prune_member_msg_cache(member, keep_sec=cache_keep_sec, ref_ts=event_ts or 0.0)
            member.cancel_merge = False
            member.in_merging = True
            (
                selected_snapshots,
                selected_keys,
                request_seq,
                unresolved_count,
                member.trigger_msg_id,
            ) = prepare_dynamic_merge_batch(
                member=member,
                group_state=group_state,
                current_snapshot=current_snapshot,
                premerge_snapshots=premerge_snapshots,
                uid=uid,
                allow_multi_user=allow_multi_user,
                merge_max_count=merge_max_count,
                merge_delay=merge_delay,
                merged_skip_ttl=merged_skip_ttl,
            )
            member.dynamic_source_event = event
            member.dynamic_source_event_seq = int(request_seq or 0)

            if is_recent_recalled and selected_snapshots:
                filtered_snapshots: list[Dict[str, Any]] = []
                removed_set: set[str] = set()
                for item in selected_snapshots:
                    mid = str(item.get("msg_id") or "").strip()
                    if mid and is_recent_recalled(mid):
                        removed_recalled_ids.append(mid)
                        removed_set.add(mid)
                        remove_pending_msg_id(group_state, member, mid)
                        member.merged_msg_ids.pop(mid, None)
                        continue
                    filtered_snapshots.append(item)
                selected_snapshots = filtered_snapshots
                selected_keys = [ensure_snapshot_merge_key(item) for item in selected_snapshots]
                if removed_set:
                    member.dynamic_unresolved_msgs = [
                        item
                        for item in member.dynamic_unresolved_msgs
                        if str(item.get("msg_id") or "").strip() not in removed_set
                    ]
                    member.dynamic_attached_premerge_msgs = [
                        item
                        for item in member.dynamic_attached_premerge_msgs
                        if str(item.get("msg_id") or "").strip() not in removed_set
                    ]
                    member.recent_wake_msgs = [
                        item
                        for item in member.recent_wake_msgs
                        if str(item.get("msg_id") or "").strip() not in removed_set
                    ]

                if removed_recalled_ids and str(member.trigger_msg_id or "") in set(removed_recalled_ids):
                    member.trigger_msg_id = (
                        str(selected_snapshots[0].get("msg_id") or "").strip() if selected_snapshots else None
                    )
    finally:
        async with member.lock:
            member.in_merging = False

    # 动态模式兜底校验：与硬等待一致，最终用 get_msg 过滤不可用/已撤回消息。
    if selected_snapshots:
        filtered_snapshots: list[Dict[str, Any]] = []
        for item in selected_snapshots:
            mid = str(item.get("msg_id") or "").strip()
            if mid and not await is_msg_still_available(event, mid):
                removed_unavailable_ids.append(mid)
                continue
            filtered_snapshots.append(item)

        if removed_unavailable_ids:
            removed_unavailable_set = set(removed_unavailable_ids)
            async with member.lock:
                for mid in removed_unavailable_set:
                    remove_pending_msg_id(group_state, member, mid)
                    member.merged_msg_ids.pop(mid, None)
                member.dynamic_unresolved_msgs = [
                    item
                    for item in member.dynamic_unresolved_msgs
                    if str(item.get("msg_id") or "").strip() not in removed_unavailable_set
                ]
                member.dynamic_attached_premerge_msgs = [
                    item
                    for item in member.dynamic_attached_premerge_msgs
                    if str(item.get("msg_id") or "").strip() not in removed_unavailable_set
                ]
                member.recent_wake_msgs = [
                    item
                    for item in member.recent_wake_msgs
                    if str(item.get("msg_id") or "").strip() not in removed_unavailable_set
                ]
                if str(member.trigger_msg_id or "") in removed_unavailable_set:
                    member.trigger_msg_id = (
                        str(filtered_snapshots[0].get("msg_id") or "").strip() if filtered_snapshots else None
                    )

            selected_snapshots = filtered_snapshots
            selected_keys = [ensure_snapshot_merge_key(item) for item in selected_snapshots]

    event.set_extra("_llme_dynamic_request_seq", request_seq)
    event.set_extra("_llme_dynamic_batch_keys", selected_keys)
    event.set_extra(
        "_llme_dynamic_batch_msg_ids",
        [str(item.get("msg_id") or "").strip() for item in selected_snapshots if str(item.get("msg_id") or "").strip()],
    )

    if not selected_snapshots:
        async with member.lock:
            member.cancel_merge = True
            member.dynamic_attached_premerge_msgs = []
            if member.dynamic_source_event_seq == int(request_seq or 0):
                member.dynamic_source_event = None
                member.dynamic_source_event_seq = 0
        event.stop_event()
        return {
            "cancelled": True,
            "request_seq": request_seq,
            "selected_keys": selected_keys,
            "unresolved_count": unresolved_count,
            "message_count": 0,
            "sender_count": 0,
            "additional_components": [],
            "removed_recalled_ids": removed_recalled_ids,
            "removed_unavailable_ids": removed_unavailable_ids,
            "selected_msg_ids": [],
        }

    message_buffer = build_message_buffer_from_snapshots(
        selected_snapshots,
        default_sender_name=event.get_sender_name(),
    )
    additional_components = collect_additional_components_from_snapshots(selected_snapshots)
    sender_count = apply_merged_message_to_request(event, req, message_buffer)

    return {
        "cancelled": False,
        "request_seq": request_seq,
        "selected_keys": selected_keys,
        "selected_msg_ids": [str(item.get("msg_id") or "").strip() for item in selected_snapshots if str(item.get("msg_id") or "").strip()],
        "unresolved_count": unresolved_count,
        "message_count": len(message_buffer),
        "sender_count": sender_count,
        "additional_components": additional_components,
        "removed_recalled_ids": removed_recalled_ids,
        "removed_unavailable_ids": removed_unavailable_ids,
    }


async def is_msg_still_available(event: AstrMessageEvent, msg_id: str) -> bool:
    """最终阶段通过协议端校验消息是否仍可获取（仅 AIOCQHTTP）。"""
    if not msg_id:
        return True
    if not isinstance(event, AiocqhttpMessageEvent):
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
            logger.debug(f"[LLMEnhancement] get_msg 指示消息不可用/已撤回: msg_id={msg_id}")
            return False
        logger.debug(f"[LLMEnhancement] get_msg 调用异常但按可用处理: msg_id={msg_id}, err={e}")
        return True

    if isinstance(detail, dict):
        msg_status = detail.get("status")
        if msg_status == "deleted":
            logger.debug(f"[LLMEnhancement] get_msg status=deleted，判定消息已撤回: msg_id={msg_id}")
            return False

    if isinstance(detail, dict) and isinstance(detail.get("data"), dict):
        detail = detail["data"]

    if not isinstance(detail, dict):
        logger.debug(f"[LLMEnhancement] get_msg 返回结构异常，判定不可用: msg_id={msg_id}")
        return False

    msg_content = detail.get("message")
    if msg_content is None:
        logger.debug(f"[LLMEnhancement] get_msg 未返回 message 字段，判定不可用: msg_id={msg_id}")
        return False
    if isinstance(msg_content, (list, str)) and len(msg_content) == 0:
        logger.debug(f"[LLMEnhancement] get_msg 返回空内容，判定不可用: msg_id={msg_id}")
        return False

    return True
