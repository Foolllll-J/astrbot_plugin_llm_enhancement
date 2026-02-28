import json
import hashlib
import secrets
import time
import re
from typing import List, Dict, Any, Optional, Literal
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp

# 检查是否为 aiocqhttp 平台
try: 
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent 
    IS_AIOCQHTTP = True 
except ImportError: 
    IS_AIOCQHTTP = False 

PermissionPolicy = Literal["admin_only", "admin_or_self"]
INJECTABLE_MEMBER_FIELDS: Dict[str, str] = {
    "user_id": "用户ID",
    "nickname": "昵称",
    "card": "群名片",
    "sex": "性别",
    "age": "年龄",
    "join_time": "入群时间",
    "last_sent_time": "最后发言时间",
    "level": "群等级",
    "qq_level": "QQ等级",
    "role": "群身份",
    "title": "群头衔",
    "area": "地区",
    "shut_up_timestamp": "禁言时间戳",
    "qage": "Q龄",
}
TOOL_OPTION_LABEL_TO_ID: Dict[str, str] = {
    "禁言成员": "set_group_ban",
    "群组踢人": "kick_group_member",
    "全员禁言": "set_group_whole_ban",
    "设置群管理员": "set_group_admin",
    "设置群名片": "set_group_card",
    "设置群头衔": "set_group_special_title",
    "设置精华消息": "set_essence_msg",
    "移出精华消息": "delete_essence_msg",
    "设置群名称": "set_group_name",
    "发送群公告": "send_group_notice",
    "删除群公告": "delete_group_notice",
    "解散群": "dismiss_group",
    "批量踢出群成员": "set_group_kick_members",
    "拉黑用户": "block_user",
    "解除拉黑": "unblock_user",
}
DANGEROUS_TOOL_IDS = {
    "set_group_ban",
    "kick_group_member",
    "set_group_whole_ban",
    "set_group_admin",
    "delete_essence_msg",
    "delete_group_notice",
    "dismiss_group",
    "set_group_kick_members",
}
CONFIRMATION_REQUIRED_TOOL_IDS = {
    "kick_group_member",
    "set_group_whole_ban",
    "set_group_admin",
    "delete_essence_msg",
    "delete_group_notice",
    "dismiss_group",
    "set_group_kick_members",
}
PENDING_CONFIRMATION_MAX = 64
CONFIRM_TIMEOUT_DEFAULT_SEC = 90
CONFIRM_TIMEOUT_MIN_SEC = 10
CONFIRM_TIMEOUT_MAX_SEC = 600
_PENDING_TOOL_CONFIRMATIONS: Dict[str, Dict[str, Any]] = {}
_TOOL_LABEL_LOWER_TO_ID = {k.lower(): v for k, v in TOOL_OPTION_LABEL_TO_ID.items()}
for _tool_id in TOOL_OPTION_LABEL_TO_ID.values():
    _TOOL_LABEL_LOWER_TO_ID[_tool_id.lower()] = _tool_id


def validate_write_permission(
    event: AstrMessageEvent,
    *,
    target_user_id: str,
    strict: bool,
    policy: PermissionPolicy,
    action: str,
) -> Optional[str]:
    if not strict:
        return None

    sender_id = str(event.get_sender_id() or "")
    is_admin = bool(event.is_admin())
    is_self_action = bool(target_user_id) and target_user_id == sender_id

    if policy == "admin_only":
        if not is_admin:
            return f"权限不足。只有管理员可以{action}。"
        return None

    if policy == "admin_or_self":
        if not is_admin and not is_self_action:
            return f"权限不足。您({sender_id})没有权限{action}其他用户({target_user_id})。"
        return None

    return f"权限策略错误: {policy}"


def normalize_tool_selection(raw_tools: Any) -> set[str]:
    """将配置中的工具选项规范化为内部 tool_id 集合。"""
    if not isinstance(raw_tools, (list, tuple, set)):
        return set()
    normalized: set[str] = set()
    for item in raw_tools:
        text = str(item or "").strip()
        if not text:
            continue
        tool_id = _TOOL_LABEL_LOWER_TO_ID.get(text.lower())
        if tool_id:
            normalized.add(tool_id)
    return normalized


def is_tool_admin_required(tool_id: str, admin_required_tools: Any) -> bool:
    return tool_id in normalize_tool_selection(admin_required_tools)


def is_dangerous_tool_enabled(tool_id: str, enabled_dangerous_tools: Any) -> bool:
    selected = normalize_tool_selection(enabled_dangerous_tools)
    return tool_id in selected


def is_tool_confirmation_required(tool_id: str, confirm_required_tools: Any) -> bool:
    if tool_id not in CONFIRMATION_REQUIRED_TOOL_IDS:
        return False
    return tool_id in normalize_tool_selection(confirm_required_tools)


def _coerce_confirm_timeout(confirm_timeout_sec: Any) -> int:
    try:
        value = int(confirm_timeout_sec)
    except Exception:
        value = CONFIRM_TIMEOUT_DEFAULT_SEC
    return max(CONFIRM_TIMEOUT_MIN_SEC, min(CONFIRM_TIMEOUT_MAX_SEC, value))


def _build_confirmation_key(sender_id: str, tool_id: str, group_scope: str) -> str:
    return f"{sender_id}:{group_scope}:{tool_id}"


def _build_confirmation_fingerprint(tool_id: str, payload: Dict[str, Any]) -> str:
    canonical = json.dumps(
        {"tool_id": tool_id, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cleanup_pending_confirmations(now_ts: Optional[float] = None) -> None:
    now = float(now_ts or time.time())
    expired_keys = [
        k for k, v in _PENDING_TOOL_CONFIRMATIONS.items()
        if float(v.get("expire_at", 0)) <= now
    ]
    for k in expired_keys:
        _PENDING_TOOL_CONFIRMATIONS.pop(k, None)

    if len(_PENDING_TOOL_CONFIRMATIONS) <= PENDING_CONFIRMATION_MAX:
        return

    ordered = sorted(
        _PENDING_TOOL_CONFIRMATIONS.items(),
        key=lambda item: float(item[1].get("created_at", 0)),
    )
    overflow = len(_PENDING_TOOL_CONFIRMATIONS) - PENDING_CONFIRMATION_MAX
    for i in range(max(0, overflow)):
        _PENDING_TOOL_CONFIRMATIONS.pop(ordered[i][0], None)


def _build_confirmation_response(
    *,
    reason: str,
    tool_id: str,
    action: str,
    confirm_token: str,
    expire_at: float,
    group_scope: str,
) -> str:
    tag = "CONFIRM_REFRESHED" if reason == "payload_changed" else "CONFIRM_REQUIRED"
    instruction = "call_same_tool_with_same_args_and_confirm_token"
    one_liner = (
        f"{tag} tool_id={tool_id} action={action} confirm_token={confirm_token} "
        f"expires_at={int(expire_at)} next={instruction}"
    )
    return json.dumps(
        {
            "success": False,
            "need_confirmation": True,
            "confirm_reason": reason,
            "tool_id": tool_id,
            "action": action,
            "confirm_token": confirm_token,
            "expires_at": int(expire_at),
            "group_scope": group_scope,
            "confirm_instruction": instruction,
            "message": one_liner,
        },
        ensure_ascii=False,
    )


def _check_write_tool_confirmation(
    event: AstrMessageEvent,
    *,
    tool_id: str,
    action: str,
    group_scope: str,
    fingerprint_payload: Dict[str, Any],
    confirm_required_tools: Any,
    confirm_timeout_sec: Any,
    confirm_token: str,
) -> Optional[str]:
    if not is_tool_confirmation_required(tool_id, confirm_required_tools):
        return None

    sender_id = str(event.get_sender_id() or "").strip()
    if not sender_id:
        return _json_error("无法识别操作者，已拒绝执行。", need_confirmation=False, tool_id=tool_id)

    scope = str(group_scope or "").strip() or "unknown"
    now_ts = time.time()
    timeout_sec = _coerce_confirm_timeout(confirm_timeout_sec)
    _cleanup_pending_confirmations(now_ts)

    key = _build_confirmation_key(sender_id, tool_id, scope)
    fingerprint = _build_confirmation_fingerprint(tool_id, fingerprint_payload)
    provided_token = str(confirm_token or "").strip()
    pending = _PENDING_TOOL_CONFIRMATIONS.get(key)

    if provided_token:
        if not pending:
            return _json_error(
                f"CONFIRM_TOKEN_MISSING tool_id={tool_id} next=call_same_tool_without_confirm_token",
                need_confirmation=True,
                tool_id=tool_id,
            )

        if float(pending.get("expire_at", 0)) <= now_ts:
            _PENDING_TOOL_CONFIRMATIONS.pop(key, None)
            return _json_error(
                f"CONFIRM_TOKEN_EXPIRED tool_id={tool_id} next=call_same_tool_without_confirm_token",
                need_confirmation=True,
                tool_id=tool_id,
            )

        if str(pending.get("fingerprint", "")) != fingerprint:
            new_token = secrets.token_urlsafe(12)
            new_expire_at = now_ts + timeout_sec
            _PENDING_TOOL_CONFIRMATIONS[key] = {
                "token": new_token,
                "fingerprint": fingerprint,
                "expire_at": new_expire_at,
                "created_at": now_ts,
            }
            _cleanup_pending_confirmations(now_ts)
            return _build_confirmation_response(
                reason="payload_changed",
                tool_id=tool_id,
                action=action,
                confirm_token=new_token,
                expire_at=new_expire_at,
                group_scope=scope,
            )

        if str(pending.get("token", "")) != provided_token:
            return _json_error(
                f"CONFIRM_TOKEN_INVALID tool_id={tool_id} next=use_latest_confirm_token_or_reissue",
                need_confirmation=True,
                tool_id=tool_id,
            )

        _PENDING_TOOL_CONFIRMATIONS.pop(key, None)
        return None

    if pending and float(pending.get("expire_at", 0)) > now_ts and str(pending.get("fingerprint", "")) == fingerprint:
        token = str(pending.get("token", ""))
        expire_at = float(pending.get("expire_at", now_ts + timeout_sec))
    else:
        token = secrets.token_urlsafe(12)
        expire_at = now_ts + timeout_sec
        _PENDING_TOOL_CONFIRMATIONS[key] = {
            "token": token,
            "fingerprint": fingerprint,
            "expire_at": expire_at,
            "created_at": now_ts,
        }
        _cleanup_pending_confirmations(now_ts)

    return _build_confirmation_response(
        reason="initial",
        tool_id=tool_id,
        action=action,
        confirm_token=token,
        expire_at=expire_at,
        group_scope=scope,
    )


def _build_tool_disabled_json(tool_id: str) -> str:
    display = next((k for k, v in TOOL_OPTION_LABEL_TO_ID.items() if v == tool_id), tool_id)
    return json.dumps(
        {
            "success": False,
            "message": f"这个操作我现在不能直接执行（{display}）。",
            "tool_id": tool_id,
        },
        ensure_ascii=False,
    )


def _check_write_tool_access(
    event: AstrMessageEvent,
    *,
    tool_id: str,
    action: str,
    admin_required_tools: Any,
    enabled_dangerous_tools: Any,
    policy: PermissionPolicy = "admin_only",
    target_user_id: str = "",
) -> Optional[str]:
    if tool_id in DANGEROUS_TOOL_IDS and not is_dangerous_tool_enabled(tool_id, enabled_dangerous_tools):
        return _build_tool_disabled_json(tool_id)

    permission_error = validate_write_permission(
        event,
        target_user_id=target_user_id or str(event.get_sender_id() or ""),
        strict=is_tool_admin_required(tool_id, admin_required_tools),
        policy=policy,
        action=action,
    )
    if permission_error:
        return json.dumps({"success": False, "message": permission_error}, ensure_ascii=False)
    return None


def _normalize_member_injection_fields(raw_fields: Any) -> List[str]:
    """将配置中的字段列表规范化为内部字段 key 列表。"""
    if not isinstance(raw_fields, list):
        return []

    normalized: List[str] = []
    alias_map = {
        "用户id": "user_id",
        "昵称": "nickname",
        "群名片": "card",
        "性别": "sex",
        "年龄": "age",
        "入群时间": "join_time",
        "最后发言时间": "last_sent_time",
        "群等级": "level",
        "qq等级": "qq_level",
        "群身份": "role",
        "角色": "role",
        "群头衔": "title",
        "头衔": "title",
        "地区": "area",
        "禁言时间戳": "shut_up_timestamp",
        "q龄": "qage",
    }

    for item in raw_fields:
        text = str(item or "").strip()
        if not text:
            continue
        if "(" in text and text.endswith(")"):
            text = text.split("(", 1)[0].strip()
        key = alias_map.get(text.lower(), text.lower())
        if key in INJECTABLE_MEMBER_FIELDS and key not in normalized:
            normalized.append(key)
    return normalized


async def inject_sender_group_member_info(
    event: AstrMessageEvent,
    req: Any,
    raw_fields: Any,
    no_cache: bool = False,
) -> bool:
    """
    按配置字段将当前发送者群成员信息注入到 ProviderRequest.prompt。
    返回是否完成注入。
    """
    return await _inject_group_member_info(
        event=event,
        req=req,
        raw_fields=raw_fields,
        target_user_id=str(event.get_sender_id() or "").strip(),
        subject_label="当前消息发送者",
        no_cache=no_cache,
    )


async def inject_bot_group_member_info(
    event: AstrMessageEvent,
    req: Any,
    raw_fields: Any,
    no_cache: bool = False,
) -> bool:
    """
    按配置字段将当前 Bot 在本群的成员信息注入到 ProviderRequest.prompt。
    返回是否完成注入。
    """
    return await _inject_group_member_info(
        event=event,
        req=req,
        raw_fields=raw_fields,
        target_user_id=str(event.get_self_id() or "").strip(),
        subject_label="当前Bot",
        no_cache=no_cache,
    )


async def _inject_group_member_info(
    event: AstrMessageEvent,
    req: Any,
    raw_fields: Any,
    target_user_id: str,
    subject_label: str,
    no_cache: bool = False,
) -> bool:
    """将指定 user_id 在当前群中的成员信息按字段注入到 ProviderRequest.prompt。"""
    if not req or not hasattr(req, "prompt"):
        return False
    if not (IS_AIOCQHTTP and isinstance(event, AiocqhttpMessageEvent)):
        return False

    target_group_id = str(event.get_group_id() or "").strip()
    if not target_group_id or not target_user_id:
        return False

    selected_fields = _normalize_member_injection_fields(raw_fields)
    if not selected_fields:
        return False

    member_info = await get_group_member_info_internal(
        event,
        group_id=target_group_id,
        user_id=target_user_id,
        no_cache=no_cache,
    )
    if not member_info:
        logger.debug(
            "[LLMEnhancement] 群成员信息注入跳过："
            f"subject={subject_label}, "
            f"group={target_group_id}, uid={target_user_id}, reason=not_found"
        )
        return False

    payload: Dict[str, Any] = {}
    for field in selected_fields:
        if field in member_info:
            value = member_info[field]
            if field in {"user_id"}:
                value = str(value)
            payload[field] = value

    if not payload:
        return False

    readable_fields = [INJECTABLE_MEMBER_FIELDS.get(f, f) for f in payload.keys()]
    context_prompt = (
        f"\n\n以下是{subject_label}群成员信息，请结合这些信息理解用户意图并回答：\n"
        f"(注入字段: {', '.join(readable_fields)})\n"
        "--- 注入内容开始 ---\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "--- 注入内容结束 ---"
    )
    user_question = str(getattr(req, "prompt", "") or "").strip()
    req.prompt = (user_question + context_prompt) if user_question else context_prompt.strip()
    logger.debug(
        "[LLMEnhancement] 群成员信息注入完成："
        f"subject={subject_label}, "
        f"group={target_group_id}, uid={target_user_id}, no_cache={bool(no_cache)}, "
        f"fields={list(payload.keys())}, payload={json.dumps(payload, ensure_ascii=False)}"
    )
    return True

async def get_group_members_internal(event: AstrMessageEvent, group_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]: 
    """ 调用API获取群成员列表 """ 
    try: 
        target_group_id = group_id or event.get_group_id() 
        if not target_group_id: 
            return None 

        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return None

        client = event.bot 
        params = {"group_id": int(target_group_id)}
        try:
            raw_result = await client.api.call_action("get_group_member_list", **params)
        except Exception:
            if hasattr(client, "get_group_member_list"):
                raw_result = await client.get_group_member_list(**params)
            else:
                raise
        data = _unwrap_action_data(raw_result)
        if isinstance(data, list):
            return data
        if isinstance(raw_result, list):
            return raw_result
        return None
    except Exception as e: 
        logger.info(f"API调用失败: {e}") 
        return None


async def get_group_member_info_internal(
    event: AstrMessageEvent,
    group_id: Optional[str] = None,
    user_id: Optional[str] = None,
    no_cache: bool = False,
) -> Optional[Dict[str, Any]]:
    """调用 API 获取单个群成员详情。"""
    try:
        target_group_id = group_id or event.get_group_id()
        target_user_id = user_id or event.get_sender_id()
        if not target_group_id or not target_user_id:
            return None

        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return None

        client = event.bot
        params = {
            "group_id": int(target_group_id),
            "user_id": int(target_user_id),
            "no_cache": bool(no_cache),
        }
        raw_result = await client.api.call_action("get_group_member_info", **params)
        if isinstance(raw_result, dict):
            if isinstance(raw_result.get("data"), dict):
                return raw_result["data"]
            if "user_id" in raw_result or "group_id" in raw_result:
                return raw_result
            return None
        return None
    except Exception as e:
        logger.info(f"API调用失败: {e}")
        return None

async def process_group_members_info(event: AstrMessageEvent, group_id: Optional[str] = None) -> str:
    """
    获取并处理QQ群成员信息的逻辑。
    """
    start_time = time.time() 
    
    try: 
        target_group_id = group_id or event.get_group_id() 
        if not target_group_id: 
            logger.info("用户在非群聊环境中调用群成员查询工具且未提供群号") 
            return json.dumps({"error": "未识别到群聊环境，请提供目标群号。"}) 
        
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent): 
            logger.info(f"不支持的平台: {event.get_platform_name()}") 
            return json.dumps({"error": f"此功能仅支持QQ群聊(aiocqhttp平台)，当前平台为 {event.get_platform_name()}"}) 

        # 从API获取 
        members_info = await get_group_members_internal(event, group_id=target_group_id) 
        if not members_info: 
            logger.info(f"无法获取群 {target_group_id} 的成员信息") 
            return json.dumps({"error": f"无法获取群 {target_group_id} 的成员信息，请确认你是否在该群内。"}) 
        
        processed_members = [ 
            { 
                "user_id": str(member.get("user_id", "")), 
                "nickname": member.get("nickname") or f"用户{member.get('user_id')}",
                "card": member.get("card") or "",
                "role": member.get("role", "member") 
            } 
            for member in members_info if member.get("user_id") 
        ] 
        
        group_info = { 
            "group_id": str(target_group_id), 
            "member_count": len(processed_members), 
            "members": processed_members 
        } 
        
        elapsed_time = time.time() - start_time 
        logger.info(f"成功获取群 {target_group_id} 的 {len(processed_members)} 名成员信息，耗时 {elapsed_time:.2f}s") 
        
        return json.dumps(group_info, ensure_ascii=False, indent=2) 
    except Exception as e: 
        elapsed_time = time.time() - start_time 
        logger.info(f"获取群成员信息时发生错误: {e}，耗时 {elapsed_time:.2f}s") 
        return json.dumps({"error": f"获取群成员信息时发生内部错误: {str(e)}"})


async def process_group_member_info(
    event: AstrMessageEvent,
    user_id: Optional[str] = None,
    group_id: Optional[str] = None,
    no_cache: bool = False,
) -> str:
    """获取并处理单个QQ群成员信息。"""
    start_time = time.time()

    try:
        target_group_id = group_id or event.get_group_id()
        target_user_id = user_id or event.get_sender_id()
        if not target_group_id:
            logger.info("用户在非群聊环境中调用群成员详情工具且未提供群号")
            return json.dumps({"error": "未识别到群聊环境，请提供目标群号。"}, ensure_ascii=False)
        if not target_user_id:
            return json.dumps({"error": "请提供目标用户ID(user_id)。"}, ensure_ascii=False)

        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            logger.info(f"不支持的平台: {event.get_platform_name()}")
            return json.dumps(
                {"error": f"此功能仅支持QQ群聊(aiocqhttp平台)，当前平台为 {event.get_platform_name()}"},
                ensure_ascii=False,
            )

        member_info = await get_group_member_info_internal(
            event,
            group_id=target_group_id,
            user_id=target_user_id,
            no_cache=no_cache,
        )
        if not member_info:
            logger.info(f"无法获取群 {target_group_id} 用户 {target_user_id} 的成员信息")
            return json.dumps(
                {"error": f"无法获取群 {target_group_id} 用户 {target_user_id} 的成员信息。"},
                ensure_ascii=False,
            )

        if "group_id" in member_info:
            member_info["group_id"] = str(member_info.get("group_id"))
        if "user_id" in member_info:
            member_info["user_id"] = str(member_info.get("user_id"))

        result = {
            "group_id": str(target_group_id),
            "user_id": str(target_user_id),
            "no_cache": bool(no_cache),
            "member": member_info,
        }
        elapsed_time = time.time() - start_time
        logger.info(f"成功获取群 {target_group_id} 用户 {target_user_id} 的成员详情，耗时 {elapsed_time:.2f}s")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.info(f"获取群成员详情时发生错误: {e}，耗时 {elapsed_time:.2f}s")
        return json.dumps({"error": f"获取群成员详情时发生内部错误: {str(e)}"}, ensure_ascii=False)


def _unwrap_action_data(raw_result: Any) -> Any:
    """兼容 call_action 可能返回的包裹层。"""
    if isinstance(raw_result, dict):
        if "data" in raw_result:
            return raw_result.get("data")
    return raw_result


def _json_error(message: str, **extra: Any) -> str:
    payload = {"success": False, "message": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _json_success(message: str, **extra: Any) -> str:
    payload = {"success": True, "message": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ensure_group_write_context(
    event: AstrMessageEvent,
    group_id: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
        return None, _json_error(f"此功能仅支持 QQ 平台 (aiocqhttp)，当前平台为 {event.get_platform_name()}")
    target_group_id = str(group_id or event.get_group_id() or "").strip()
    if not target_group_id:
        return None, _json_error("未识别到群聊环境，请提供目标群号(group_id)。")
    return target_group_id, None


def _extract_reply_message_id(event: AstrMessageEvent) -> Optional[str]:
    """从消息引用(reply)中提取被引用消息 ID。"""
    try:
        message_obj = getattr(event, "message_obj", None)
        segments = getattr(message_obj, "message", None) if message_obj is not None else None
        if isinstance(segments, list):
            for seg in segments:
                if isinstance(seg, Comp.Reply):
                    msg_id = str(getattr(seg, "id", "") or getattr(seg, "message_id", "") or "").strip()
                    if msg_id:
                        return msg_id
                if isinstance(seg, dict) and str(seg.get("type", "")).lower() == "reply":
                    data = seg.get("data", {}) or {}
                    msg_id = str(data.get("id", "") or data.get("message_id", "") or "").strip()
                    if msg_id:
                        return msg_id
    except Exception:
        pass

    try:
        message_obj = getattr(event, "message_obj", None)
        raw_message = getattr(message_obj, "raw_message", None) if message_obj is not None else None
        raw_segments = raw_message.get("message") if isinstance(raw_message, dict) else None
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                if isinstance(seg, dict) and str(seg.get("type", "")).lower() == "reply":
                    data = seg.get("data", {}) or {}
                    msg_id = str(data.get("id", "") or data.get("message_id", "") or "").strip()
                    if msg_id:
                        return msg_id
    except Exception:
        pass

    text = str(getattr(event, "message_str", "") or "")
    match = re.search(r"\[CQ:reply,id=([0-9]+)\]", text, flags=re.IGNORECASE)
    if match:
        return str(match.group(1)).strip()
    return None


async def _call_action(
    event: AstrMessageEvent,
    action: str,
    *,
    fallback_method: Optional[str] = None,
    **params: Any,
) -> Any:
    client = event.bot
    try:
        return await client.api.call_action(action, **params)
    except Exception:
        if fallback_method and hasattr(client, fallback_method):
            return await getattr(client, fallback_method)(**params)
        raise


def _parse_user_id_list(user_ids: Any) -> List[str]:
    if isinstance(user_ids, (list, tuple, set)):
        raw_items = [str(x or "").strip() for x in user_ids]
    else:
        text = str(user_ids or "").replace("，", ",")
        raw_items = [part.strip() for part in text.split(",")]
    return [uid for uid in raw_items if uid]


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "开启", "启用", "是"}:
        return True
    if text in {"0", "false", "no", "n", "off", "关闭", "禁用", "否"}:
        return False
    return default


async def get_group_info_internal(
    event: AstrMessageEvent,
    group_id: Optional[str] = None,
    no_cache: bool = False,
) -> Optional[Dict[str, Any]]:
    """调用 API 获取群信息。"""
    try:
        target_group_id = group_id or event.get_group_id()
        if not target_group_id:
            return None
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return None

        client = event.bot
        params = {
            "group_id": int(target_group_id),
            "no_cache": bool(no_cache),
        }
        try:
            raw_result = await client.api.call_action("get_group_info", **params)
        except Exception:
            if hasattr(client, "get_group_info"):
                raw_result = await client.get_group_info(**params)
            else:
                raise
        data = _unwrap_action_data(raw_result)
        if isinstance(data, dict):
            return data
        return None
    except Exception as e:
        logger.info(f"获取群信息 API 调用失败: {e}")
        return None


async def process_group_info(
    event: AstrMessageEvent,
    group_id: Optional[str] = None,
    no_cache: bool = False,
) -> str:
    """获取并处理群信息。"""
    start_time = time.time()
    try:
        target_group_id = group_id or event.get_group_id()
        if not target_group_id:
            return json.dumps({"error": "未识别到群聊环境，请提供目标群号。"}, ensure_ascii=False)
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return json.dumps(
                {"error": f"此功能仅支持QQ群聊(aiocqhttp平台)，当前平台为 {event.get_platform_name()}"},
                ensure_ascii=False,
            )

        group_info = await get_group_info_internal(event, group_id=target_group_id, no_cache=no_cache)
        if not group_info:
            return json.dumps({"error": f"无法获取群 {target_group_id} 的群信息。"}, ensure_ascii=False)

        if "group_id" in group_info:
            group_info["group_id"] = str(group_info.get("group_id"))

        result = {
            "group_id": str(target_group_id),
            "no_cache": bool(no_cache),
            "group_info": group_info,
        }
        elapsed_time = time.time() - start_time
        logger.info(f"成功获取群 {target_group_id} 的群信息，耗时 {elapsed_time:.2f}s")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.info(f"获取群信息时发生错误: {e}，耗时 {elapsed_time:.2f}s")
        return json.dumps({"error": f"获取群信息时发生内部错误: {str(e)}"}, ensure_ascii=False)


async def get_group_notices_internal(event: AstrMessageEvent, group_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """调用 API 获取群公告列表。"""
    try:
        target_group_id = group_id or event.get_group_id()
        if not target_group_id:
            return None
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return None

        client = event.bot
        try:
            raw_result = await client.api.call_action("_get_group_notice", group_id=int(target_group_id))
        except Exception:
            if hasattr(client, "_get_group_notice"):
                raw_result = await client._get_group_notice(group_id=int(target_group_id))
            else:
                raise
        data = _unwrap_action_data(raw_result)
        if isinstance(data, list):
            return data
        if isinstance(raw_result, list):
            return raw_result
        return None
    except Exception as e:
        logger.info(f"获取群公告 API 调用失败: {e}")
        return None


async def process_group_notices(
    event: AstrMessageEvent,
    group_id: Optional[str] = None,
    limit: int = 10,
) -> str:
    """获取并处理群公告。"""
    start_time = time.time()
    try:
        target_group_id = group_id or event.get_group_id()
        if not target_group_id:
            return json.dumps({"error": "未识别到群聊环境，请提供目标群号。"}, ensure_ascii=False)
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return json.dumps(
                {"error": f"此功能仅支持QQ群聊(aiocqhttp平台)，当前平台为 {event.get_platform_name()}"},
                ensure_ascii=False,
            )

        notices = await get_group_notices_internal(event, group_id=target_group_id)
        if notices is None:
            return json.dumps({"error": f"无法获取群 {target_group_id} 的公告信息。"}, ensure_ascii=False)

        safe_limit = max(1, min(int(limit or 10), 50))
        processed_notices: List[Dict[str, Any]] = []
        for n in notices[:safe_limit]:
            msg = n.get("message", {}) if isinstance(n, dict) else {}
            settings = n.get("settings", {}) if isinstance(n, dict) else {}
            item = {
                "notice_id": n.get("notice_id"),
                "sender_id": n.get("sender_id"),
                "publish_time": n.get("publish_time"),
                "text": (msg.get("text", "") or "").replace("&#10;", "\n").replace("&nbsp;", " "),
                "read_num": n.get("read_num"),
                "settings": {
                    "is_show_edit_card": settings.get("is_show_edit_card"),
                    "tip_window_type": settings.get("tip_window_type"),
                    "confirm_required": settings.get("confirm_required"),
                },
            }
            images = msg.get("image") or msg.get("images")
            if images:
                if not isinstance(images, list):
                    images = [images]
                image_urls: List[str] = []
                for img in images:
                    if not isinstance(img, dict):
                        continue
                    img_url = img.get("url")
                    img_id = img.get("id")
                    if not img_url and img_id:
                        img_url = f"https://gdynamic.qpic.cn/gdynamic/{img_id}/628"
                    if img_url:
                        image_urls.append(img_url)
                if image_urls:
                    item["images"] = image_urls
            processed_notices.append(item)

        result = {
            "group_id": str(target_group_id),
            "notice_count": len(notices),
            "returned_count": len(processed_notices),
            "notices": processed_notices,
        }
        elapsed_time = time.time() - start_time
        logger.info(f"成功获取群 {target_group_id} 的公告信息，耗时 {elapsed_time:.2f}s")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.info(f"获取群公告时发生错误: {e}，耗时 {elapsed_time:.2f}s")
        return json.dumps({"error": f"获取群公告时发生内部错误: {str(e)}"}, ensure_ascii=False)


async def get_group_essence_internal(event: AstrMessageEvent, group_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """调用 API 获取群精华列表。"""
    try:
        target_group_id = group_id or event.get_group_id()
        if not target_group_id:
            return None
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return None

        client = event.bot
        try:
            raw_result = await client.api.call_action("get_essence_msg_list", group_id=int(target_group_id))
        except Exception:
            if hasattr(client, "get_essence_msg_list"):
                raw_result = await client.get_essence_msg_list(group_id=int(target_group_id))
            else:
                raise
        data = _unwrap_action_data(raw_result)
        if isinstance(data, list):
            return data
        if isinstance(raw_result, list):
            return raw_result
        return None
    except Exception as e:
        logger.info(f"获取群精华 API 调用失败: {e}")
        return None


async def process_group_essence(
    event: AstrMessageEvent,
    group_id: Optional[str] = None,
    limit: int = 10,
) -> str:
    """获取并处理群精华。"""
    start_time = time.time()
    try:
        target_group_id = group_id or event.get_group_id()
        if not target_group_id:
            return json.dumps({"error": "未识别到群聊环境，请提供目标群号。"}, ensure_ascii=False)
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return json.dumps(
                {"error": f"此功能仅支持QQ群聊(aiocqhttp平台)，当前平台为 {event.get_platform_name()}"},
                ensure_ascii=False,
            )

        essence_list = await get_group_essence_internal(event, group_id=target_group_id)
        if essence_list is None:
            return json.dumps({"error": f"无法获取群 {target_group_id} 的精华信息。"}, ensure_ascii=False)

        safe_limit = max(1, min(int(limit or 10), 50))
        processed_list: List[Dict[str, Any]] = []
        for e in essence_list[:safe_limit]:
            if not isinstance(e, dict):
                continue
            processed_list.append(
                {
                    "message_id": e.get("message_id"),
                    "sender_id": e.get("sender_id"),
                    "sender_nick": e.get("sender_nick"),
                    "operator_id": e.get("operator_id"),
                    "operator_nick": e.get("operator_nick"),
                    "operator_time": e.get("operator_time"),
                    "content": e.get("content"),
                }
            )

        result = {
            "group_id": str(target_group_id),
            "essence_count": len(essence_list),
            "returned_count": len(processed_list),
            "essence": processed_list,
        }
        elapsed_time = time.time() - start_time
        logger.info(f"成功获取群 {target_group_id} 的精华信息，耗时 {elapsed_time:.2f}s")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.info(f"获取群精华时发生错误: {e}，耗时 {elapsed_time:.2f}s")
        return json.dumps({"error": f"获取群精华时发生内部错误: {str(e)}"}, ensure_ascii=False)

async def set_group_ban_logic(
    event: AstrMessageEvent,
    user_id: str,
    duration: int,
    user_name: str,
    group_id: str = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
) -> str:
    """
    在群聊中禁言某用户的逻辑。
    """
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error

        target_user_id = str(user_id or "").strip()
        if not target_user_id:
            return _json_error("请提供要禁言/解除禁言的目标用户 ID。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_group_ban",
            action="禁言",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
            policy="admin_or_self",
            target_user_id=target_user_id,
        )
        if disabled_resp:
            return disabled_resp

        sender_id = str(event.get_sender_id())
        strict_permission_check = is_tool_admin_required("set_group_ban", admin_required_tools)
        permission_error = validate_write_permission(
            event,
            target_user_id=target_user_id,
            strict=strict_permission_check,
            policy="admin_or_self",
            action="禁言",
        )
        if permission_error:
            logger.warning(f"用户 {sender_id} 尝试禁言 {target_user_id}，权限不足（严格校验开启）。")
            return _json_error(permission_error)

        # 3. 执行禁言
        params = {
            "group_id": int(target_group_id),
            "user_id": int(target_user_id),
            "duration": duration
        }

        await _call_action(event, "set_group_ban", **params)
        
        logger.info(f"调用方 {sender_id} 通过工具禁言了用户 {target_user_id} ({user_name})，时长 {duration} 秒。")

        return _json_success(
            f"用户 {user_name} ({target_user_id}) 已被禁言 {duration} 秒。",
            user_id=target_user_id,
            user_name=user_name,
            duration=duration,
            timestamp=int(time.time()),
        )

    except Exception as e:
        logger.error(f"禁言用户 {user_id} 失败: {e}")
        return _json_error(f"操作失败：无法禁言用户 {user_name}", error=str(e))


async def kick_group_member_logic(
    event: AstrMessageEvent,
    user_id: str,
    group_id: Optional[str] = None,
    reject_add_request: bool = False,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
    confirm_required_tools: Any = None,
    confirm_timeout_sec: int = CONFIRM_TIMEOUT_DEFAULT_SEC,
    confirm_token: str = "",
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error
        target_user_id = str(user_id or "").strip()
        if not target_user_id:
            return _json_error("请提供要踢出的用户 ID。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="kick_group_member",
            action="踢人",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        confirm_resp = _check_write_tool_confirmation(
            event,
            tool_id="kick_group_member",
            action="群组踢人",
            group_scope=str(target_group_id),
            fingerprint_payload={
                "group_id": str(target_group_id),
                "user_id": target_user_id,
                "reject_add_request": _to_bool(reject_add_request, False),
            },
            confirm_required_tools=confirm_required_tools,
            confirm_timeout_sec=confirm_timeout_sec,
            confirm_token=confirm_token,
        )
        if confirm_resp:
            return confirm_resp

        await _call_action(
            event,
            "set_group_kick",
            group_id=int(target_group_id),
            user_id=int(target_user_id),
            reject_add_request=_to_bool(reject_add_request, False),
        )
        return _json_success(
            f"已将用户 {target_user_id} 踢出群 {target_group_id}。",
            group_id=str(target_group_id),
            user_id=target_user_id,
            reject_add_request=_to_bool(reject_add_request, False),
        )
    except Exception as e:
        return _json_error("踢人失败。", error=str(e))


async def set_group_whole_ban_logic(
    event: AstrMessageEvent,
    enable: bool,
    group_id: Optional[str] = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
    confirm_required_tools: Any = None,
    confirm_timeout_sec: int = CONFIRM_TIMEOUT_DEFAULT_SEC,
    confirm_token: str = "",
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_group_whole_ban",
            action="全员禁言",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        enabled = _to_bool(enable, False)
        confirm_resp = _check_write_tool_confirmation(
            event,
            tool_id="set_group_whole_ban",
            action="全员禁言",
            group_scope=str(target_group_id),
            fingerprint_payload={
                "group_id": str(target_group_id),
                "enable": enabled,
            },
            confirm_required_tools=confirm_required_tools,
            confirm_timeout_sec=confirm_timeout_sec,
            confirm_token=confirm_token,
        )
        if confirm_resp:
            return confirm_resp

        await _call_action(
            event,
            "set_group_whole_ban",
            group_id=int(target_group_id),
            enable=enabled,
        )
        action_text = "开启" if enabled else "关闭"
        return _json_success(
            f"已{action_text}群 {target_group_id} 的全员禁言。",
            group_id=str(target_group_id),
            enable=enabled,
        )
    except Exception as e:
        return _json_error("设置全员禁言失败。", error=str(e))


async def set_group_admin_logic(
    event: AstrMessageEvent,
    user_id: str,
    enable: bool = True,
    group_id: Optional[str] = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
    confirm_required_tools: Any = None,
    confirm_timeout_sec: int = CONFIRM_TIMEOUT_DEFAULT_SEC,
    confirm_token: str = "",
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error
        target_user_id = str(user_id or "").strip()
        if not target_user_id:
            return _json_error("请提供目标用户 ID。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_group_admin",
            action="设置群管理员",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        enabled = _to_bool(enable, True)
        confirm_resp = _check_write_tool_confirmation(
            event,
            tool_id="set_group_admin",
            action="设置群管理员",
            group_scope=str(target_group_id),
            fingerprint_payload={
                "group_id": str(target_group_id),
                "user_id": target_user_id,
                "enable": enabled,
            },
            confirm_required_tools=confirm_required_tools,
            confirm_timeout_sec=confirm_timeout_sec,
            confirm_token=confirm_token,
        )
        if confirm_resp:
            return confirm_resp

        await _call_action(
            event,
            "set_group_admin",
            group_id=int(target_group_id),
            user_id=int(target_user_id),
            enable=enabled,
        )
        action_text = "设为管理员" if enabled else "取消管理员"
        return _json_success(
            f"已将用户 {target_user_id} {action_text}。",
            group_id=str(target_group_id),
            user_id=target_user_id,
            enable=enabled,
        )
    except Exception as e:
        return _json_error("设置群管理员失败。", error=str(e))


async def set_group_card_logic(
    event: AstrMessageEvent,
    user_id: str,
    card: str,
    group_id: Optional[str] = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error
        target_user_id = str(user_id or "").strip()
        if not target_user_id:
            return _json_error("请提供目标用户 ID。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_group_card",
            action="设置群名片",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        card_text = str(card or "")
        await _call_action(
            event,
            "set_group_card",
            group_id=int(target_group_id),
            user_id=int(target_user_id),
            card=card_text,
        )
        return _json_success(
            f"已设置用户 {target_user_id} 的群名片。",
            group_id=str(target_group_id),
            user_id=target_user_id,
            card=card_text,
        )
    except Exception as e:
        return _json_error("设置群名片失败。", error=str(e))


async def set_group_special_title_logic(
    event: AstrMessageEvent,
    user_id: str,
    special_title: str,
    group_id: Optional[str] = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error
        target_user_id = str(user_id or "").strip()
        if not target_user_id:
            return _json_error("请提供目标用户 ID。")
        title_text = str(special_title or "").strip()
        if not title_text:
            return _json_error("请提供头衔文本(special_title)。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_group_special_title",
            action="设置群头衔",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        await _call_action(
            event,
            "set_group_special_title",
            group_id=int(target_group_id),
            user_id=int(target_user_id),
            special_title=title_text,
        )
        return _json_success(
            f"已设置用户 {target_user_id} 的群头衔。",
            group_id=str(target_group_id),
            user_id=target_user_id,
            special_title=title_text,
        )
    except Exception as e:
        return _json_error("设置群头衔失败。", error=str(e))


async def set_essence_msg_logic(
    event: AstrMessageEvent,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
) -> str:
    try:
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return _json_error(f"此功能仅支持 QQ 平台 (aiocqhttp)，当前平台为 {event.get_platform_name()}")
        target_group_id = str(event.get_group_id() or "").strip()
        if not target_group_id:
            return _json_error("此工具仅支持在群聊中使用。")
        target_message_id = str(_extract_reply_message_id(event) or "").strip()
        if not target_message_id:
            return _json_error("无法确定目标消息，请先引用一条消息再调用本工具。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_essence_msg",
            action="设置精华消息",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        await _call_action(event, "set_essence_msg", message_id=int(target_message_id))
        return _json_success("已设置群精华消息。", message_id=target_message_id)
    except Exception as e:
        return _json_error("设置精华消息失败。", error=str(e))


async def delete_essence_msg_logic(
    event: AstrMessageEvent,
    message_id: str = "",
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
    confirm_required_tools: Any = None,
    confirm_timeout_sec: int = CONFIRM_TIMEOUT_DEFAULT_SEC,
    confirm_token: str = "",
) -> str:
    try:
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return _json_error(f"此功能仅支持 QQ 平台 (aiocqhttp)，当前平台为 {event.get_platform_name()}")
        target_group_id = str(event.get_group_id() or "").strip()
        if not target_group_id:
            return _json_error("此工具仅支持在群聊中使用。")
        target_message_id = str(message_id or "").strip()
        if not target_message_id:
            target_message_id = str(_extract_reply_message_id(event) or "").strip()
        if not target_message_id:
            return _json_error("无法确定目标消息，请传入 message_id 或先引用一条消息再调用本工具。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="delete_essence_msg",
            action="移出精华消息",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        confirm_resp = _check_write_tool_confirmation(
            event,
            tool_id="delete_essence_msg",
            action="移出精华消息",
            group_scope=target_group_id,
            fingerprint_payload={
                "group_id": target_group_id,
                "message_id": target_message_id,
            },
            confirm_required_tools=confirm_required_tools,
            confirm_timeout_sec=confirm_timeout_sec,
            confirm_token=confirm_token,
        )
        if confirm_resp:
            return confirm_resp

        await _call_action(
            event,
            "delete_essence_msg",
            message_id=int(target_message_id),
            group_id=int(target_group_id),
        )
        return _json_success("已移出群精华消息。", message_id=target_message_id, group_id=target_group_id)
    except Exception as e:
        return _json_error("移出精华消息失败。", error=str(e))


async def set_group_name_logic(
    event: AstrMessageEvent,
    group_name: str,
    group_id: Optional[str] = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error
        target_group_name = str(group_name or "").strip()
        if not target_group_name:
            return _json_error("请提供群名称(group_name)。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_group_name",
            action="设置群名称",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        await _call_action(
            event,
            "set_group_name",
            group_id=int(target_group_id),
            group_name=target_group_name,
        )
        return _json_success(
            f"已将群 {target_group_id} 名称修改为 {target_group_name}。",
            group_id=str(target_group_id),
            group_name=target_group_name,
        )
    except Exception as e:
        return _json_error("设置群名称失败。", error=str(e))


async def send_group_notice_logic(
    event: AstrMessageEvent,
    content: str,
    group_id: Optional[str] = None,
    pinned: bool = False,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error
        content_text = str(content or "").strip()
        if not content_text:
            return _json_error("请提供公告内容(content)。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="send_group_notice",
            action="发送群公告",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        params: Dict[str, Any] = {
            "group_id": int(target_group_id),
            "content": content_text,
            "pinned": _to_bool(pinned, False),
        }
        await _call_action(
            event,
            "_send_group_notice",
            fallback_method="_send_group_notice",
            **params,
        )
        return _json_success(
            "已发送群公告。",
            group_id=str(target_group_id),
            content=content_text,
            pinned=_to_bool(pinned, False),
        )
    except Exception as e:
        return _json_error("发送群公告失败。", error=str(e))


async def delete_group_notice_logic(
    event: AstrMessageEvent,
    notice_id: str,
    group_id: Optional[str] = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
    confirm_required_tools: Any = None,
    confirm_timeout_sec: int = CONFIRM_TIMEOUT_DEFAULT_SEC,
    confirm_token: str = "",
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error
        target_notice_id = str(notice_id or "").strip()
        if not target_notice_id:
            return _json_error("请提供公告 ID(notice_id)。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="delete_group_notice",
            action="删除群公告",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        confirm_resp = _check_write_tool_confirmation(
            event,
            tool_id="delete_group_notice",
            action="删除群公告",
            group_scope=str(target_group_id),
            fingerprint_payload={
                "group_id": str(target_group_id),
                "notice_id": target_notice_id,
            },
            confirm_required_tools=confirm_required_tools,
            confirm_timeout_sec=confirm_timeout_sec,
            confirm_token=confirm_token,
        )
        if confirm_resp:
            return confirm_resp

        await _call_action(
            event,
            "_del_group_notice",
            fallback_method="_del_group_notice",
            group_id=int(target_group_id),
            notice_id=target_notice_id,
        )
        return _json_success(
            "已删除群公告。",
            group_id=str(target_group_id),
            notice_id=target_notice_id,
        )
    except Exception as e:
        return _json_error("删除群公告失败。", error=str(e))


async def dismiss_group_logic(
    event: AstrMessageEvent,
    group_id: Optional[str] = None,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
    confirm_required_tools: Any = None,
    confirm_timeout_sec: int = CONFIRM_TIMEOUT_DEFAULT_SEC,
    confirm_token: str = "",
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="dismiss_group",
            action="解散群",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        confirm_resp = _check_write_tool_confirmation(
            event,
            tool_id="dismiss_group",
            action="解散群",
            group_scope=str(target_group_id),
            fingerprint_payload={"group_id": str(target_group_id), "is_dismiss": True},
            confirm_required_tools=confirm_required_tools,
            confirm_timeout_sec=confirm_timeout_sec,
            confirm_token=confirm_token,
        )
        if confirm_resp:
            return confirm_resp

        await _call_action(
            event,
            "set_group_leave",
            group_id=int(target_group_id),
            is_dismiss=True,
        )
        return _json_success("已发起解散群操作。", group_id=str(target_group_id), is_dismiss=True)
    except Exception as e:
        return _json_error("解散群失败。", error=str(e))


async def set_group_kick_members_logic(
    event: AstrMessageEvent,
    user_ids: Any,
    group_id: Optional[str] = None,
    reject_add_request: bool = False,
    admin_required_tools: Any = None,
    enabled_dangerous_tools: Any = None,
    confirm_required_tools: Any = None,
    confirm_timeout_sec: int = CONFIRM_TIMEOUT_DEFAULT_SEC,
    confirm_token: str = "",
) -> str:
    try:
        target_group_id, env_error = _ensure_group_write_context(event, group_id=group_id)
        if env_error:
            return env_error

        users = _parse_user_id_list(user_ids)
        if not users:
            return _json_error("请提供用户 ID 列表(user_ids)，支持逗号分隔字符串或数组。")

        disabled_resp = _check_write_tool_access(
            event,
            tool_id="set_group_kick_members",
            action="批量踢出群成员",
            admin_required_tools=admin_required_tools,
            enabled_dangerous_tools=enabled_dangerous_tools,
        )
        if disabled_resp:
            return disabled_resp

        confirm_resp = _check_write_tool_confirmation(
            event,
            tool_id="set_group_kick_members",
            action="批量踢出群成员",
            group_scope=str(target_group_id),
            fingerprint_payload={
                "group_id": str(target_group_id),
                "user_ids": users,
                "reject_add_request": _to_bool(reject_add_request, False),
            },
            confirm_required_tools=confirm_required_tools,
            confirm_timeout_sec=confirm_timeout_sec,
            confirm_token=confirm_token,
        )
        if confirm_resp:
            return confirm_resp

        await _call_action(
            event,
            "set_group_kick_members",
            group_id=str(target_group_id),
            user_id=users,
            reject_add_request=_to_bool(reject_add_request, False),
        )
        return _json_success(
            "批量踢出群成员操作已执行。",
            group_id=str(target_group_id),
            user_ids=users,
            reject_add_request=_to_bool(reject_add_request, False),
            count=len(users),
        )
    except Exception as e:
        return _json_error("批量踢人失败。", error=str(e))
