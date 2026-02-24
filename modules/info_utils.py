import json
import time
from typing import List, Dict, Any, Optional, Literal
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

# 检查是否为 aiocqhttp 平台
try: 
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent 
    IS_AIOCQHTTP = True 
except ImportError: 
    IS_AIOCQHTTP = False 

PermissionPolicy = Literal["admin_only", "admin_or_self"]


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
        return await client.api.call_action('get_group_member_list', **params) 
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

async def set_group_ban_logic(
    event: AstrMessageEvent,
    user_id: str,
    duration: int,
    user_name: str,
    group_id: str = None,
    strict_permission_check: bool = False,
) -> str:
    """
    在群聊中禁言某用户的逻辑。
    """
    try:
        # 1. 平台和环境检查
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return json.dumps({
                "success": False,
                "message": "此功能仅支持 QQ 平台 (aiocqhttp)。"
            }, ensure_ascii=False)

        target_group_id = group_id or event.get_group_id()
        if not target_group_id:
            return json.dumps({
                "success": False,
                "message": "未识别到群聊环境，请提供目标群号(group_id)。"
            }, ensure_ascii=False)

        target_user_id = str(user_id or "").strip()
        if not target_user_id:
            return json.dumps(
                {"success": False, "message": "请提供要禁言/解除禁言的目标用户 ID。"},
                ensure_ascii=False,
            )

        sender_id = str(event.get_sender_id())
        permission_error = validate_write_permission(
            event,
            target_user_id=target_user_id,
            strict=strict_permission_check,
            policy="admin_or_self",
            action="禁言",
        )
        if permission_error:
            logger.warning(f"用户 {sender_id} 尝试禁言 {target_user_id}，权限不足（严格校验开启）。")
            return json.dumps(
                {
                    "success": False,
                    "message": permission_error,
                },
                ensure_ascii=False,
            )

        # 3. 执行禁言
        client = event.bot
        params = {
            "group_id": int(target_group_id),
            "user_id": int(target_user_id),
            "duration": duration
        }
        
        await client.api.call_action('set_group_ban', **params)
        
        logger.info(f"调用方 {sender_id} 通过工具禁言了用户 {target_user_id} ({user_name})，时长 {duration} 秒。")
        
        return json.dumps({
            "success": True,
            "message": f"用户 {user_name} ({target_user_id}) 已被禁言 {duration} 秒。",
            "user_id": target_user_id,
            "user_name": user_name,
            "duration": duration,
            "timestamp": int(time.time())
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"禁言用户 {user_id} 失败: {e}")
        return json.dumps({
            "success": False,
            "message": f"操作失败：无法禁言用户 {user_name}",
            "error": str(e)
        }, ensure_ascii=False, indent=2)
