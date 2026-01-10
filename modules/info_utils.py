import json
import time
from typing import List, Dict, Any, Optional
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

# 检查是否为 aiocqhttp 平台
try: 
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent 
    IS_AIOCQHTTP = True 
except ImportError: 
    IS_AIOCQHTTP = False 

async def get_group_members_internal(event: AstrMessageEvent) -> Optional[List[Dict[str, Any]]]: 
    """ 
    内部函数，用于调用API获取群成员列表 
    
    Args: 
        event: AstrMessageEvent实例 
        
    Returns: 
        群成员列表，失败时返回None 
    """ 
    try: 
        group_id = event.get_group_id() 
        if not group_id: 
            return None 

        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return None

        client = event.bot 
        params = {"group_id": group_id} 
        return await client.api.call_action('get_group_member_list', **params) 
    except Exception as e: 
        logger.info(f"API调用失败: {e}") 
        return None

async def process_group_members_info(event: AstrMessageEvent) -> str:
    """
    获取并处理QQ群成员信息的逻辑。
    """
    start_time = time.time() 
    
    try: 
        group_id = event.get_group_id() 
        if not group_id: 
            logger.info("用户在非群聊环境中调用群成员查询工具") 
            return json.dumps({"error": "这不是群聊"}) 
        
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent): 
            logger.info(f"不支持的平台: {event.get_platform_name()}") 
            return json.dumps({"error": f"此功能仅支持QQ群聊(aiocqhttp平台)，当前平台为 {event.get_platform_name()}"}) 

        # 从API获取 
        members_info = await get_group_members_internal(event) 
        if not members_info: 
            logger.info(f"无法获取群 {group_id} 的成员信息") 
            return json.dumps({"error": "获取群成员信息失败，可能是权限不足或网络问题"}) 
        
        processed_members = [ 
            { 
                "user_id": str(member.get("user_id", "")), 
                "display_name": member.get("card") or member.get("nickname") or f"用户{member.get('user_id')}", 
                "username": member.get("nickname") or f"用户{member.get('user_id')}",  # 用户的QQ昵称 
                "role": member.get("role", "member") 
            } 
            for member in members_info if member.get("user_id") 
        ] 
        
        group_info = { 
            "group_id": group_id, 
            "member_count": len(processed_members), 
            "members": processed_members 
        } 
        
        elapsed_time = time.time() - start_time 
        logger.info(f"成功获取群 {group_id} 的 {len(processed_members)} 名成员信息，耗时 {elapsed_time:.2f}s") 
        
        return json.dumps(group_info, ensure_ascii=False, indent=2) 
    except Exception as e: 
        elapsed_time = time.time() - start_time 
        logger.info(f"获取群成员信息时发生错误: {e}，耗时 {elapsed_time:.2f}s") 
        return json.dumps({"error": f"获取群成员信息时发生内部错误: {str(e)}"})

async def set_group_ban_logic(event: AstrMessageEvent, user_id: str, duration: int, user_name: str) -> str:
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

        group_id = event.get_group_id()
        if not group_id:
            return json.dumps({
                "success": False,
                "message": "此工具只能在群聊中使用。"
            }, ensure_ascii=False)

        # 2. 权限检查：仅限 Bot 管理员或 Bot 自身调用
        sender_id = event.get_sender_id()
        self_id = event.get_self_id()
        
        if sender_id != self_id and not event.is_admin():
            logger.warning(f"用户 {sender_id} 尝试使用禁言工具，但权限不足。")
            return json.dumps({
                "success": False,
                "message": "权限不足。只有 Bot 管理员可以执行禁言操作。"
            }, ensure_ascii=False)

        # 3. 执行禁言
        client = event.bot
        params = {
            "group_id": int(group_id),
            "user_id": int(user_id),
            "duration": duration
        }
        
        await client.api.call_action('set_group_ban', **params)
        
        logger.info(f"管理员 {event.get_sender_id()} 通过工具禁言了用户 {user_id} ({user_name})，时长 {duration} 秒。")
        
        return json.dumps({
            "success": True,
            "message": f"用户 {user_name} ({user_id}) 已被禁言 {duration} 秒。",
            "user_id": user_id,
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
