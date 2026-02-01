import asyncio
from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class MemberState(BaseModel):
    """群成员状态"""
    uid: str                                              # 用户ID
    silence_until: float = 0.0                            # 沉默截止时间（时间戳）
    last_request: float = 0.0                             # 最后一次发送LLM请求的时间（时间戳）
    last_response: float = 0.0                            # 最后一次LLM响应的时间（时间戳）
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)  # 异步锁
    in_merging: bool = False                              # 是否正在消息合并状态中
    pending_msg_ids: set = Field(default_factory=set)     # 正在合并中的消息ID集合
    cancel_merge: bool = False                            # 是否取消当前的合并流程
    trigger_msg_id: Optional[str] = None                  # 触发当前流程的首条消息ID
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GroupState(BaseModel):
    """群组状态"""
    gid: str                                              # 群组ID
    members: dict[str, MemberState] = Field(default_factory=dict)  # 成员状态字典
    shutup_until: float = 0.0                             # 群组闭嘴截止时间（时间戳）


class StateManager:
    """状态管理器 - 管理所有群组和成员的状态"""
    
    _groups: Dict[str, GroupState] = {}
    
    @classmethod
    def get_group(cls, gid: str) -> GroupState:
        """获取或创建群组状态"""
        if gid not in cls._groups:
            cls._groups[gid] = GroupState(gid=gid)
        return cls._groups[gid]
