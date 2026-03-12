import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class MemberState(BaseModel):
    """群成员状态"""
    uid: str                                               # 用户ID
    silence_until: float = 0.0                             # 沉默截止时间（时间戳）
    last_request: float = 0.0                              # 最后一次发送LLM请求的时间（时间戳）
    last_response: float = 0.0                             # 最后一次LLM响应的时间（时间戳）
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)  # 异步锁
    in_merging: bool = False                               # 是否正在消息合并状态中
    pending_msg_ids: set[str] = Field(default_factory=set) # 正在合并中的消息ID集合
    cancel_merge: bool = False                             # 是否取消当前的合并流程
    trigger_msg_id: Optional[str] = None                   # 触发当前流程的首条消息ID
    recent_wake_msgs: list[dict[str, Any]] = Field(default_factory=list) # 唤醒阶段预采样消息
    merged_msg_ids: dict[str, float] = Field(default_factory=dict)        # 已并入其他请求的消息ID过期表
    dynamic_unresolved_msgs: list[dict[str, Any]] = Field(default_factory=list)  # 动态合并待确认消息
    dynamic_request_seq: int = 0                            # 动态合并请求序号
    dynamic_inflight_seq: int = 0                           # 当前进行中的动态请求序号
    dynamic_discard_before_seq: int = 0                     # 小于等于该序号的响应将被丢弃
    dynamic_discarded_response_cache: dict[str, Any] = Field(default_factory=dict)  # 动态合并被丢弃响应缓存

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GroupState(BaseModel):
    """群组状态"""
    gid: str                                               # 群组ID
    members: dict[str, MemberState] = Field(default_factory=dict)   # 成员状态字典
    shutup_until: float = 0.0                              # 群组闭嘴截止时间（时间戳）
    pending_msg_index: dict[str, str] = Field(default_factory=dict) # 撤回快速索引：message_id -> uid
    last_response_uid: Optional[str] = None                # 最近一次触发响应的用户
    last_response_ts: float = 0.0                          # 最近一次响应时间（时间戳）
    wake_extend_consumed_ref_ts: float = 0.0               # 已消费一次性唤醒延长期所对应的响应时间戳
    dynamic_owner_uid: Optional[str] = None                # 动态合并当前会话所有者（用于多人合并）


class StateManager:
    """状态管理器 - 管理所有群组和成员的状态"""

    _groups: Dict[str, GroupState] = {}

    @classmethod
    def get_group(cls, gid: str) -> GroupState:
        """获取或创建群组状态"""
        if gid not in cls._groups:
            cls._groups[gid] = GroupState(gid=gid)
        return cls._groups[gid]

    @classmethod
    def get_group_if_exists(cls, gid: str) -> Optional[GroupState]:
        """仅获取已存在的群组状态，不创建。"""
        return cls._groups.get(gid)

    @classmethod
    def iter_groups_items(cls):
        """迭代所有群组状态条目。"""
        return tuple(cls._groups.items())
