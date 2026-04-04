import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class MemberState(BaseModel):
    """群成员状态"""
    uid: str                                               # 用户ID
    last_response: float = 0.0                             # 最后一次LLM响应的时间（时间戳）
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)  # 异步锁
    in_merging: bool = False                               # 是否正在消息合并状态中
    pending_msg_ids: set[str] = Field(default_factory=set) # 正在合并中的消息ID集合
    cancel_merge: bool = False                             # 是否取消当前的合并流程
    trigger_msg_id: Optional[str] = None                   # 触发当前流程的首条消息ID
    recent_wake_msgs: list[dict[str, Any]] = Field(default_factory=list) # 唤醒阶段预采样消息
    last_wake_ts: float = 0.0                              # 最近一次唤醒时间（时间戳）
    merge_start_ts: float = 0.0                            # 合并会话起点时间（硬等待: 当前请求；动态: 当前会话）
    merged_msg_ids: dict[str, float] = Field(default_factory=dict)        # 已并入其他请求的消息ID过期表
    dynamic_unresolved_msgs: list[dict[str, Any]] = Field(default_factory=list)  # 动态合并待确认消息
    dynamic_request_seq: int = 0                            # 动态合并请求序号
    dynamic_inflight_seq: int = 0                           # 当前进行中的动态请求序号
    dynamic_discard_before_seq: int = 0                     # 小于等于该序号的响应将被丢弃
    dynamic_discarded_response_cache: dict[str, Any] = Field(default_factory=dict)  # 动态合并被丢弃响应缓存
    dynamic_capture_count: int = 0                          # 动态会话已接收（去重后）消息数量

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GroupState(BaseModel):
    """群组状态"""
    gid: str                                               # 群组ID
    members: dict[str, MemberState] = Field(default_factory=dict)   # 成员状态字典
    pending_msg_index: dict[str, str] = Field(default_factory=dict) # 撤回快速索引：message_id -> uid
    last_response_uid: Optional[str] = None                # 最近一次触发响应的用户
    last_response_ts: float = 0.0                          # 最近一次响应时间（时间戳）
    wake_extend_consumed_ref_ts: float = 0.0               # 已消费一次性唤醒延长期所对应的响应时间戳
    active_wake_new_msg_count: int = 0                     # 距上次Bot回复后累计的普通群消息数
    prob_wake_pending_count: int = 0                       # 距上次概率观察后累计的普通群消息数
    prob_wake_no_reply_count: int = 0                      # 概率观察连续未触发次数
    prob_wake_last_check_ts: float = 0.0                   # 最近一次概率观察时间戳
    wake_extend_batch_count: int = 0                       # 当前唤醒延长窗口内累计的普通后续消息数
    dynamic_owner_uid: Optional[str] = None                # 动态合并当前会话所有者（用于多人合并）
    context_messages: list[dict[str, Any]] = Field(default_factory=list) # 轻量上下文缓存（环形裁剪）
    last_user_interaction: dict[str, float] = Field(default_factory=dict) # 每个用户最近互动时间戳
    context_bot_last_replied_to_uid: str = ""              # Bot 最近一次回复目标用户ID


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
