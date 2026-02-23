import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import aiosqlite
from astrbot.api import logger
from astrbot.api.star import StarTools

BLACKLIST_CACHE_TTL_SEC = 15
BLACKLIST_CACHE_MAX_SIZE = 4096
_BLACKLIST_CACHE: Dict[str, Tuple[bool, float]] = {}


def _set_blacklist_cache(user_id: str, is_blocked: bool, now_ts: float) -> None:
    """缓存 blacklist 结果，避免高频 DB 查询。"""
    if len(_BLACKLIST_CACHE) >= BLACKLIST_CACHE_MAX_SIZE:
        expired_keys = [k for k, (_, exp_ts) in _BLACKLIST_CACHE.items() if exp_ts <= now_ts]
        for key in expired_keys:
            _BLACKLIST_CACHE.pop(key, None)
        if len(_BLACKLIST_CACHE) >= BLACKLIST_CACHE_MAX_SIZE:
            _BLACKLIST_CACHE.clear()
    _BLACKLIST_CACHE[user_id] = (is_blocked, now_ts + BLACKLIST_CACHE_TTL_SEC)


def _get_blacklist_db_path() -> Optional[Path]:
    """获取黑名单插件数据库路径"""
    try:
        target_data_dir = StarTools.get_data_dir("astrbot_plugin_blacklist_tools")
        db_path = Path(target_data_dir) / "blacklist.db"

        if db_path.exists():
            return db_path
        logger.debug(f"[LLMEnhancement] 未找到黑名单数据库。目标路径: {db_path}")
    except Exception as e:
        logger.error(f"[LLMEnhancement] 获取黑名单插件数据目录失败: {e}")
    return None


async def is_user_blacklisted_via_blacklist_plugin(user_id: str) -> bool:
    """通过黑名单插件查询用户是否在黑名单中"""
    now_ts = time.time()
    cached = _BLACKLIST_CACHE.get(user_id)
    if cached and cached[1] > now_ts:
        return cached[0]

    db_path = _get_blacklist_db_path()
    if db_path is None:
        _set_blacklist_cache(user_id, False, now_ts)
        return False

    try:
        async with aiosqlite.connect(str(db_path)) as db:
            cursor = await db.execute(
                "SELECT * FROM blacklist WHERE user_id = ?",
                (user_id,),
            )
            user = await cursor.fetchone()
            if not user:
                _set_blacklist_cache(user_id, False, now_ts)
                return False

            expire_time_str = user[2]
            if expire_time_str:
                try:
                    expire_datetime = datetime.fromisoformat(expire_time_str)
                    if datetime.now() > expire_datetime:
                        _set_blacklist_cache(user_id, False, now_ts)
                        return False
                    logger.info(f"[LLMEnhancement] 用户 {user_id} 在黑名单中 (到期时间: {expire_time_str})")
                    _set_blacklist_cache(user_id, True, now_ts)
                    return True
                except ValueError:
                    _set_blacklist_cache(user_id, True, now_ts)
                    return True

            logger.info(f"[LLMEnhancement] 用户 {user_id} 在永久黑名单中")
            _set_blacklist_cache(user_id, True, now_ts)
            return True
    except Exception as e:
        logger.error(f"[LLMEnhancement] 查询 blacklist 插件数据库失败: {e}")
        return False
