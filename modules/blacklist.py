import asyncio
import base64
import io
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import aiosqlite
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp
from .qq_utils import (
    check_self_only_operation,
    _parse_user_id_list,
    get_group_info_internal,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

FONT_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "font" / "MiSans-Regular.ttf"
)
_FONT_CACHE: dict[int, object] = {}
BLACKLIST_WORDING_HINT = (
    "向用户转述时请使用“拉黑/解除拉黑”表述，不要使用“解封/解禁”表述。"
)
PROTECTED_USER_BLOCK_MESSAGE = "该用户 ID 在禁止拉黑名单中，不能拉黑。"
BLACKLIST_LEVEL_LABELS = {
    "llm_only": "LLM 请求",
    "command_and_llm": "指令与 LLM 请求",
    "all_messages": "所有消息",
}


def _fmt_user(uid: str, name: str = "") -> str:
    if name:
        return f"{name}({uid})"
    return uid


def _parse_iso_datetime(iso_text: Optional[str]) -> Optional[datetime]:
    if not iso_text:
        return None
    try:
        return datetime.fromisoformat(iso_text)
    except Exception:
        return None


def _load_font(font_size: int):
    if ImageFont is None:
        return None
    if font_size not in _FONT_CACHE:
        try:
            if FONT_PATH.exists():
                _FONT_CACHE[font_size] = ImageFont.truetype(str(FONT_PATH), font_size)
            else:
                _FONT_CACHE[font_size] = ImageFont.load_default()
                logger.warning("[LLMEnhancement] 黑名单字体缺失，回退默认字体。")
        except Exception as e:
            logger.warning(f"[LLMEnhancement] 黑名单字体加载失败，回退默认字体: {e}")
            _FONT_CACHE[font_size] = ImageFont.load_default()
    return _FONT_CACHE[font_size]


def _calc_text_width(lines: list[str], font, font_size: int) -> int:
    if not lines:
        return 100
    max_width = 0
    for line in lines:
        if not line:
            continue
        try:
            line_width = int(font.getlength(line))
        except Exception:
            line_width = len(line) * max(font_size // 2, 8)
        max_width = max(max_width, line_width)
    return max(max_width, 100)


def _truncate_text_to_fit(text: str, font, max_px: int) -> str:
    if not text:
        return text
    try:
        if font.getlength(text) <= max_px:
            return text
    except Exception:
        return text
    ellipsis = "……"
    try:
        ellipsis_w = font.getlength(ellipsis)
    except Exception:
        ellipsis_w = font.size * 2
    if max_px <= ellipsis_w:
        try:
            single = "…"
            return single if font.getlength(single) <= max_px else ""
        except Exception:
            return ""
    available = max_px - ellipsis_w
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            w = font.getlength(text[:mid])
        except Exception:
            w = font.size * mid
        if w <= available:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ellipsis


def _render_table_to_image_base64(
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    footers: list[str] | None = None,
    max_col_widths: list[int] | None = None,
    show_header: bool = True,
) -> Optional[str]:
    if Image is None or ImageDraw is None or ImageFont is None:
        return None

    try:
        c_bg = (26, 27, 38)
        c_title = (169, 177, 214)
        c_header_bg = (50, 52, 74)
        c_header_text = (122, 162, 247)
        c_row_even = (36, 37, 57)
        c_row_odd = (26, 27, 38)
        c_row_text = (154, 165, 206)
        c_sep = (47, 51, 70)
        c_footer = (90, 100, 130)

        title_font = _load_font(28)
        header_font = _load_font(18)
        body_font = _load_font(16)
        footer_font = _load_font(14)
        if not all([title_font, header_font, body_font, footer_font]):
            return None

        cell_pad_x = 14
        cell_pad_y = 5
        sep_w = 1
        n_cols = len(headers)

        # 自然宽度（浮点精度），然后向上取整避免截断。
        nat_widths: list[float] = []
        for i in range(n_cols):
            h_w = header_font.getlength(headers[i])
            max_w = h_w
            for row in rows:
                if i < len(row):
                    r_w = body_font.getlength(row[i])
                    if r_w > max_w:
                        max_w = r_w
            nat_widths.append(max_w + 2 * cell_pad_x)

        were_capped: list[bool] = []
        col_widths: list[int] = []
        for i in range(n_cols):
            w = math.ceil(nat_widths[i])
            if max_col_widths and i < len(max_col_widths) and w > max_col_widths[i]:
                w = max_col_widths[i]
                were_capped.append(True)
            else:
                were_capped.append(False)
            col_widths.append(w)

        # 仅截断受限列中的单元格。
        out_rows: list[list[str]] = []
        for row in rows:
            out_row: list[str] = []
            for i in range(n_cols):
                raw = row[i] if i < len(row) else ""
                if were_capped[i]:
                    avail = col_widths[i] - 2 * cell_pad_x
                    out_row.append(_truncate_text_to_fit(raw, body_font, avail))
                else:
                    out_row.append(raw)
            out_rows.append(out_row)

        total_w = sum(col_widths) + 2 * sep_w
        header_h = int(header_font.size) + 2 * cell_pad_y
        body_h = int(body_font.size) + 2 * cell_pad_y + 2
        sep_y = 2
        title_top = 16
        title_bot = 10
        footer_top = 8
        footer_bot = 14

        title_area_h = title_top + int(title_font.size) + title_bot
        footer_area_h = (
            (int(footer_font.size) + 4) * len(footers or []) + footer_top + footer_bot
        )
        if show_header:
            total_h = (
                title_area_h
                + sep_y
                + header_h
                + sep_y
                + body_h * len(rows)
                + sep_w
                + footer_area_h
            )
        else:
            total_h = title_area_h + sep_y + body_h * len(rows) + sep_w + footer_area_h

        img = Image.new("RGB", (total_w, total_h), c_bg)
        draw = ImageDraw.Draw(img)

        y = title_top
        draw.text((total_w // 2, y), title, font=title_font, fill=c_title, anchor="mt")
        y = title_area_h

        # 标题下方分隔线
        draw.rectangle([(0, y), (total_w - 1, y + sep_y - 1)], fill=c_sep)
        y += sep_y

        if show_header:
            # 表头行
            draw.rectangle([(0, y), (total_w, y + header_h)], fill=c_header_bg)
            x = sep_w
            for i, h in enumerate(headers):
                draw.text(
                    (x + cell_pad_x, y + cell_pad_y),
                    h,
                    font=header_font,
                    fill=c_header_text,
                )
                x += col_widths[i]
                if i < n_cols - 1:
                    draw.line([(x, y), (x, y + header_h)], fill=c_sep, width=sep_w)
            y += header_h

            # 表头下方分隔线
            draw.rectangle([(0, y), (total_w - 1, y + sep_y - 1)], fill=c_sep)
            y += sep_y

        # 数据行
        for ri, row in enumerate(out_rows):
            row_bg = c_row_even if ri % 2 == 0 else c_row_odd
            draw.rectangle([(0, y), (total_w, y + body_h)], fill=row_bg)
            x = sep_w
            for ci in range(n_cols):
                draw.text(
                    (x + cell_pad_x, y + cell_pad_y),
                    row[ci],
                    font=body_font,
                    fill=c_row_text,
                )
                x += col_widths[ci]
                if ci < n_cols - 1:
                    draw.line([(x, y), (x, y + body_h)], fill=c_sep, width=sep_w)
            y += body_h

        # 底部边框
        draw.rectangle([(0, y), (total_w - 1, y + sep_w - 1)], fill=c_sep)
        y += sep_w

        # 页脚
        y += footer_top
        for ft in footers or []:
            draw.text(
                (total_w // 2, y), ft, font=footer_font, fill=c_footer, anchor="mt"
            )
            y += int(footer_font.size) + 4

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"[LLMEnhancement] 表格渲染图片失败: {e}")
        return None


async def table_to_image_base64(
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    footers: list[str] | None = None,
    max_col_widths: list[int] | None = None,
    show_header: bool = True,
) -> Optional[str]:
    """将表格数据渲染为图片并返回 Base64 编码的字符串。"""
    return await asyncio.to_thread(
        _render_table_to_image_base64,
        title=title,
        headers=headers,
        rows=rows,
        footers=footers,
        max_col_widths=max_col_widths,
        show_header=show_header,
    )


@dataclass
class BlacklistCommandResult:
    text: str
    image_base64: Optional[str] = None


class BlacklistDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA cache_size=10000")
        await self._init_db()

    async def terminate(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_db(self) -> None:
        if not self._db:
            return
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS blacklist (
                user_id TEXT PRIMARY KEY,
                user_name TEXT,
                ban_time TEXT NOT NULL,
                expire_time TEXT,
                reason TEXT,
                level TEXT
            )
            """,
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_blacklist_expire_time ON blacklist(expire_time)",
        )
        cursor = await self._db.execute("PRAGMA table_info(blacklist)")
        columns = [str(row[1] or "") for row in await cursor.fetchall()]
        if "level" not in columns:
            await self._db.execute("ALTER TABLE blacklist ADD COLUMN level TEXT")
        await self._db.commit()

    async def get_all_users(self) -> list[tuple]:
        if not self._db:
            return []
        cursor = await self._db.execute(
            """
            SELECT user_id, user_name, ban_time, expire_time, reason, level
            FROM blacklist
            """,
        )
        return await cursor.fetchall()

    async def add_user(
        self,
        user_id: str,
        ban_time: str,
        user_name: str = "",
        expire_time: Optional[str] = None,
        reason: str = "",
        level: Optional[str] = None,
    ) -> bool:
        if not self._db:
            return False
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO blacklist (user_id, user_name, ban_time, expire_time, reason, level)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(user_id), user_name, ban_time, expire_time, reason, level),
            )
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"[LLMEnhancement] 添加黑名单用户失败 {user_id}: {e}")
            return False

    async def update_user_level(self, user_id: str, level: Optional[str]) -> bool:
        if not self._db:
            return False
        try:
            await self._db.execute(
                "UPDATE blacklist SET level = ? WHERE user_id = ?",
                (level, str(user_id)),
            )
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"[LLMEnhancement] 更新黑名单等级失败 {user_id}: {e}")
            return False

    async def remove_user(self, user_id: str) -> bool:
        if not self._db:
            return False
        try:
            await self._db.execute(
                "DELETE FROM blacklist WHERE user_id = ?",
                (str(user_id),),
            )
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"[LLMEnhancement] 移除黑名单用户失败 {user_id}: {e}")
            return False

    async def remove_users(self, user_ids: list[str]) -> int:
        if not self._db or not user_ids:
            return 0
        try:
            placeholders = ",".join("?" * len(user_ids))
            cursor = await self._db.execute(
                f"DELETE FROM blacklist WHERE user_id IN ({placeholders})",
                [str(uid) for uid in user_ids],
            )
            await self._db.commit()
            removed = cursor.rowcount if cursor.rowcount is not None else 0
            if removed < 0:
                changes_cursor = await self._db.execute("SELECT changes()")
                row = await changes_cursor.fetchone()
                removed = int(row[0]) if row else 0
            return removed
        except Exception as e:
            logger.error(f"[LLMEnhancement] 批量移除黑名单用户失败: {e}")
            return 0

    async def clear_blacklist(self) -> bool:
        if not self._db:
            return False
        try:
            await self._db.execute("DELETE FROM blacklist")
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"[LLMEnhancement] 清空黑名单失败: {e}")
            return False


@dataclass
class BlacklistRecord:
    user_id: str
    user_name: str
    ban_time: str
    expire_time: Optional[str]
    reason: str
    level: Optional[str]


class BlacklistManager:
    def __init__(
        self,
        data_dir: str | Path,
        get_cfg: Callable[[str, Any], Any],
        send_message_cb: Optional[Callable[[str, str], Any]] = None,
    ):
        self._data_dir = Path(data_dir)
        self._db = BlacklistDatabase(self._data_dir / "blacklist.db")
        self._get_cfg = get_cfg
        self._send_message_cb = send_message_cb
        self._records: Dict[str, BlacklistRecord] = {}

    def _is_record_expired(self, record: BlacklistRecord) -> bool:
        if not record.expire_time:
            return False
        expire_dt = _parse_iso_datetime(record.expire_time)
        if not expire_dt:
            return False
        return expire_dt <= datetime.now()

    def _cfg_str(self, key: str, default: str) -> str:
        raw = self._get_cfg(key, default)
        return str(raw if raw is not None else default).strip()

    def max_blacklist_duration(self) -> int:
        return max(0, int(self._get_cfg("max_blacklist_duration", 86400)))

    def protected_blacklist_user_ids(self) -> set[str]:
        raw_ids = self._get_cfg("protected_blacklist_user_ids", []) or []
        if not isinstance(raw_ids, (list, tuple, set)):
            return set()
        return {
            str(uid).strip()
            for uid in raw_ids
            if uid is not None and str(uid).strip()
        }

    def allow_blacklist_level_param(self) -> bool:
        return bool(self._get_cfg("allow_blacklist_level_param", False))

    def blacklist_intercept_level(self) -> str:
        raw = self._cfg_str("blacklist_intercept_level", "llm_only").lower()
        if raw in {"llm_only", "command_and_llm", "all_messages"}:
            return raw
        return "llm_only"

    def _normalize_level(self, raw: Any) -> Optional[str]:
        text = str(raw or "").strip().lower()
        if text in BLACKLIST_LEVEL_LABELS:
            return text
        return None

    def _level_label(self, level: Optional[str]) -> str:
        normalized = self._normalize_level(level)
        if not normalized:
            return ""
        return BLACKLIST_LEVEL_LABELS.get(normalized, "")

    def _get_notify_sessions(self) -> list[str]:
        raw_sessions = self._get_cfg("blacklist_notify_sessions", []) or []
        if not isinstance(raw_sessions, (list, tuple, set)):
            return []
        sessions: list[str] = []
        for item in raw_sessions:
            text = str(item or "").strip()
            if text:
                sessions.append(text)
        return sessions

    async def _effective_intercept_level_for(self, user_id: str) -> str:
        record = self._records.get(user_id)
        if record:
            record_level = self._normalize_level(record.level)
            if record_level:
                return record_level
        return self.blacklist_intercept_level()

    def _should_render_image(self) -> bool:
        mode = self._cfg_str("blacklist_output_mode", "image").lower()
        return mode != "text"

    async def initialize(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        await self._db.initialize()
        await self._load_all_records()

    async def _load_all_records(self) -> None:
        """启动时全量加载黑名单到内存镜像。加载失败仅记日志，空镜像继续运行。"""
        try:
            rows = await self._db.get_all_users()
        except Exception as e:
            logger.error(f"[LLMEnhancement] 黑名单全量加载失败，内存镜像为空: {e}")
            return
        expired_ids: list[str] = []
        for row in rows:
            record = BlacklistRecord(
                user_id=str(row[0] or ""),
                user_name=str(row[1] or ""),
                ban_time=str(row[2] or ""),
                expire_time=row[3],
                reason=str(row[4] or ""),
                level=row[5],
            )
            if self._is_record_expired(record):
                expired_ids.append(record.user_id)
                continue
            self._records[record.user_id] = record
        if expired_ids:
            removed = await self._db.remove_users(expired_ids)
            logger.info(
                f"[LLMEnhancement] 黑名单加载时过滤过期记录 {len(expired_ids)} 条"
                f"（数据库清理 {removed} 条）。"
            )

    async def terminate(self) -> None:
        await self._db.terminate()

    async def _cleanup_expired_on_query(self) -> None:
        expired_ids = [
            user_id
            for user_id, record in self._records.items()
            if self._is_record_expired(record)
        ]
        if not expired_ids:
            return
        for user_id in expired_ids:
            self._records.pop(user_id, None)
        removed = await self._db.remove_users(expired_ids)
        logger.info(
            f"[LLMEnhancement] 黑名单清理过期记录 {len(expired_ids)} 条"
            f"（数据库清理 {removed} 条）。"
        )

    def _parse_duration_seconds(
        self, duration: Any
    ) -> tuple[Optional[int], Optional[str]]:
        if duration is None or duration == "":
            return 0, None
        if isinstance(duration, bool):
            return None, "duration 参数格式错误，请传入秒数。"
        if isinstance(duration, (int, float)):
            return max(int(duration), 0), None

        text = str(duration).strip()
        if not text:
            return 0, None

        if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
            return max(int(float(text)), 0), None

        match = re.search(r"[+-]?\d+", text)
        if match:
            return max(int(match.group(0)), 0), None

        return None, "duration 参数格式错误，请传入秒数。"

    def _parse_user_ref(self, user_ref: str) -> tuple[Optional[str], str]:
        text = str(user_ref or "").strip()
        if not text:
            return None, ""

        if text.isdigit():
            return text, ""

        # OneBot CQ at 格式: [CQ:at,qq=123456]。
        cq_match = re.search(r"\[CQ:at,qq=([^,\]]+)", text, flags=re.IGNORECASE)
        if cq_match:
            target = str(cq_match.group(1) or "").strip()
            if target and target.lower() != "all":
                return target, ""

        # aiocqhttp message_str 中常见格式: @昵称(123456)。
        named_match = re.search(r"@?([^\(\)\s]+)?\(([^()\s]+)\)$", text)
        if named_match:
            target_name = str(named_match.group(1) or "").strip()
            target_id = str(named_match.group(2) or "").strip()
            if target_id and target_id.lower() != "all":
                return target_id, target_name

        # 通用兜底：提取 at 风格文本中的 id。
        generic_match = re.search(r"@([A-Za-z0-9_\-:]+)$", text)
        if generic_match:
            target = str(generic_match.group(1) or "").strip()
            if target and target.lower() != "all":
                return target, ""

        return None, ""

    def _extract_first_mention_target(
        self, event: AstrMessageEvent
    ) -> tuple[Optional[str], str]:
        if not hasattr(event, "message_obj") or not hasattr(
            event.message_obj, "message"
        ):
            return None, ""

        self_id = str(event.get_self_id() or "")
        for seg in event.message_obj.message or []:
            if not isinstance(seg, Comp.At):
                continue
            target_id = str(getattr(seg, "qq", "") or "").strip()
            if (
                not target_id
                or target_id.lower() == "all"
                or (self_id and target_id == self_id)
            ):
                continue
            target_name = str(getattr(seg, "name", "") or "").strip()
            return target_id, target_name

        return None, ""

    def _extract_all_mention_targets(
        self, event: AstrMessageEvent
    ) -> list[tuple[str, str]]:
        if not hasattr(event, "message_obj") or not hasattr(
            event.message_obj, "message"
        ):
            return []

        self_id = str(event.get_self_id() or "")
        results: list[tuple[str, str]] = []
        for seg in event.message_obj.message or []:
            if not isinstance(seg, Comp.At):
                continue
            target_id = str(getattr(seg, "qq", "") or "").strip()
            if (
                not target_id
                or target_id.lower() == "all"
                or (self_id and target_id == self_id)
            ):
                continue
            target_name = str(getattr(seg, "name", "") or "").strip()
            results.append((target_id, target_name))
        return results

    def _resolve_target_user(
        self, event: AstrMessageEvent, user_ref: str = ""
    ) -> tuple[Optional[str], str]:
        target_id, target_name = self._parse_user_ref(user_ref)
        if target_id:
            if not target_name:
                mention_id, mention_name = self._extract_first_mention_target(event)
                if mention_id and mention_id == target_id:
                    target_name = mention_name
            return target_id, target_name

        return self._extract_first_mention_target(event)

    async def is_user_blacklisted(self, user_id: str) -> bool:
        if not user_id:
            return False

        record = self._records.get(user_id)
        if not record:
            return False

        if self._is_record_expired(record):
            self._records.pop(user_id, None)
            await self._db.remove_user(user_id)
            logger.info(f"[LLMEnhancement] 黑名单记录已过期，自动移除 {user_id}。")
            return False
        return True

    async def intercept_event(
        self, event: AstrMessageEvent, command_trigger: bool = False
    ) -> bool:
        sender_id = str(event.get_sender_id() or "")
        if not sender_id:
            return False

        blocked = await self.is_user_blacklisted(sender_id)
        if not blocked:
            return False

        effective_level = await self._effective_intercept_level_for(sender_id)
        if effective_level == "all_messages":
            logger.debug(f"[LLMEnhancement] {sender_id} 在黑名单中，已拦截消息。")
            event.stop_event()
            return True
        if effective_level == "command_and_llm" and command_trigger:
            logger.debug(f"[LLMEnhancement] {sender_id} 在黑名单中，已拦截指令消息。")
            event.stop_event()
            return True
        return False

    async def intercept_llm_request(self, event: AstrMessageEvent) -> bool:
        sender_id = str(event.get_sender_id() or "")
        if not sender_id:
            return False

        blocked = await self.is_user_blacklisted(sender_id)
        if not blocked:
            return False

        logger.debug(f"[LLMEnhancement] {sender_id} 在黑名单中，已拦截 LLM 请求。")
        event.stop_event()
        return True

    def _schedule_block_notify(
        self,
        *,
        event: AstrMessageEvent,
        users: list[tuple[str, str]],
        duration: int,
        reason: str,
        level: Optional[str],
    ) -> None:
        if not users:
            return
        if self._send_message_cb is None:
            return
        sessions = self._get_notify_sessions()
        if not sessions:
            return
        asyncio.create_task(
            self._notify_block_task(
                event=event,
                users=users,
                duration=duration,
                reason=reason,
                level=level,
                sessions=sessions,
            )
        )

    async def _notify_block_task(
        self,
        *,
        event: AstrMessageEvent,
        users: list[tuple[str, str]],
        duration: int,
        reason: str,
        level: Optional[str],
        sessions: list[str],
    ) -> None:
        try:
            text = await self._build_block_notify_text(
                event=event,
                users=users,
                duration=duration,
                reason=reason,
                level=level,
            )
        except Exception as e:
            logger.warning(f"[LLMEnhancement] 构建拉黑通知失败: {e}")
            return
        for umo in sessions:
            try:
                await self._send_message_cb(umo, text)
            except Exception as e:
                logger.warning(f"[LLMEnhancement] 拉黑通知发送失败 umo={umo}: {e}")

    async def _build_block_notify_text(
        self,
        *,
        event: AstrMessageEvent,
        users: list[tuple[str, str]],
        duration: int,
        reason: str,
        level: Optional[str],
    ) -> str:
        lines = ["【黑名单通知】"]
        user_text = "、".join(_fmt_user(tid, name) for tid, name in users)
        lines.append(f"用户: {user_text}")
        gid = str(event.get_group_id() or "").strip()
        if gid:
            group_label = gid
            try:
                group_info = await get_group_info_internal(event, group_id=gid)
                if group_info:
                    group_name = str(
                        group_info.get("group_name") or group_info.get("name") or ""
                    ).strip()
                    if group_name:
                        group_label = f"{group_name}({gid})"
            except Exception:
                pass
            lines.append(f"群: {group_label}")
        lines.append(f"时长: {duration if duration and duration > 0 else '永久'}")
        if reason:
            lines.append(f"原因: {reason}")
        if level:
            lines.append(f"等级: {self._level_label(level)}")
        return "\n".join(lines)

    def _format_datetime(
        self,
        iso_datetime_str: Optional[str],
        *,
        show_remaining: bool = False,
        check_expire: bool = False,
    ) -> str:
        if not iso_datetime_str:
            return "永久"
        try:
            datetime_obj = datetime.fromisoformat(iso_datetime_str)
            if check_expire and datetime.now() > datetime_obj:
                return "已过期"

            formatted_time = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
            if not show_remaining:
                return formatted_time

            if datetime.now() > datetime_obj:
                return "已过期"

            remaining_time = datetime_obj - datetime.now()
            days = remaining_time.days
            hours, remainder = divmod(remaining_time.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            return f"{formatted_time} (剩余: {days}天 {hours}小时 {minutes}分钟)"
        except Exception:
            return "格式错误"

    def _format_datetime_compact(
        self,
        iso_datetime_str: Optional[str],
        *,
        check_expire: bool = False,
    ) -> str:
        if not iso_datetime_str:
            return "永久"
        try:
            datetime_obj = datetime.fromisoformat(iso_datetime_str)
            if check_expire and datetime.now() > datetime_obj:
                return "已过期"
            return datetime_obj.strftime("%m-%d %H:%M")
        except Exception:
            return "格式错误"

    def _text_display_width(self, text: str) -> int:
        width = 0
        for ch in text:
            width += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
        return width

    def _truncate_for_table(self, text: str, max_width: int) -> str:
        raw = str(text or "")
        if self._text_display_width(raw) <= max_width:
            return raw

        ellipsis = "……"
        ellipsis_width = self._text_display_width(ellipsis)
        if max_width <= ellipsis_width:
            return ellipsis[:max_width]

        output = []
        current_width = 0
        for ch in raw:
            ch_width = 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
            if current_width + ch_width + ellipsis_width > max_width:
                break
            output.append(ch)
            current_width += ch_width
        return "".join(output) + ellipsis

    def _pad_for_table(self, text: str, width: int) -> str:
        clipped = self._truncate_for_table(text, width)
        pad = width - self._text_display_width(clipped)
        if pad > 0:
            clipped += " " * pad
        return clipped

    async def command_ls(
        self, page: int = 1, page_size: int = 10
    ) -> BlacklistCommandResult:
        await self._cleanup_expired_on_query()

        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 10), 50))
        total_count = len(self._records)
        if total_count == 0:
            return BlacklistCommandResult(text="黑名单为空。")

        total_pages = max(1, (total_count + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages

        sorted_records = sorted(
            self._records.values(), key=lambda r: r.ban_time, reverse=True
        )
        offset = (page - 1) * page_size
        users = sorted_records[offset : offset + page_size]

        text_headers = [
            self._pad_for_table("序号", 4),
            self._pad_for_table("用户ID", 14),
            self._pad_for_table("用户名", 12),
            self._pad_for_table("加入时间", 11),
            self._pad_for_table("过期时间", 11),
            self._pad_for_table("原因", 20),
        ]
        text_header_row = " | ".join(text_headers)
        text_lines = [
            "黑名单列表",
            "=" * len(text_header_row),
            text_header_row,
            "-" * len(text_header_row),
        ]

        table_headers = ["序号", "用户ID", "用户名", "加入时间", "过期时间", "原因"]
        table_rows: list[list[str]] = []

        for idx, user in enumerate(users, start=1 + (page - 1) * page_size):
            user_id, user_name, ban_time, expire_time, reason = (
                user.user_id,
                user.user_name,
                user.ban_time,
                user.expire_time,
                user.reason,
            )
            text_cells = [
                self._pad_for_table(str(idx), 4),
                self._pad_for_table(str(user_id or ""), 14),
                self._pad_for_table(str(user_name or "未知"), 12),
                self._pad_for_table(self._format_datetime_compact(ban_time), 11),
                self._pad_for_table(
                    self._format_datetime_compact(expire_time, check_expire=True), 11
                ),
                self._pad_for_table(str(reason or "无"), 20),
            ]
            text_lines.append(" | ".join(text_cells))

            table_rows.append(
                [
                    str(idx),
                    str(user_id or ""),
                    str(user_name or "未知"),
                    self._format_datetime_compact(ban_time),
                    self._format_datetime_compact(expire_time, check_expire=True),
                    str(reason or "无"),
                ]
            )

        table_footers = [f"第 {page}/{total_pages} 页，共 {total_count} 条记录"]
        text_lines.append(table_footers[0])
        if page > 1:
            link = f"上一页: /黑名单 列表 {page - 1} {page_size}"
            text_lines.append(link)
            table_footers.append(link)
        if page < total_pages:
            link = f"下一页: /黑名单 列表 {page + 1} {page_size}"
            text_lines.append(link)
            table_footers.append(link)

        text = "\n".join(text_lines)
        if self._should_render_image():
            image_base64 = await table_to_image_base64(
                title="黑名单列表",
                headers=table_headers,
                rows=table_rows,
                footers=table_footers,
                max_col_widths=[60, 140, 200, 150, 150, 400],
            )
        else:
            image_base64 = None
        return BlacklistCommandResult(text=text, image_base64=image_base64)

    async def command_info(
        self, event: AstrMessageEvent, user_ref: str = ""
    ) -> BlacklistCommandResult:
        await self._cleanup_expired_on_query()

        target_id, target_name = self._resolve_target_user(event, user_ref)
        if not target_id:
            return BlacklistCommandResult(text="请提供用户 ID 或 @目标用户。")

        record = self._records.get(target_id)
        if not record:
            return BlacklistCommandResult(
                text=f"{_fmt_user(target_id, target_name)} 不在黑名单中。"
            )

        user_name, ban_time, expire_time, reason, level = (
            record.user_name,
            record.ban_time,
            record.expire_time,
            record.reason,
            record.level,
        )
        display_name = user_name or ""
        level_label = self._level_label(level)
        text_lines = [
            f"{_fmt_user(target_id, display_name)} 的黑名单信息",
            "=" * 36,
            f"用户名: {display_name or '未知'}",
            f"加入时间: {self._format_datetime(ban_time)}",
            f"过期时间: {self._format_datetime(expire_time, show_remaining=True, check_expire=True)}",
            f"等级: {level_label}",
            f"原因: {reason or '无'}",
        ]
        text = "\n".join(text_lines)
        if self._should_render_image():
            expire_str = self._format_datetime(
                expire_time, show_remaining=True, check_expire=True
            )
            info_rows = [
                ["用户名", user_name or "未知"],
                ["加入时间", self._format_datetime(ban_time)],
                ["过期时间", expire_str],
            ]
            if level_label:
                info_rows.append(["等级", level_label])
            info_rows.append(["原因", reason or "无"])
            image_base64 = await table_to_image_base64(
                title=f"{_fmt_user(target_id, display_name)}",
                headers=["字段", "值"],
                rows=info_rows,
                max_col_widths=[120, 600],
                show_header=False,
            )
        else:
            image_base64 = None
        return BlacklistCommandResult(text=text, image_base64=image_base64)

    async def command_add(
        self,
        event: AstrMessageEvent,
        user_ref: str = "",
        duration: Any = 0,
        reason: str = "",
    ) -> str:
        mention_id, _mention_name = self._extract_first_mention_target(event)
        user_ref_text = str(user_ref or "").strip()
        duration_text = str(duration or "").strip()
        # 部分平台指令解析不会把 @ 写入 message_str，这里将被错位的参数纠正回来。
        if (
            mention_id
            and user_ref_text.isdigit()
            and duration_text
            and not duration_text.isdigit()
        ):
            reason = duration_text
            duration = user_ref_text
            user_ref = ""

        duration_sec, err = self._parse_duration_seconds(duration)
        if err:
            return err

        mentions = self._extract_all_mention_targets(event)
        if len(mentions) > 1:
            # 多个 @ → 批量模式。
            ids = [m[0] for m in mentions]
            names = [m[1] for m in mentions]
        else:
            ids = _parse_user_id_list(user_ref)
            if len(ids) <= 1:
                # 单用户模式（支持 @ 解析）
                target_id, target_name = self._resolve_target_user(event, user_ref)
                if not target_id:
                    return "请提供用户 ID 或 @目标用户。"
                ids = [target_id]
                names = [target_name]
            else:
                names = [""] * len(ids)

        ban_time = datetime.now().isoformat()
        expire_time = None
        if duration_sec and duration_sec > 0:
            expire_time = (datetime.now() + timedelta(seconds=duration_sec)).isoformat()

        success_items: list[tuple[str, str]] = []
        fail_messages: list[str] = []
        protected_ids = self.protected_blacklist_user_ids()
        for tid, tname in zip(ids, names):
            if tid in protected_ids:
                fail_messages.append(f"{_fmt_user(tid, tname)} 在禁止拉黑名单中")
                continue

            ok = await self._db.add_user(
                user_id=tid,
                user_name=tname,
                ban_time=ban_time,
                expire_time=expire_time,
                reason=reason or "",
            )
            if not ok:
                fail_messages.append(f"{_fmt_user(tid, tname)} 写入失败")
                continue

            self._records[tid] = BlacklistRecord(
                user_id=tid,
                user_name=tname,
                ban_time=ban_time,
                expire_time=expire_time,
                reason=reason or "",
                level=None,
            )
            success_items.append((tid, tname))

        parts: list[str] = []
        if success_items:
            if len(success_items) > 1:
                part = f"已拉黑 {len(success_items)} 人"
            else:
                sid, sname = success_items[0]
                if duration_sec and duration_sec > 0:
                    part = (
                        f"{_fmt_user(sid, sname)} 已加入黑名单，时长 {duration_sec} 秒"
                    )
                else:
                    part = f"{_fmt_user(sid, sname)} 已永久加入黑名单"
            parts.append(part)
        if fail_messages:
            parts.append(f"失败 {len(fail_messages)} 人：{'；'.join(fail_messages)}")
        return "，".join(parts) + "。" if parts else "操作失败。"

    async def command_rm(self, event: AstrMessageEvent, user_ref: str = "") -> str:
        await self._cleanup_expired_on_query()

        mentions = self._extract_all_mention_targets(event)
        if len(mentions) > 1:
            ids = [m[0] for m in mentions]
        else:
            ids = _parse_user_id_list(user_ref)
            if len(ids) <= 1:
                target_id, _ = self._resolve_target_user(event, user_ref)
                if not target_id:
                    return "请提供用户 ID 或 @目标用户。"
                ids = [target_id] if target_id else []

        success_items: list[tuple[str, str]] = []
        not_found_ids: list[str] = []
        fail_messages: list[str] = []
        for tid in ids:
            record = self._records.get(tid)
            if not record:
                not_found_ids.append(tid)
                continue

            ok = await self._db.remove_user(tid)
            if not ok:
                fail_messages.append(f"{_fmt_user(tid, record.user_name)} 删除失败")
                continue
            self._records.pop(tid, None)
            success_items.append((tid, record.user_name))

        parts: list[str] = []
        if success_items:
            part = (
                f"已解除拉黑 {len(success_items)} 人"
                if len(success_items) > 1
                else f"{_fmt_user(success_items[0][0], success_items[0][1])} 已解除拉黑"
            )
            parts.append(part)
        if not_found_ids:
            parts.append(f"用户 ID {'、'.join(not_found_ids)} 不在黑名单中")
        if fail_messages:
            parts.append("；".join(fail_messages))
        return "，".join(parts) + "。" if parts else "操作失败。"

    async def command_clear(self) -> str:
        await self._cleanup_expired_on_query()

        count = len(self._records)
        if count == 0:
            return "黑名单已经为空。"

        ok = await self._db.clear_blacklist()
        if not ok:
            return "清空黑名单时出错。"
        self._records.clear()
        return f"黑名单已清空，共移除 {count} 个用户。"

    async def tool_block_user(
        self,
        event: AstrMessageEvent,
        user_ids: str = "",
        user_name: str = "",
        duration: int = 0,
        reason: str = "",
        level: str = "",
    ) -> str:
        await self._cleanup_expired_on_query()

        user_id_list = _parse_user_id_list(user_ids)
        if not user_id_list:
            return json.dumps(
                {"success": False, "message": "请提供要拉黑的目标用户 ID。"},
                ensure_ascii=False,
            )

        sender_id = str(event.get_sender_id() or "")

        # 权限检查：非管理员只能操作自己。
        self_only_tools = self._get_cfg("self_only_tools", [])
        for target_user_id in user_id_list:
            permission_error = check_self_only_operation(
                event,
                "block_user",
                str(target_user_id or "").strip(),
                self_only_tools,
            )
            if permission_error:
                return permission_error

        parsed_duration, err = self._parse_duration_seconds(duration)
        if err:
            return json.dumps(
                {"success": False, "message": err},
                ensure_ascii=False,
            )

        level_val = self._normalize_level(level)
        invalid_level_hint = ""
        if self.allow_blacklist_level_param():
            if str(level or "").strip() and not level_val:
                invalid_level_hint = "，等级参数无效，已按默认处理"
        else:
            level_val = None

        actual_duration = parsed_duration or 0
        max_duration = self.max_blacklist_duration()
        if actual_duration == 0 and max_duration > 0:
            actual_duration = max_duration
        if max_duration > 0 and actual_duration > max_duration:
            actual_duration = max_duration

        ban_time = datetime.now().isoformat()
        expire_time = None
        if actual_duration > 0:
            expire_time = (
                datetime.now() + timedelta(seconds=actual_duration)
            ).isoformat()

        target_name = str(user_name or "").strip()

        results = []
        success_count = 0
        fail_count = 0
        notify_users: list[tuple[str, str]] = []
        protected_ids = self.protected_blacklist_user_ids()

        for target_user_id in user_id_list:
            target_user_id = str(target_user_id or "").strip()
            if not target_user_id:
                fail_count += 1
                results.append(
                    {
                        "user_id": target_user_id,
                        "success": False,
                        "error": "empty user_id",
                    }
                )
                continue

            is_self_defense = target_user_id == sender_id

            if target_user_id in protected_ids:
                fail_count += 1
                results.append(
                    {
                        "user_id": target_user_id,
                        "success": False,
                        "error": PROTECTED_USER_BLOCK_MESSAGE,
                    }
                )
                continue

            # 检查是否已在黑名单
            existing_record = self._records.get(target_user_id)
            if existing_record:
                existing_level = self._normalize_level(existing_record.level)
                level_updated = False
                if level_val and level_val != existing_level:
                    level_updated = await self._db.update_user_level(
                        target_user_id, level_val
                    )
                    if level_updated:
                        existing_record.level = level_val
                message = (
                    f"{_fmt_user(target_user_id, target_name)} 已在黑名单中，已更新等级"
                    if level_updated
                    else f"{_fmt_user(target_user_id, target_name)} 已在黑名单中，无需重复添加"
                )
                results.append(
                    {
                        "success": True,
                        "message": message + invalid_level_hint + "。",
                        "user_id": target_user_id,
                        "level": level_val or "",
                        "wording_hint": BLACKLIST_WORDING_HINT,
                    }
                )
                success_count += 1
                continue

            # 写入数据库前先确定本次展示名，避免误用 sender 名称。
            current_target_name = (
                target_name
                if target_user_id != sender_id
                else (str(event.get_sender_name() or "") or target_name)
            )
            ok = await self._db.add_user(
                user_id=target_user_id,
                user_name=current_target_name,
                ban_time=ban_time,
                expire_time=expire_time,
                reason=reason or "",
                level=level_val,
            )
            if not ok:
                fail_count += 1
                results.append(
                    {
                        "user_id": target_user_id,
                        "success": False,
                        "error": "数据库写入异常。",
                    }
                )
                continue

            self._records[target_user_id] = BlacklistRecord(
                user_id=target_user_id,
                user_name=current_target_name,
                ban_time=ban_time,
                expire_time=expire_time,
                reason=reason or "",
                level=level_val,
            )
            logger.info(
                f"[LLMEnhancement] {target_user_id} 已由 {sender_id} 通过 LLM 工具拉黑。"
            )
            notify_users.append((target_user_id, current_target_name))
            results.append(
                {
                    "success": True,
                    "message": f"{_fmt_user(target_user_id, current_target_name)} 已拉黑{invalid_level_hint}。",
                    "user_id": target_user_id,
                    "user_name": current_target_name,
                    "duration": actual_duration if actual_duration > 0 else "永久",
                    "reason": reason,
                    "level": level_val or "",
                    "hint": "操作已生效。"
                    if not is_self_defense
                    else "操作已生效，将来这段时间内对方向你发送的消息将被屏蔽。",
                    "wording_hint": BLACKLIST_WORDING_HINT,
                }
            )
            success_count += 1

        if notify_users:
            self._schedule_block_notify(
                event=event,
                users=notify_users,
                duration=actual_duration,
                reason=reason or "",
                level=level_val,
            )

        if len(user_id_list) == 1:
            if results and results[0].get("success"):
                return json.dumps(results[0], ensure_ascii=False)
            return json.dumps(
                results[0] if results else {"success": False, "message": "操作失败。"},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "success_count": success_count,
                "fail_count": fail_count,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )

    async def tool_unblock_user(self, event: AstrMessageEvent, user_ids: str) -> str:
        await self._cleanup_expired_on_query()

        user_id_list = _parse_user_id_list(user_ids)
        if not user_id_list:
            return json.dumps(
                {"success": False, "message": "请提供要解除拉黑的用户 ID。"},
                ensure_ascii=False,
            )

        sender_id = str(event.get_sender_id() or "")

        # 权限检查：非管理员只能操作自己。
        self_only_tools = self._get_cfg("self_only_tools", [])
        for target_user_id in user_id_list:
            permission_error = check_self_only_operation(
                event,
                "unblock_user",
                str(target_user_id or "").strip(),
                self_only_tools,
            )
            if permission_error:
                return permission_error

        results = []
        success_count = 0
        fail_count = 0

        for target_user_id in user_id_list:
            target_user_id = str(target_user_id or "").strip()
            if not target_user_id:
                fail_count += 1
                results.append(
                    {
                        "user_id": target_user_id,
                        "success": False,
                        "error": "empty user_id",
                    }
                )
                continue

            record = self._records.get(target_user_id)
            if not record:
                results.append(
                    {
                        "success": True,
                        "message": f"{target_user_id} 不在黑名单中。",
                        "user_id": target_user_id,
                        "user_name": "",
                        "wording_hint": BLACKLIST_WORDING_HINT,
                    }
                )
                success_count += 1
                continue

            user_name = record.user_name
            ok = await self._db.remove_user(target_user_id)
            if not ok:
                fail_count += 1
                results.append(
                    {
                        "user_id": target_user_id,
                        "success": False,
                        "error": "数据库删除失败。",
                    }
                )
                continue

            self._records.pop(target_user_id, None)
            logger.info(
                f"[LLMEnhancement] {target_user_id} 已由 {sender_id} 通过 LLM 工具解除拉黑。"
            )
            results.append(
                {
                    "success": True,
                    "message": f"{_fmt_user(target_user_id, user_name)} 已解除拉黑。",
                    "user_id": target_user_id,
                    "user_name": user_name or "",
                    "wording_hint": BLACKLIST_WORDING_HINT,
                }
            )
            success_count += 1

        if len(user_id_list) == 1:
            if results and results[0].get("success"):
                return json.dumps(results[0], ensure_ascii=False)
            return json.dumps(
                results[0] if results else {"success": False, "message": "操作失败。"},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "success_count": success_count,
                "fail_count": fail_count,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )

    async def tool_list_blacklist(
        self, event: AstrMessageEvent, page: int = 1, page_size: int = 20
    ) -> str:
        await self._cleanup_expired_on_query()

        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 20), 50))
        total_count = len(self._records)
        if total_count == 0:
            return json.dumps(
                {
                    "total_count": 0,
                    "total_pages": 0,
                    "current_page": 1,
                    "page_size": page_size,
                    "has_more": False,
                    "next_page": None,
                    "users": [],
                    "expire_time_hint": "expire_time 表示该用户黑名单失效时间，失效后意味着你将其移出黑名单。",
                    "wording_hint": BLACKLIST_WORDING_HINT,
                },
                ensure_ascii=False,
            )

        total_pages = max(1, (total_count + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages

        sorted_records = sorted(
            self._records.values(), key=lambda r: r.ban_time, reverse=True
        )
        offset = (page - 1) * page_size
        page_records = sorted_records[offset : offset + page_size]
        users = []
        for record in page_records:
            users.append(
                {
                    "user_id": record.user_id,
                    "user_name": record.user_name or "",
                    "ban_time": record.ban_time,
                    "expire_time": record.expire_time if record.expire_time else "永久",
                    "reason": record.reason if record.reason else "无",
                    "level": record.level or "",
                },
            )

        return json.dumps(
            {
                "total_count": total_count,
                "total_pages": total_pages,
                "current_page": page,
                "page_size": page_size,
                "has_more": page < total_pages,
                "next_page": (page + 1) if page < total_pages else None,
                "users": users,
                "expire_time_hint": "expire_time 表示该用户黑名单失效时间，失效后意味着你将其移出黑名单。",
                "wording_hint": BLACKLIST_WORDING_HINT,
            },
            ensure_ascii=False,
        )

    async def tool_get_blacklist_status(
        self, event: AstrMessageEvent, user_ids: str
    ) -> str:
        await self._cleanup_expired_on_query()

        user_id_list = _parse_user_id_list(user_ids)
        if not user_id_list:
            return json.dumps(
                {
                    "is_blacklisted": False,
                    "user_id": "",
                    "message": "请提供要查询的目标用户 ID。",
                    "wording_hint": BLACKLIST_WORDING_HINT,
                },
                ensure_ascii=False,
            )

        results = []
        success_count = 0
        fail_count = 0

        for target_id in user_id_list:
            target_id = str(target_id or "").strip()
            if not target_id:
                fail_count += 1
                results.append(
                    {
                        "user_id": target_id,
                        "is_blacklisted": False,
                        "error": "empty user_id",
                    }
                )
                continue

            record = self._records.get(target_id)
            if record:
                results.append(
                    {
                        "is_blacklisted": True,
                        "user_id": record.user_id,
                        "user_name": record.user_name or "",
                        "ban_time": record.ban_time,
                        "expire_time": record.expire_time if record.expire_time else "永久",
                        "reason": record.reason if record.reason else "无",
                        "level": record.level or "",
                        "expire_time_hint": "expire_time 表示黑名单失效时间，失效后意味着你将其移出黑名单。",
                        "wording_hint": BLACKLIST_WORDING_HINT,
                    }
                )
            else:
                results.append(
                    {
                        "is_blacklisted": False,
                        "user_id": target_id,
                        "wording_hint": BLACKLIST_WORDING_HINT,
                    }
                )
            success_count += 1

        if len(user_id_list) == 1:
            return json.dumps(
                results[0]
                if results
                else {"is_blacklisted": False, "message": "查询失败。"},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "success_count": success_count,
                "fail_count": fail_count,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
