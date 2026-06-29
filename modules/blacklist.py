import asyncio
import base64
import io
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import aiosqlite
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp
from .qq_utils import check_self_only_operation, _parse_user_id_list

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

CACHE_TTL_SEC = 15
CACHE_MAX_SIZE = 4096
FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "font"
    / "MiSans-Regular.ttf"
)
_FONT_CACHE: dict[int, object] = {}
BLACKLIST_WORDING_HINT = (
    "向用户转述时请使用“拉黑/解除拉黑”表述，"
    "不要使用“解封/解禁”表述。"
)
BOT_ADMIN_BLOCK_MESSAGE = (
    "这个人是管理员，我不能把他拉黑。"
)


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


def _render_text_to_image_base64(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    if Image is None or ImageDraw is None or ImageFont is None:
        return None

    try:
        font_size = 24
        padding = 20
        line_spacing = 6
        min_width = 420
        max_width = 1400

        font = _load_font(font_size)
        if font is None:
            return None

        lines = text.strip().splitlines()
        line_height = font_size + line_spacing
        text_height = max(line_height * max(len(lines), 1), font_size)
        text_width = _calc_text_width(lines, font, font_size)
        width = max(min_width, min(text_width + padding * 2, max_width))
        height = text_height + padding * 2

        image = Image.new("RGB", (width, height), (18, 18, 18))
        draw = ImageDraw.Draw(image)
        y = padding
        for line in lines:
            if line:
                draw.text((padding, y), line, font=font, fill=(245, 245, 245))
            y += line_height

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"[LLMEnhancement] 黑名单文本渲染图片失败，回退纯文本: {e}")
        return None


async def text_to_image_base64(text: str) -> Optional[str]:
    return await asyncio.to_thread(_render_text_to_image_base64, text)


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

        # natural width (float precision), then int-ceil to avoid truncation
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

        # only truncate cells in capped columns
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
        footer_area_h = (int(footer_font.size) + 4) * len(footers or []) + footer_top + footer_bot
        if show_header:
            total_h = title_area_h + sep_y + header_h + sep_y + body_h * len(rows) + sep_w + footer_area_h
        else:
            total_h = title_area_h + sep_y + body_h * len(rows) + sep_w + footer_area_h

        img = Image.new("RGB", (total_w, total_h), c_bg)
        draw = ImageDraw.Draw(img)

        y = title_top
        draw.text((total_w // 2, y), title, font=title_font, fill=c_title, anchor="mt")
        y = title_area_h

        # separator under title
        draw.rectangle([(0, y), (total_w - 1, y + sep_y - 1)], fill=c_sep)
        y += sep_y

        if show_header:
            # header row
            draw.rectangle([(0, y), (total_w, y + header_h)], fill=c_header_bg)
            x = sep_w
            for i, h in enumerate(headers):
                draw.text((x + cell_pad_x, y + cell_pad_y), h, font=header_font, fill=c_header_text)
                x += col_widths[i]
                if i < n_cols - 1:
                    draw.line([(x, y), (x, y + header_h)], fill=c_sep, width=sep_w)
            y += header_h

            # separator below header
            draw.rectangle([(0, y), (total_w - 1, y + sep_y - 1)], fill=c_sep)
            y += sep_y

        # data rows
        for ri, row in enumerate(out_rows):
            row_bg = c_row_even if ri % 2 == 0 else c_row_odd
            draw.rectangle([(0, y), (total_w, y + body_h)], fill=row_bg)
            x = sep_w
            for ci in range(n_cols):
                draw.text((x + cell_pad_x, y + cell_pad_y), row[ci], font=body_font, fill=c_row_text)
                x += col_widths[ci]
                if ci < n_cols - 1:
                    draw.line([(x, y), (x, y + body_h)], fill=c_sep, width=sep_w)
            y += body_h

        # bottom border
        draw.rectangle([(0, y), (total_w - 1, y + sep_w - 1)], fill=c_sep)
        y += sep_w

        # footers
        y += footer_top
        for ft in (footers or []):
            draw.text((total_w // 2, y), ft, font=footer_font, fill=c_footer, anchor="mt")
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
                reason TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_blacklist_expire_time ON blacklist(expire_time)"
        )
        await self._db.commit()

    async def get_user_info(self, user_id: str) -> Optional[tuple]:
        if not self._db:
            return None
        cursor = await self._db.execute(
            "SELECT user_id, user_name, ban_time, expire_time, reason FROM blacklist WHERE user_id = ?",
            (str(user_id),),
        )
        return await cursor.fetchone()

    async def get_blacklist_count(self) -> int:
        if not self._db:
            return 0
        cursor = await self._db.execute("SELECT COUNT(*) FROM blacklist")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def get_blacklist_users(self, page: int = 1, page_size: int = 10) -> list[tuple]:
        if not self._db:
            return []
        offset = max(page - 1, 0) * page_size
        cursor = await self._db.execute(
            """
            SELECT user_id, user_name, ban_time, expire_time, reason
            FROM blacklist
            ORDER BY ban_time DESC LIMIT ? OFFSET ?
            """,
            (page_size, offset),
        )
        return await cursor.fetchall()

    async def add_user(
        self,
        user_id: str,
        ban_time: str,
        user_name: str = "",
        expire_time: Optional[str] = None,
        reason: str = "",
    ) -> bool:
        if not self._db:
            return False
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO blacklist (user_id, user_name, ban_time, expire_time, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(user_id), user_name, ban_time, expire_time, reason),
            )
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"[LLMEnhancement] 添加黑名单用户失败 {user_id}: {e}")
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

    async def cleanup_expired_records(self) -> int:
        if not self._db:
            return 0
        try:
            now_iso = datetime.now().isoformat()
            cursor = await self._db.execute(
                """
                DELETE FROM blacklist
                WHERE expire_time IS NOT NULL
                  AND expire_time <= ?
                """,
                (now_iso,),
            )
            await self._db.commit()
            removed = cursor.rowcount if cursor.rowcount is not None else 0
            if removed < 0:
                changes_cursor = await self._db.execute("SELECT changes()")
                row = await changes_cursor.fetchone()
                removed = int(row[0]) if row else 0
            return removed
        except Exception as e:
            logger.warning(f"[LLMEnhancement] 清理过期黑名单失败: {e}")
            return 0


class BlacklistManager:
    def __init__(
        self,
        data_dir: str | Path,
        get_cfg: Callable[[str, Any], Any],
    ):
        self._data_dir = Path(data_dir)
        self._db = BlacklistDatabase(self._data_dir / "blacklist.db")
        self._get_cfg = get_cfg
        self._cache: Dict[str, Tuple[bool, float]] = {}

    def _set_cache(self, user_id: str, blocked: bool, now_ts: float) -> None:
        if len(self._cache) >= CACHE_MAX_SIZE:
            expired_keys = [k for k, (_, exp_ts) in self._cache.items() if exp_ts <= now_ts]
            for key in expired_keys:
                self._cache.pop(key, None)
            if len(self._cache) >= CACHE_MAX_SIZE:
                self._cache.clear()
        self._cache[user_id] = (blocked, now_ts + CACHE_TTL_SEC)

    def _get_cache(self, user_id: str, now_ts: float) -> Optional[bool]:
        cached = self._cache.get(user_id)
        if not cached:
            return None
        if cached[1] <= now_ts:
            self._cache.pop(user_id, None)
            return None
        return cached[0]

    def _invalidate_cache(self, user_id: str) -> None:
        self._cache.pop(user_id, None)

    def _clear_cache(self) -> None:
        self._cache.clear()

    def _cfg_int(self, key: str, default: int, *, min_value: Optional[int] = None) -> int:
        raw = self._get_cfg(key, default)
        try:
            value = int(raw)
        except Exception:
            value = default
        if min_value is not None and value < min_value:
            value = min_value
        return value

    def _cfg_str(self, key: str, default: str) -> str:
        raw = self._get_cfg(key, default)
        return str(raw if raw is not None else default).strip()

    def max_blacklist_duration(self) -> int:
        return self._cfg_int("max_blacklist_duration", 86400, min_value=0)

    def allow_blacklist_bot_admin(self) -> bool:
        return bool(self._get_cfg("allow_blacklist_bot_admin", False))

    def blacklist_intercept_level(self) -> str:
        raw = self._cfg_str("blacklist_intercept_level", "llm_only").lower()
        if raw in {"llm_only", "command_and_llm", "all_messages"}:
            return raw
        return "llm_only"

    def _should_render_image(self) -> bool:
        mode = self._cfg_str("blacklist_output_mode", "image").lower()
        return mode != "text"

    def _get_bot_admin_ids(self) -> set[str]:
        try:
            from astrbot.core import astrbot_config

            raw_admin_ids = astrbot_config.get("admins_id", [])
        except Exception:
            return set()

        if not isinstance(raw_admin_ids, (list, tuple, set)):
            return set()

        admin_ids: set[str] = set()
        for admin_id in raw_admin_ids:
            text = str(admin_id or "").strip()
            if text:
                admin_ids.add(text)
        return admin_ids

    def _is_target_bot_admin(self, event: AstrMessageEvent, target_user_id: str) -> bool:
        target_id = str(target_user_id or "").strip()
        if not target_id:
            return False
        sender_id = str(event.get_sender_id() or "").strip()
        if target_id == sender_id and bool(event.is_admin()):
            return True
        return target_id in self._get_bot_admin_ids()

    async def initialize(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        await self._db.initialize()

    async def terminate(self) -> None:
        await self._db.terminate()

    async def _cleanup_expired_on_query(self) -> None:
        removed = await self._db.cleanup_expired_records()
        if removed > 0:
            self._clear_cache()
            logger.info(f"[LLMEnhancement] 黑名单清理过期记录 {removed} 条。")

    async def _is_user_blacklisted_now(self, user_id: str) -> bool:
        user = await self._db.get_user_info(user_id)
        if not user:
            return False

        expire_time = user[3]
        expire_dt = _parse_iso_datetime(expire_time) if expire_time else None
        if expire_dt and expire_dt <= datetime.now():
            await self._db.remove_user(user_id)
            self._invalidate_cache(user_id)
            return False
        return True

    def _parse_duration_seconds(self, duration: Any) -> tuple[Optional[int], Optional[str]]:
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

        # OneBot CQ at 格式: [CQ:at,qq=123456]
        cq_match = re.search(r"\[CQ:at,qq=([^,\]]+)", text, flags=re.IGNORECASE)
        if cq_match:
            target = str(cq_match.group(1) or "").strip()
            if target and target.lower() != "all":
                return target, ""

        # aiocqhttp message_str 中常见格式: @昵称(123456)
        named_match = re.search(r"@?([^\(\)\s]+)?\(([^()\s]+)\)$", text)
        if named_match:
            target_name = str(named_match.group(1) or "").strip()
            target_id = str(named_match.group(2) or "").strip()
            if target_id and target_id.lower() != "all":
                return target_id, target_name

        # 通用兜底：提取 at 风格文本中的 id
        generic_match = re.search(r"@([A-Za-z0-9_\-:]+)$", text)
        if generic_match:
            target = str(generic_match.group(1) or "").strip()
            if target and target.lower() != "all":
                return target, ""

        return None, ""

    def _extract_first_mention_target(self, event: AstrMessageEvent) -> tuple[Optional[str], str]:
        if not hasattr(event, "message_obj") or not hasattr(event.message_obj, "message"):
            return None, ""

        self_id = str(event.get_self_id() or "")
        for seg in event.message_obj.message or []:
            if not isinstance(seg, Comp.At):
                continue
            target_id = str(getattr(seg, "qq", "") or "").strip()
            if not target_id or target_id.lower() == "all" or (self_id and target_id == self_id):
                continue
            target_name = str(getattr(seg, "name", "") or "").strip()
            return target_id, target_name

        return None, ""

    def _extract_all_mention_targets(self, event: AstrMessageEvent) -> list[tuple[str, str]]:
        if not hasattr(event, "message_obj") or not hasattr(event.message_obj, "message"):
            return []

        self_id = str(event.get_self_id() or "")
        results: list[tuple[str, str]] = []
        for seg in event.message_obj.message or []:
            if not isinstance(seg, Comp.At):
                continue
            target_id = str(getattr(seg, "qq", "") or "").strip()
            if not target_id or target_id.lower() == "all" or (self_id and target_id == self_id):
                continue
            target_name = str(getattr(seg, "name", "") or "").strip()
            results.append((target_id, target_name))
        return results

    def _resolve_target_user(self, event: AstrMessageEvent, user_ref: str = "") -> tuple[Optional[str], str]:
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

        now_ts = time.time()
        cached = self._get_cache(user_id, now_ts)
        if cached is not None:
            return cached

        blocked = await self._is_user_blacklisted_now(user_id)
        self._set_cache(user_id, blocked, now_ts)
        return blocked

    async def intercept_event(self, event: AstrMessageEvent) -> bool:
        sender_id = str(event.get_sender_id() or "")
        if not sender_id:
            return False

        blocked = await self.is_user_blacklisted(sender_id)
        if not blocked:
            return False

        logger.debug(f"[LLMEnhancement] 用户 {sender_id} 在黑名单中，已拦截消息。")
        event.stop_event()
        return True

    async def intercept_llm_request(self, event: AstrMessageEvent) -> bool:
        sender_id = str(event.get_sender_id() or "")
        if not sender_id:
            return False

        blocked = await self.is_user_blacklisted(sender_id)
        if not blocked:
            return False

        logger.debug(f"[LLMEnhancement] 用户 {sender_id} 在黑名单中，已拦截 LLM 请求。")
        event.stop_event()
        return True

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

    async def command_ls(self, page: int = 1, page_size: int = 10) -> BlacklistCommandResult:
        await self._cleanup_expired_on_query()

        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 10), 50))
        total_count = await self._db.get_blacklist_count()
        if total_count == 0:
            return BlacklistCommandResult(text="黑名单为空。")

        total_pages = max(1, (total_count + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages

        users = await self._db.get_blacklist_users(page, page_size)

        text_headers = [
            self._pad_for_table("序号", 4),
            self._pad_for_table("用户ID", 14),
            self._pad_for_table("用户名", 12),
            self._pad_for_table("加入时间", 11),
            self._pad_for_table("过期时间", 11),
            self._pad_for_table("原因", 20),
        ]
        text_header_row = " | ".join(text_headers)
        text_lines = ["黑名单列表", "=" * len(text_header_row), text_header_row, "-" * len(text_header_row)]

        table_headers = ["序号", "用户ID", "用户名", "加入时间", "过期时间", "原因"]
        table_rows: list[list[str]] = []

        for idx, user in enumerate(users, start=1 + (page - 1) * page_size):
            user_id, user_name, ban_time, expire_time, reason = user
            text_cells = [
                self._pad_for_table(str(idx), 4),
                self._pad_for_table(str(user_id or ""), 14),
                self._pad_for_table(str(user_name or "未知"), 12),
                self._pad_for_table(self._format_datetime_compact(ban_time), 11),
                self._pad_for_table(self._format_datetime_compact(expire_time, check_expire=True), 11),
                self._pad_for_table(str(reason or "无"), 20),
            ]
            text_lines.append(" | ".join(text_cells))

            table_rows.append([
                str(idx),
                str(user_id or ""),
                str(user_name or "未知"),
                self._format_datetime_compact(ban_time),
                self._format_datetime_compact(expire_time, check_expire=True),
                str(reason or "无"),
            ])

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

    async def command_info(self, event: AstrMessageEvent, user_ref: str = "") -> BlacklistCommandResult:
        await self._cleanup_expired_on_query()

        target_id, _ = self._resolve_target_user(event, user_ref)
        if not target_id:
            return BlacklistCommandResult(text="请提供用户 ID 或 @目标用户。")

        user = await self._db.get_user_info(target_id)
        if not user:
            return BlacklistCommandResult(text=f"用户 {target_id} 不在黑名单中。")

        _uid, user_name, ban_time, expire_time, reason = user
        text_lines = [
            f"用户 {target_id} 的黑名单信息",
            "=" * 36,
            f"用户名: {user_name or '未知'}",
            f"加入时间: {self._format_datetime(ban_time)}",
            f"过期时间: {self._format_datetime(expire_time, show_remaining=True, check_expire=True)}",
            f"原因: {reason or '无'}",
        ]
        text = "\n".join(text_lines)
        if self._should_render_image():
            expire_str = self._format_datetime(expire_time, show_remaining=True, check_expire=True)
            image_base64 = await table_to_image_base64(
                title=f"用户 {target_id}",
                headers=["字段", "值"],
                rows=[
                    ["用户名", user_name or "未知"],
                    ["加入时间", self._format_datetime(ban_time)],
                    ["过期时间", expire_str],
                    ["原因", reason or "无"],
                ],
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
        # 部分平台指令解析不会把 @ 写入 message_str，这里将被错位的参数纠正回来
        if mention_id and user_ref_text.isdigit() and duration_text and not duration_text.isdigit():
            reason = duration_text
            duration = user_ref_text
            user_ref = ""

        duration_sec, err = self._parse_duration_seconds(duration)
        if err:
            return err

        mentions = self._extract_all_mention_targets(event)
        if len(mentions) > 1:
            # 多个 @ → 批量模式
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

        success_ids: list[str] = []
        fail_messages: list[str] = []
        for tid, tname in zip(ids, names):
            if (not self.allow_blacklist_bot_admin()) and self._is_target_bot_admin(event, tid):
                fail_messages.append(f"用户 {tid} 是 Bot 管理员")
                continue

            ok = await self._db.add_user(
                user_id=tid,
                user_name=tname,
                ban_time=ban_time,
                expire_time=expire_time,
                reason=reason or "",
            )
            if not ok:
                fail_messages.append(f"用户 {tid} 写入失败")
                continue

            self._invalidate_cache(tid)
            success_ids.append(tid)

        parts: list[str] = []
        if success_ids:
            if len(success_ids) > 1:
                part = f"已拉黑 {len(success_ids)} 人"
            else:
                if duration_sec and duration_sec > 0:
                    part = f"用户 {success_ids[0]} 已加入黑名单，时长 {duration_sec} 秒"
                else:
                    part = f"用户 {success_ids[0]} 已永久加入黑名单"
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

        success_ids: list[str] = []
        not_found_ids: list[str] = []
        fail_messages: list[str] = []
        for tid in ids:
            user = await self._db.get_user_info(tid)
            if not user:
                not_found_ids.append(tid)
                continue

            ok = await self._db.remove_user(tid)
            if not ok:
                fail_messages.append(f"用户 {tid} 删除失败")
                continue
            self._invalidate_cache(tid)
            success_ids.append(tid)

        parts: list[str] = []
        if success_ids:
            part = f"已解除拉黑 {len(success_ids)} 人" if len(success_ids) > 1 else f"用户 {success_ids[0]} 已解除拉黑"
            parts.append(part)
        if not_found_ids:
            parts.append(f"不在黑名单中：{'，'.join(not_found_ids)}")
        if fail_messages:
            parts.append("；".join(fail_messages))
        return "，".join(parts) + "。" if parts else "操作失败。"

    async def command_clear(self) -> str:
        await self._cleanup_expired_on_query()

        count = await self._db.get_blacklist_count()
        if count == 0:
            return "黑名单已经为空。"

        ok = await self._db.clear_blacklist()
        if not ok:
            return "清空黑名单时出错。"
        self._clear_cache()
        return f"黑名单已清空，共移除 {count} 个用户。"

    async def tool_block_user(
        self,
        event: AstrMessageEvent,
        user_ids: str = "",
        user_name: str = "",
        duration: int = 0,
        reason: str = "",
    ) -> str:
        await self._cleanup_expired_on_query()

        user_id_list = _parse_user_id_list(user_ids)
        if not user_id_list:
            return json.dumps(
                {"success": False, "message": "请提供要拉黑的目标用户 ID。"},
                ensure_ascii=False,
            )

        sender_id = str(event.get_sender_id() or "")

        # 权限检查：非管理员只能操作自己
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

        # 解析 duration
        parsed_duration, err = self._parse_duration_seconds(duration)
        if err:
            return json.dumps(
                {"success": False, "message": err},
                ensure_ascii=False,
            )

        actual_duration = parsed_duration or 0
        max_duration = self.max_blacklist_duration()
        if actual_duration == 0 and max_duration > 0:
            actual_duration = max_duration
        if max_duration > 0 and actual_duration > max_duration:
            actual_duration = max_duration

        ban_time = datetime.now().isoformat()
        expire_time = None
        if actual_duration > 0:
            expire_time = (datetime.now() + timedelta(seconds=actual_duration)).isoformat()

        target_name = str(user_name or "").strip()
        if not target_name:
            target_name = str(event.get_sender_name() or "")

        results = []
        success_count = 0
        fail_count = 0

        for target_user_id in user_id_list:
            target_user_id = str(target_user_id or "").strip()
            if not target_user_id:
                fail_count += 1
                results.append({"user_id": target_user_id, "success": False, "error": "empty user_id"})
                continue

            is_self_defense = target_user_id == sender_id

            # Bot 管理员检查
            if (not self.allow_blacklist_bot_admin()) and self._is_target_bot_admin(event, target_user_id):
                fail_count += 1
                results.append({"user_id": target_user_id, "success": False, "error": BOT_ADMIN_BLOCK_MESSAGE})
                continue

            # 检查是否已在黑名单
            if await self._db.get_user_info(target_user_id):
                results.append({
                    "success": True,
                    "message": f"用户 {target_user_id} 已在黑名单中，无需重复添加。",
                    "user_id": target_user_id,
                    "wording_hint": BLACKLIST_WORDING_HINT,
                })
                success_count += 1
                continue

            # 写入数据库
            current_target_name = target_name if target_user_id != sender_id else (str(event.get_sender_name() or "") or target_name)
            ok = await self._db.add_user(
                user_id=target_user_id,
                user_name=current_target_name,
                ban_time=ban_time,
                expire_time=expire_time,
                reason=reason or "",
            )
            if not ok:
                fail_count += 1
                results.append({"user_id": target_user_id, "success": False, "error": "数据库写入异常。"})
                continue

            self._invalidate_cache(target_user_id)
            logger.info(f"[LLMEnhancement] 用户 {target_user_id} 已由 {sender_id} 通过 LLM 工具拉黑。")
            results.append({
                "success": True,
                "message": f"用户 {target_user_id} 已拉黑。",
                "user_id": target_user_id,
                "user_name": current_target_name,
                "duration": actual_duration if actual_duration > 0 else "永久",
                "reason": reason,
                "hint": "操作已生效。" if not is_self_defense else "操作已生效，将来这段时间内对方向你发送的消息将被屏蔽。",
                "wording_hint": BLACKLIST_WORDING_HINT,
            })
            success_count += 1

        if len(user_id_list) == 1:
            if results and results[0].get("success"):
                return json.dumps(results[0], ensure_ascii=False)
            return json.dumps(results[0] if results else {"success": False, "message": "操作失败。"}, ensure_ascii=False)

        return json.dumps({
            "success_count": success_count,
            "fail_count": fail_count,
            "results": results
        }, ensure_ascii=False, indent=2)

    async def tool_unblock_user(self, event: AstrMessageEvent, user_ids: str) -> str:
        await self._cleanup_expired_on_query()

        user_id_list = _parse_user_id_list(user_ids)
        if not user_id_list:
            return json.dumps(
                {"success": False, "message": "请提供要解除拉黑的用户 ID。"},
                ensure_ascii=False,
            )

        sender_id = str(event.get_sender_id() or "")

        # 权限检查：非管理员只能操作自己
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
                results.append({"user_id": target_user_id, "success": False, "error": "empty user_id"})
                continue

            user = await self._db.get_user_info(target_user_id)
            if not user:
                results.append({
                    "success": True,
                    "message": f"用户 {target_user_id} 不在黑名单中。",
                    "user_id": target_user_id,
                    "user_name": "",
                    "wording_hint": BLACKLIST_WORDING_HINT,
                })
                success_count += 1
                continue

            _uid, user_name, _ban_time, _expire_time, _reason = user
            ok = await self._db.remove_user(target_user_id)
            if not ok:
                fail_count += 1
                results.append({"user_id": target_user_id, "success": False, "error": "数据库删除失败。"})
                continue

            self._invalidate_cache(target_user_id)
            logger.info(f"[LLMEnhancement] 用户 {target_user_id} 已由 {sender_id} 通过 LLM 工具解除拉黑。")
            results.append({
                "success": True,
                "message": f"用户 {target_user_id} 已解除拉黑。",
                "user_id": target_user_id,
                "user_name": user_name or "",
                "wording_hint": BLACKLIST_WORDING_HINT,
            })
            success_count += 1

        if len(user_id_list) == 1:
            if results and results[0].get("success"):
                return json.dumps(results[0], ensure_ascii=False)
            return json.dumps(results[0] if results else {"success": False, "message": "操作失败。"}, ensure_ascii=False)

        return json.dumps({
            "success_count": success_count,
            "fail_count": fail_count,
            "results": results
        }, ensure_ascii=False, indent=2)

    async def tool_list_blacklist(self, event: AstrMessageEvent, page: int = 1, page_size: int = 20) -> str:
        await self._cleanup_expired_on_query()

        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 20), 50))
        total_count = await self._db.get_blacklist_count()
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

        users_data = await self._db.get_blacklist_users(page, page_size)
        users = []
        for user in users_data:
            user_id, user_name, ban_time, expire_time, reason = user
            users.append(
                {
                    "user_id": user_id,
                    "user_name": user_name or "",
                    "ban_time": ban_time,
                    "expire_time": expire_time if expire_time else "永久",
                    "reason": reason if reason else "无",
                }
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

    async def tool_get_blacklist_status(self, event: AstrMessageEvent, user_ids: str) -> str:
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
                results.append({"user_id": target_id, "is_blacklisted": False, "error": "empty user_id"})
                continue

            user_info = await self._db.get_user_info(target_id)
            if user_info:
                uid, user_name, ban_time, expire_time, reason = user_info
                results.append({
                    "is_blacklisted": True,
                    "user_id": uid,
                    "user_name": user_name or "",
                    "ban_time": ban_time,
                    "expire_time": expire_time if expire_time else "永久",
                    "reason": reason if reason else "无",
                    "expire_time_hint": "expire_time 表示黑名单失效时间，失效后意味着你将其移出黑名单。",
                    "wording_hint": BLACKLIST_WORDING_HINT,
                })
                success_count += 1
            else:
                results.append({
                    "is_blacklisted": False,
                    "user_id": target_id,
                    "wording_hint": BLACKLIST_WORDING_HINT,
                })
                success_count += 1

        if len(user_id_list) == 1:
            return json.dumps(results[0] if results else {"is_blacklisted": False, "message": "查询失败。"}, ensure_ascii=False)

        return json.dumps({
            "success_count": success_count,
            "fail_count": fail_count,
            "results": results
        }, ensure_ascii=False, indent=2)
