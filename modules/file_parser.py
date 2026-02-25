import os
from typing import Any, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

try:
    import chardet
except ImportError:
    chardet = None

TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".log",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".conf",
    ".cfg",
    ".toml",
    ".py",
    ".js",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".sh",
    ".bash",
    ".html",
    ".htm",
    ".css",
    ".jsx",
    ".tsx",
    ".ts",
    ".vue",
    ".sql",
    ".csv",
    ".properties",
    ".env",
}


def _normalize_local_path(path_or_url: str) -> str:
    candidate = str(path_or_url or "").strip()
    if not candidate:
        return ""
    if candidate.startswith("file://"):
        candidate = candidate[7:]
        if candidate.startswith("/") and len(candidate) > 2 and candidate[2] == ":":
            candidate = candidate[1:]
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return ""


def _is_text_file_name(file_name: str) -> bool:
    ext = os.path.splitext(str(file_name or "").lower())[1]
    return ext in TEXT_FILE_EXTENSIONS


def _read_text_excerpt(path: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    safe_max = min(max_chars, 5000)
    max_bytes = min(max(4096, safe_max * 8), 256 * 1024)

    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes)
    except Exception:
        return ""

    candidate_encodings: list[str] = []
    seen_encodings: set[str] = set()

    def _push_encoding(enc: str) -> None:
        enc_norm = str(enc or "").strip().lower()
        if not enc_norm or enc_norm in seen_encodings:
            return
        seen_encodings.add(enc_norm)
        candidate_encodings.append(enc_norm)

    detected_enc = ""
    detected_confidence = 0.0
    # 1) 参考 file_checker：优先使用 chardet 探测编码
    if chardet is not None:
        try:
            detection = chardet.detect(raw)
            detected_enc = str((detection or {}).get("encoding") or "").strip()
            detected_confidence = float((detection or {}).get("confidence") or 0.0)
        except Exception:
            pass

    # 高置信度时直接优先按探测编码解码（允许忽略个别脏字节，避免误判导致整段乱码）
    if detected_enc and detected_confidence >= 0.8:
        try:
            text = raw.decode(detected_enc, errors="ignore").replace("\x00", "").strip()
            if text:
                logger.debug(
                    "[LLMEnhancement] 文件摘录解码结果: "
                    f"file={os.path.basename(path)}, selected={detected_enc}(chardet-first), "
                    f"chardet={detected_enc}({detected_confidence:.2f})"
                )
                if len(text) > safe_max:
                    return text[: max(1, safe_max - 3)] + "..."
                return text
        except Exception:
            pass

    # 2) BOM 作为次级线索，不再强优先
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        _push_encoding("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        _push_encoding("utf-8-sig")

    # 3) 常规回退编码（不使用 latin-1，避免无意义“成功解码”导致乱码注入）
    if detected_enc:
        _push_encoding(detected_enc)
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16", "utf-16-le", "utf-16-be"):
        _push_encoding(enc)

    text = ""
    selected_encoding = ""
    selected_mode = ""
    for enc in candidate_encodings:
        try:
            text = raw.decode(enc, errors="strict")
        except Exception:
            continue
        text = text.replace("\x00", "").strip()
        if text:
            selected_encoding = enc
            selected_mode = "strict"
            break

    if not text:
        # 最后一次温和回退，避免因为个别脏字节导致整段丢失
        for enc in ("utf-8", "gb18030", "gbk"):
            try:
                text = raw.decode(enc, errors="ignore").replace("\x00", "").strip()
                if text:
                    selected_encoding = enc
                    selected_mode = "ignore"
                    break
            except Exception:
                continue

    if not text:
        logger.debug(
            "[LLMEnhancement] 文件摘录解码失败: "
            f"file={os.path.basename(path)}, bytes={len(raw)}, "
            f"chardet={detected_enc or 'none'}({detected_confidence:.2f})"
        )
        return ""

    logger.debug(
        "[LLMEnhancement] 文件摘录解码结果: "
        f"file={os.path.basename(path)}, selected={selected_encoding or 'unknown'}({selected_mode or 'unknown'}), "
        f"chardet={detected_enc or 'none'}({detected_confidence:.2f})"
    )

    if len(text) > safe_max:
        return text[: max(1, safe_max - 3)] + "..."
    return text


async def _resolve_file_url_by_id(event: AstrMessageEvent, file_id: str) -> str:
    client = getattr(event, "bot", None)
    api = getattr(client, "api", None) if client else None
    if api is None or not file_id:
        return ""

    gid = event.get_group_id()
    actions = []
    if gid:
        try:
            actions.append(("get_group_file_url", {"group_id": int(gid), "file_id": file_id}))
        except Exception:
            actions.append(("get_group_file_url", {"group_id": gid, "file_id": file_id}))
    else:
        actions.append(("get_private_file_url", {"file_id": file_id}))

    for action, params in actions:
        try:
            ret = await api.call_action(action, **params)
            data = ret.get("data", ret) if isinstance(ret, dict) else {}
            if not isinstance(data, dict):
                continue
            url = str(data.get("url") or "").strip()
            if url:
                return url
            file_value = str(data.get("file") or "").strip()
            if file_value:
                return file_value
        except Exception:
            continue
    return ""


async def extract_file_infos_from_chain(
    event: AstrMessageEvent,
    chain: list[Any],
    max_chars: int,
    cleanup_paths: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """从消息链提取可读取文本文件内容，返回 [(文件名, 摘录内容)]。"""
    if max_chars <= 0 or not isinstance(chain, list):
        return []

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    file_seg_count = 0
    skip_non_text = 0
    skip_path_missing = 0
    skip_empty_excerpt = 0
    skip_duplicate = 0
    error_count = 0

    for seg in chain:
        file_name = ""
        local_path = ""
        download_path = ""

        try:
            if isinstance(seg, Comp.File):
                file_seg_count += 1
                file_name = str(getattr(seg, "name", "") or "")
                if not _is_text_file_name(file_name):
                    skip_non_text += 1
                    continue
                download_path = await seg.get_file()
                local_path = _normalize_local_path(download_path)
            elif isinstance(seg, dict):
                if str(seg.get("type") or "").lower() != "file":
                    continue
                file_seg_count += 1
                data = seg.get("data") or {}
                if not isinstance(data, dict):
                    error_count += 1
                    continue

                file_name = str(
                    data.get("name") or data.get("file_name") or data.get("file") or ""
                ).strip()
                if not _is_text_file_name(file_name):
                    skip_non_text += 1
                    continue

                local_path = _normalize_local_path(str(data.get("file") or data.get("path") or ""))
                if not local_path:
                    file_url = str(data.get("url") or "").strip()
                    if not file_url:
                        file_id = str(data.get("file_id") or "").strip()
                        if file_id:
                            file_url = await _resolve_file_url_by_id(event, file_id)

                    if file_url:
                        file_comp = Comp.File(name=file_name or "file.txt", url=file_url)
                        download_path = await file_comp.get_file()
                        local_path = _normalize_local_path(download_path)
            else:
                continue

            if not local_path or not os.path.isfile(local_path):
                skip_path_missing += 1
                continue

            excerpt = _read_text_excerpt(local_path, max_chars)
            if not excerpt:
                skip_empty_excerpt += 1
                continue

            key = f"{file_name}::{excerpt}"
            if key in seen:
                skip_duplicate += 1
                continue
            seen.add(key)
            results.append((file_name or os.path.basename(local_path), excerpt))

            if cleanup_paths is not None and download_path and os.path.exists(download_path):
                if download_path not in cleanup_paths:
                    cleanup_paths.append(download_path)
        except Exception:
            error_count += 1
            continue

    if file_seg_count > 0:
        logger.debug(
            "[LLMEnhancement] 文件文本提取结果: "
            f"segments={file_seg_count}, success={len(results)}, "
            f"skip_non_text={skip_non_text}, skip_path_missing={skip_path_missing}, "
            f"skip_empty_excerpt={skip_empty_excerpt}, skip_duplicate={skip_duplicate}, "
            f"errors={error_count}, max_chars={max_chars}"
        )

    return results
