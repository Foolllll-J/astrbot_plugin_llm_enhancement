import os
import re
from typing import Any, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

try:
    import chardet
except ImportError:
    chardet = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

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

PDF_FILE_EXTENSION = ".pdf"
PDF_PARSE_MAX_PAGES = 8
PDF_PARSE_MAX_CHARS = 20000
FILE_INJECT_MAX_SIZE_MB_DEFAULT = 20


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


def _file_ext(name: str) -> str:
    return os.path.splitext(str(name or "").lower())[1]


def _is_text_file_name(file_name: str) -> bool:
    return _file_ext(file_name) in TEXT_FILE_EXTENSIONS


def _is_pdf_file_name(file_name: str) -> bool:
    return _file_ext(file_name) == PDF_FILE_EXTENSION


def _looks_like_pdf_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


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
    if chardet is not None:
        try:
            detection = chardet.detect(raw)
            detected_enc = str((detection or {}).get("encoding") or "").strip()
            detected_confidence = float((detection or {}).get("confidence") or 0.0)
        except Exception:
            pass

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

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        _push_encoding("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        _push_encoding("utf-8-sig")

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


def _normalize_pdf_text(raw_text: str) -> str:
    text = str(raw_text or "").replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf_excerpt(path: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if PdfReader is None:
        logger.debug("[LLMEnhancement] PDF 文本注入已启用，但未安装 pypdf，跳过 PDF 解析")
        return ""

    safe_max = min(max_chars, PDF_PARSE_MAX_CHARS)
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.debug(f"[LLMEnhancement] 读取 PDF 失败: file={os.path.basename(path)}, err={e}")
        return ""

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as e:
            logger.debug(f"[LLMEnhancement] PDF 已加密且解密失败: file={os.path.basename(path)}, err={e}")
            return ""

    parts: list[str] = []
    total_len = 0
    for idx, page in enumerate(getattr(reader, "pages", []) or []):
        if idx >= PDF_PARSE_MAX_PAGES:
            break
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""

        cleaned = _normalize_pdf_text(extracted)
        if not cleaned:
            continue

        parts.append(cleaned)
        total_len += len(cleaned)
        if total_len >= safe_max * 2:
            break

    text = "\n\n".join(parts).strip()
    if not text:
        return ""
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
    max_file_size_mb: int = FILE_INJECT_MAX_SIZE_MB_DEFAULT,
    cleanup_paths: Optional[list[str]] = None,
    failure_details: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """从消息链提取可读文件内容，返回 [(文件名, 摘录内容)]。"""
    if not isinstance(chain, list):
        return []

    try:
        max_chars = max(0, int(max_chars))
    except (TypeError, ValueError):
        max_chars = 0
    try:
        max_file_size_mb = max(0, int(max_file_size_mb))
    except (TypeError, ValueError):
        max_file_size_mb = FILE_INJECT_MAX_SIZE_MB_DEFAULT

    if max_chars <= 0:
        return []
    max_file_size_bytes = max_file_size_mb * 1024 * 1024 if max_file_size_mb > 0 else 0

    if failure_details is None:
        failure_details = []
    if not isinstance(failure_details, list):
        failure_details = []

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    file_seg_count = 0
    skip_non_text = 0
    skip_path_missing = 0
    skip_too_large = 0
    skip_empty_excerpt = 0
    skip_duplicate = 0
    error_count = 0

    def _record_failure(file_label: str, reason: str) -> None:
        if not reason:
            return
        reason_text = str(reason).strip()
        if not reason_text:
            return
        if not file_label:
            file_label = "unknown"
        failure = f"{file_label}: {reason_text}"
        if failure not in failure_details:
            failure_details.append(failure)

    for seg in chain:
        file_name = ""
        local_path = ""
        download_path = ""
        seg_file_name = ""

        try:
            if isinstance(seg, Comp.File):
                file_seg_count += 1
                file_name = str(getattr(seg, "name", "") or "")
                seg_file_name = file_name
                if file_name and (not _is_text_file_name(file_name)) and (not _is_pdf_file_name(file_name)):
                    skip_non_text += 1
                    _record_failure(file_name, "文件类型不支持文本摘要解析")
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
                    _record_failure(seg_file_name, "文件段格式异常，无法解析")
                    continue

                file_name = str(
                    data.get("name") or data.get("file_name") or data.get("file") or ""
                ).strip()
                seg_file_name = file_name
                if file_name and (not _is_text_file_name(file_name)) and (not _is_pdf_file_name(file_name)):
                    skip_non_text += 1
                    _record_failure(file_name, "文件类型不支持文本摘要解析")
                    continue

                local_path = _normalize_local_path(str(data.get("file") or data.get("path") or ""))
                if not local_path:
                    file_url = str(data.get("url") or "").strip()
                    if not file_url:
                        file_id = str(data.get("file_id") or "").strip()
                        if file_id:
                            file_url = await _resolve_file_url_by_id(event, file_id)

                    if file_url:
                        file_comp = Comp.File(name=file_name or "file.bin", url=file_url)
                        download_path = await file_comp.get_file()
                        local_path = _normalize_local_path(download_path)
                    else:
                        skip_path_missing += 1
                        _record_failure(file_name, "未找到可下载链接或本地路径")
                        continue
            else:
                continue

            if not local_path or not os.path.isfile(local_path):
                skip_path_missing += 1
                _record_failure(file_name or seg_file_name, "文件下载失败或本地文件不存在")
                continue

            effective_name = str(file_name or os.path.basename(local_path) or "")
            if max_file_size_bytes > 0:
                try:
                    file_size = os.path.getsize(local_path)
                except OSError:
                    file_size = -1
                too_large = file_size > max_file_size_bytes
                if too_large:
                    skip_too_large += 1
                    _record_failure(
                        effective_name,
                        f"文件过大（{file_size} 字节，超过 {max_file_size_mb}MB 上限）",
                    )
                    logger.debug(
                        "[LLMEnhancement] 文件超过注入大小上限，跳过文本提取: "
                        f"file={effective_name or os.path.basename(local_path)}, "
                        f"size_bytes={file_size}, limit_mb={max_file_size_mb}"
                    )
                    continue

            is_pdf = _is_pdf_file_name(effective_name) or _looks_like_pdf_file(local_path)

            if is_pdf:
                excerpt = _read_pdf_excerpt(local_path, max_chars)
            elif _is_text_file_name(effective_name):
                excerpt = _read_text_excerpt(local_path, max_chars)
            else:
                skip_non_text += 1
                _record_failure(effective_name, "文件类型不支持文本摘要解析")
                continue

            if not excerpt:
                skip_empty_excerpt += 1
                _record_failure(effective_name, "未提取到可用文本摘要")
                continue

            key = f"{effective_name}::{excerpt}"
            if key in seen:
                skip_duplicate += 1
                continue
            seen.add(key)
            results.append((effective_name or os.path.basename(local_path), excerpt))

            if cleanup_paths is not None and download_path and os.path.exists(download_path):
                if download_path not in cleanup_paths:
                    cleanup_paths.append(download_path)
        except Exception as e:
            error_count += 1
            logger.debug(
                "[LLMEnhancement] 文件处理异常: file=%s, err=%r",
                file_name or seg_file_name or "unknown",
                e,
            )
            _record_failure(file_name or seg_file_name or "unknown", "处理文件失败")
            continue

    if file_seg_count > 0:
        logger.debug(
            "[LLMEnhancement] 文件文本提取结果: "
            f"segments={file_seg_count}, success={len(results)}, "
            f"skip_non_text={skip_non_text}, skip_path_missing={skip_path_missing}, "
            f"skip_too_large={skip_too_large}, skip_empty_excerpt={skip_empty_excerpt}, "
            f"skip_duplicate={skip_duplicate}, errors={error_count}, max_chars={max_chars}, "
            f"max_file_size_mb={max_file_size_mb}"
        )

    return results
