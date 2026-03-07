import asyncio
import ipaddress
import re
import socket
import time
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

import aiohttp

URL_REGEX = re.compile(r"https?://[^\s<>'\"`]+", flags=re.IGNORECASE)
TRAILING_PUNCTUATION = ".,!?;:，。！？；：）】》」’”"
DOWNLOAD_EXTENSIONS = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".zst",
    ".exe",
    ".msi",
    ".apk",
    ".ipa",
    ".dmg",
    ".iso",
    ".img",
    ".bin",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".csv",
    ".torrent",
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wmv",
    ".webm",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
}
DOWNLOAD_QUERY_HINTS = {
    "download",
    "attachment",
    "filename",
    "file",
    "file_name",
    "response-content-disposition",
    "attname",
    "dl",
}
DEFAULT_TIMEOUT_SEC = 8
DEFAULT_MAX_DOWNLOAD_KB = 512
DEFAULT_CACHE_TTL_SEC = 600
MAX_URL_INJECT_COUNT = 3
MAX_REDIRECTS = 5
MAX_INJECT_CHARS_PER_URL = 600
URL_CACHE_MAX_SIZE = 256

_url_inject_cache: Dict[str, Dict[str, Any]] = {}
_url_inject_cache_lock = asyncio.Lock()


@dataclass
class UrlInjectResult:
    injected: bool = False
    details: List[str] = field(default_factory=list)



def _normalize_text(text: str, limit: int) -> str:
    value = str(text or "").replace("\x00", "")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(1, limit - 3)] + "..."


def _preview_text(text: str, limit: int = 160) -> str:
    value = str(text or "").replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 3)] + "..."


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return _normalize_text(unescape(text), MAX_INJECT_CHARS_PER_URL)


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _normalize_text(unescape(match.group(1)), 120)


def _decode_response_body(body: bytes, content_type: str) -> str:
    if not body:
        return ""
    content_type = str(content_type or "")
    charset = ""
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        charset = match.group(1).strip().strip("\"'")
    for enc in [charset, "utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"]:
        if not enc:
            continue
        try:
            return body.decode(enc, errors="replace")
        except Exception:
            continue
    return body.decode("utf-8", errors="replace")


def _clean_candidate_url(url: str) -> str:
    value = str(url or "").strip()
    while value and value[-1] in TRAILING_PUNCTUATION:
        value = value[:-1]
    return value.strip()


def _extract_urls_from_text(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    urls: List[str] = []
    seen = set()
    for m in URL_REGEX.finditer(text):
        u = _clean_candidate_url(m.group(0))
        if not u or u in seen:
            continue
        seen.add(u)
        urls.append(u)
    return urls


def _extract_urls_from_chain(event: AstrMessageEvent, chain: List[Any]) -> List[str]:
    text_parts: List[str] = []
    if isinstance(getattr(event, "message_str", None), str):
        text_parts.append(event.message_str)

    plain_type = getattr(Comp, "Plain", None)
    for seg in chain:
        if plain_type is not None and isinstance(seg, plain_type):
            text_parts.append(str(getattr(seg, "text", "") or ""))
            continue
        if isinstance(seg, dict):
            if str(seg.get("type") or "").lower() != "text":
                continue
            data = seg.get("data") or {}
            if isinstance(data, dict):
                text_parts.append(str(data.get("text") or ""))
            continue
        text_value = getattr(seg, "text", None)
        if isinstance(text_value, str):
            text_parts.append(text_value)

    seen = set()
    urls: List[str] = []
    for text in text_parts:
        for url in _extract_urls_from_text(text):
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) >= MAX_URL_INJECT_COUNT:
                return urls
    return urls


def _normalize_domain(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if "://" in raw:
        raw = urlparse(raw).netloc.lower()
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    return raw.strip(".")


def _is_domain_blocked(host: str, blocked_domains: List[str]) -> bool:
    normalized_host = _normalize_domain(host)
    if not normalized_host:
        return False
    for item in blocked_domains:
        domain = _normalize_domain(item)
        if not domain:
            continue
        if normalized_host == domain or normalized_host.endswith("." + domain):
            return True
    return False


def _is_private_ip(ip_text: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_text)
    except Exception:
        return False
    if ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified:
        return True
    return ip_obj.is_private


def _is_private_host_literal(host: str) -> bool:
    host_l = str(host or "").strip().lower()
    if not host_l:
        return True
    if host_l in {"localhost", "localhost.localdomain"}:
        return True
    if host_l.endswith(".localhost"):
        return True
    return _is_private_ip(host_l)


def _resolve_host_ips(host: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return []
    ips: List[str] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip = str(sockaddr[0] or "")
        if ip and ip not in ips:
            ips.append(ip)
    return ips


async def _is_private_network_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return True
    host = str(parsed.hostname or "").strip()
    if not host:
        return True
    if _is_private_host_literal(host):
        return True
    ips = await asyncio.to_thread(_resolve_host_ips, host)
    for ip in ips:
        if _is_private_ip(ip):
            return True
    return False


def _looks_like_download_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    path = str(parsed.path or "").lower()
    for ext in DOWNLOAD_EXTENSIONS:
        if path.endswith(ext):
            return True
    query_map = parse_qs(parsed.query or "", keep_blank_values=True)
    keys = {str(k).strip().lower() for k in query_map.keys()}
    if keys & DOWNLOAD_QUERY_HINTS:
        return True
    return False


def _is_html_like_content_type(content_type: str) -> bool:
    ct = str(content_type or "").split(";", 1)[0].strip().lower()
    if not ct:
        return True
    if ct in {"text/html", "application/xhtml+xml", "text/plain", "application/json", "text/markdown"}:
        return True
    if ct.startswith("text/"):
        return True
    return False


def _is_download_response(headers: Dict[str, str], content_type: str) -> bool:
    disposition = str(headers.get("content-disposition") or "").lower()
    if "attachment" in disposition or "filename=" in disposition:
        return True
    if not _is_html_like_content_type(content_type):
        return True
    return False


async def _get_cached_summary(cache_key: str) -> Optional[str]:
    now_ts = time.time()
    async with _url_inject_cache_lock:
        cached = _url_inject_cache.get(cache_key)
        if not cached:
            return None
        expire_ts = float(cached.get("expire", 0.0) or 0.0)
        if expire_ts <= now_ts:
            _url_inject_cache.pop(cache_key, None)
            return None
        return str(cached.get("summary") or "").strip() or None


async def _set_cached_summary(cache_key: str, summary: str, ttl_sec: int) -> None:
    text = str(summary or "").strip()
    if not text:
        return
    ttl_sec = max(1, int(ttl_sec))
    now_ts = time.time()
    async with _url_inject_cache_lock:
        expired_keys = [key for key, val in _url_inject_cache.items() if float(val.get("expire", 0.0) or 0.0) <= now_ts]
        for key in expired_keys:
            _url_inject_cache.pop(key, None)
        if len(_url_inject_cache) >= URL_CACHE_MAX_SIZE:
            _url_inject_cache.clear()
        _url_inject_cache[cache_key] = {"expire": now_ts + ttl_sec, "summary": text}


async def _fetch_by_aiohttp(
    url: str,
    timeout_sec: int,
    max_bytes: int,
    should_block_private_network: bool,
    blocked_domains: List[str],
) -> Optional[Dict[str, Any]]:
    headers = {
        "User-Agent": "AstrBot-LLMEnhancement/1.0",
        "Accept": "text/html,application/xhtml+xml,text/plain,application/json,*/*;q=0.8",
    }
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            current = url
            for _ in range(MAX_REDIRECTS + 1):
                parsed = urlparse(current)
                host = str(parsed.hostname or "").strip().lower()
                if _is_domain_blocked(host, blocked_domains):
                    return {"blocked": True, "reason": "blocked_domain", "url": current}
                if should_block_private_network and await _is_private_network_url(current):
                    return {"blocked": True, "reason": "private_network", "url": current}

                async with session.get(current, timeout=timeout_sec, allow_redirects=False) as resp:
                    status = int(resp.status)
                    resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
                    if 300 <= status < 400:
                        location = str(resp_headers.get("location") or "").strip()
                        if not location:
                            return {"error": f"redirect_without_location(status={status})"}
                        current = urljoin(current, location)
                        continue

                    if not (200 <= status < 400):
                        return {"error": f"http_status_{status}"}

                    cl = resp_headers.get("content-length")
                    if cl and cl.isdigit() and int(cl) > max_bytes:
                        return {
                            "download_like": True,
                            "download_reason": "content_too_large",
                            "content_type": str(resp_headers.get("content-type") or ""),
                            "final_url": current,
                        }

                    body = await resp.content.read(max_bytes + 1)
                    truncated = len(body) > max_bytes
                    if truncated:
                        body = body[:max_bytes]
                    return {
                        "status": status,
                        "headers": resp_headers,
                        "content_type": str(resp_headers.get("content-type") or ""),
                        "body": body,
                        "truncated": truncated,
                        "final_url": current,
                    }
    except Exception as e:
        parsed = urlparse(str(url or ""))
        host = str(parsed.hostname or "").strip().lower()
        logger.debug(
            "[LLMEnhancement] URL 解析 aiohttp 抓取失败: "
            f"url={url}, host={host or 'unknown'}, timeout_sec={timeout_sec}, "
            f"exc_type={type(e).__name__}, err={_preview_text(repr(e), 240)}"
        )
        return None


async def _fetch_by_urllib(
    url: str,
    timeout_sec: int,
    max_bytes: int,
    should_block_private_network: bool,
    blocked_domains: List[str],
) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    def _do() -> Dict[str, Any]:
        current = url
        opener = urllib.request.build_opener(urllib.request.HTTPHandler, urllib.request.HTTPSHandler)
        opener.addheaders = [
            ("User-Agent", "AstrBot-LLMEnhancement/1.0"),
            ("Accept", "text/html,application/xhtml+xml,text/plain,application/json,*/*;q=0.8"),
        ]
        for _ in range(MAX_REDIRECTS + 1):
            try:
                req = urllib.request.Request(current, method="GET")
                with opener.open(req, timeout=timeout_sec) as resp:
                    final_url = str(resp.geturl() or current)
                    headers = {str(k).lower(): str(v) for k, v in dict(resp.headers).items()}
                    status = int(getattr(resp, "status", 200) or 200)
                    cl = headers.get("content-length")
                    if cl and cl.isdigit() and int(cl) > max_bytes:
                        return {
                            "download_like": True,
                            "download_reason": "content_too_large",
                            "content_type": str(headers.get("content-type") or ""),
                            "final_url": final_url,
                        }
                    body = resp.read(max_bytes + 1)
                    truncated = len(body) > max_bytes
                    if truncated:
                        body = body[:max_bytes]
                    return {
                        "status": status,
                        "headers": headers,
                        "content_type": str(headers.get("content-type") or ""),
                        "body": body,
                        "truncated": truncated,
                        "final_url": final_url,
                    }
            except urllib.error.HTTPError as e:
                status = int(getattr(e, "code", 0) or 0)
                if 300 <= status < 400:
                    location = str(e.headers.get("Location") or "").strip()
                    if location:
                        current = urljoin(current, location)
                        continue
                return {"error": f"http_status_{status}"}
            except Exception as e:
                return {"error": f"fetch_error:{e}"}
        return {"error": "too_many_redirects"}

    result = await asyncio.to_thread(_do)
    final_url = str(result.get("final_url") or "")
    if final_url:
        final_host = str(urlparse(final_url).hostname or "").strip().lower()
        if _is_domain_blocked(final_host, blocked_domains):
            return {"blocked": True, "reason": "blocked_domain", "url": final_url}
        if should_block_private_network and await _is_private_network_url(final_url):
            return {"blocked": True, "reason": "private_network", "url": final_url}
    return result


async def _fetch_url_summary(
    url: str,
    timeout_sec: int,
    max_bytes: int,
    should_block_private_network: bool,
    blocked_domains: List[str],
) -> str:
    parsed = urlparse(url)
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return f"链接 {url} 无法解析主机名，已跳过。"
    if _is_domain_blocked(host, blocked_domains):
        return f"链接 {url} 命中黑名单域名，已跳过解析。"
    if should_block_private_network and await _is_private_network_url(url):
        return f"链接 {url} 指向内网地址，当前权限不允许解析。"

    if _looks_like_download_url(url):
        return f"链接 {url} 看起来是下载链接，请先下载文件后再发送文件内容进行解析。"

    response = await _fetch_by_aiohttp(
        url=url,
        timeout_sec=timeout_sec,
        max_bytes=max_bytes,
        should_block_private_network=should_block_private_network,
        blocked_domains=blocked_domains,
    )
    if response is None:
        logger.debug(
            "[LLMEnhancement] URL 解析 aiohttp 抓取失败，回退 urllib: "
            f"url={url}, timeout_sec={timeout_sec}, max_bytes={max_bytes}"
        )
        response = await _fetch_by_urllib(
            url=url,
            timeout_sec=timeout_sec,
            max_bytes=max_bytes,
            should_block_private_network=should_block_private_network,
            blocked_domains=blocked_domains,
        )

    if bool(response.get("blocked")):
        reason = str(response.get("reason") or "")
        target = str(response.get("url") or url)
        if reason == "blocked_domain":
            return f"链接 {target} 命中黑名单域名，已跳过解析。"
        if reason == "private_network":
            return f"链接 {target} 指向内网地址，当前权限不允许解析。"
        return f"链接 {target} 因安全策略被拦截，已跳过解析。"

    if bool(response.get("download_like")):
        final_url = str(response.get("final_url") or url)
        reason = str(response.get("download_reason") or "non_html")
        content_type = str(response.get("content_type") or "").strip()
        reason_text = "返回下载内容"
        if reason == "content_too_large":
            reason_text = "内容过大"
        return (
            f"链接 {final_url} 为{reason_text}"
            + (f"（content-type: {content_type}）" if content_type else "")
            + "，请先下载后再发送文件内容进行解析。"
        )

    error = str(response.get("error") or "").strip()
    if error:
        return f"链接 {url} 抓取失败（{error}），已跳过解析。"

    headers = response.get("headers") or {}
    content_type = str(response.get("content_type") or "")
    if _is_download_response(headers, content_type):
        final_url = str(response.get("final_url") or url)
        return (
            f"链接 {final_url} 返回非网页内容"
            + (f"（content-type: {content_type}）" if content_type else "")
            + "，请先下载后再发送文件内容进行解析。"
        )

    html_or_text = _decode_response_body(response.get("body") or b"", content_type)
    title = _extract_title(html_or_text)
    snippet = _strip_html(html_or_text)
    truncated = bool(response.get("truncated"))
    final_url = str(response.get("final_url") or url)

    if not snippet:
        return (
            f"链接 {final_url} 已访问，但未提取到有效正文。"
            + ("（已按下载上限截断）" if truncated else "")
        )

    suffix = "（已按下载上限截断）" if truncated else ""
    if title:
        return f"链接 {final_url} 的页面信息：标题《{title}》；正文摘要：{snippet}{suffix}"
    return f"链接 {final_url} 的页面摘要：{snippet}{suffix}"


async def extract_url_infos_from_chain(
    event: AstrMessageEvent,
    chain: List[Any],
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    max_download_kb: int = DEFAULT_MAX_DOWNLOAD_KB,
    block_private_network: bool = True,
    blocked_domains: Optional[List[str]] = None,
    cache_ttl_sec: int = DEFAULT_CACHE_TTL_SEC,
) -> UrlInjectResult:
    result = UrlInjectResult()
    urls = _extract_urls_from_chain(event, chain)
    if not urls:
        return result

    try:
        timeout_sec = max(2, int(timeout_sec))
    except Exception:
        timeout_sec = DEFAULT_TIMEOUT_SEC
    try:
        max_download_kb = max(32, int(max_download_kb))
    except Exception:
        max_download_kb = DEFAULT_MAX_DOWNLOAD_KB
    try:
        cache_ttl_sec = max(0, int(cache_ttl_sec))
    except Exception:
        cache_ttl_sec = DEFAULT_CACHE_TTL_SEC

    blocked_domains = [str(x or "").strip() for x in (blocked_domains or []) if str(x or "").strip()]
    should_block_private_network = bool(block_private_network) and (not bool(event.is_admin()))
    max_bytes = max_download_kb * 1024
    blocked_domains_sig = "|".join(sorted(_normalize_domain(x) for x in blocked_domains if _normalize_domain(x)))

    summaries: List[str] = []
    for url in urls[:MAX_URL_INJECT_COUNT]:
        cache_key = (
            f"{url}|{timeout_sec}|{max_download_kb}|"
            f"private={int(should_block_private_network)}|blocked={blocked_domains_sig}"
        )
        summary = None
        if cache_ttl_sec > 0:
            summary = await _get_cached_summary(cache_key)
        if not summary:
            summary = await _fetch_url_summary(
                url=url,
                timeout_sec=timeout_sec,
                max_bytes=max_bytes,
                should_block_private_network=should_block_private_network,
                blocked_domains=blocked_domains,
            )
            if cache_ttl_sec > 0 and summary:
                await _set_cached_summary(cache_key, summary, cache_ttl_sec)
        if summary:
            summaries.append(_normalize_text(summary, limit=MAX_INJECT_CHARS_PER_URL))

    if summaries:
        result.injected = True
        result.details = summaries
        logger.debug(
            "[LLMEnhancement] URL 注入完成: "
            f"url_count={len(urls[:MAX_URL_INJECT_COUNT])}, injected_count={len(summaries)}, "
            f"timeout_sec={timeout_sec}, max_download_kb={max_download_kb}, "
            f"block_private={bool(block_private_network)}, admin_bypass={bool(event.is_admin())}, "
            f"effective_block_private={should_block_private_network}"
        )
    return result
