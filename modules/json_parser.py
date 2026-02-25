import json
from typing import Any, Dict, List, Tuple


def _truncate_text(value: str, limit: int = 120) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "..."


def extract_json_key_info(inner_json: Dict[str, Any]) -> str:
    """从 JSON 卡片对象中提取可读的关键信息摘要。"""
    if not isinstance(inner_json, dict):
        return ""

    prompt = _truncate_text(inner_json.get("prompt", ""))
    app = _truncate_text(inner_json.get("app", ""), limit=64)
    desc = _truncate_text(inner_json.get("desc", ""))

    detail: Dict[str, Any] = {}
    meta = inner_json.get("meta")
    if isinstance(meta, dict):
        for val in meta.values():
            if isinstance(val, dict):
                detail = val
                break

    title = _truncate_text(detail.get("title", ""))
    detail_desc = _truncate_text(detail.get("desc", ""))
    url = _truncate_text(detail.get("qqdocurl") or detail.get("url") or "", limit=200)

    summary_text = prompt or detail_desc or desc
    parts: List[str] = []
    if title:
        parts.append(f"标题: {title}")
    if summary_text:
        parts.append(f"摘要: {summary_text}")
    if app:
        parts.append(f"应用: {app}")
    if url:
        parts.append(f"链接: {url}")
    return " | ".join(parts)


def parse_json_segment_data(raw_data: str) -> Tuple[List[str], str]:
    """
    解析 OneBot json 段 data 字符串。
    返回 (news 文本列表, 通用卡片关键信息)。
    """
    if not raw_data:
        return [], ""

    normalized = str(raw_data).replace("&#44;", ",")
    try:
        inner_json = json.loads(normalized)
    except Exception:
        return [], ""

    return _parse_json_data(inner_json)


def extract_json_infos_from_chain(chain: List[Any]) -> Tuple[List[str], List[str]]:
    """从消息链提取 JSON 信息，返回 (news_texts, key_infos)。"""
    news_texts: List[str] = []
    key_infos: List[str] = []
    if not isinstance(chain, list):
        return news_texts, key_infos

    for seg in chain:
        try:
            raw_data = ""
            if isinstance(seg, dict):
                seg_type = str(seg.get("type", "")).lower()
                if seg_type != "json":
                    continue
                raw_data = str((seg.get("data") or {}).get("data") or "")
            else:
                seg_type_obj = getattr(seg, "type", None)
                seg_type_name = str(getattr(seg_type_obj, "name", "")).lower()
                seg_type_text = str(seg_type_obj).lower()
                if seg_type_name != "json" and not seg_type_text.endswith(".json") and seg_type_text != "json":
                    continue

                seg_data = getattr(seg, "data", None)
                if isinstance(seg_data, dict):
                    news, info = _parse_json_data(seg_data)
                    news_texts.extend(news)
                    if info:
                        key_infos.append(info)
                    continue
                raw_data = str(seg_data or "")

            if raw_data:
                news, info = parse_json_segment_data(raw_data)
                news_texts.extend(news)
                if info:
                    key_infos.append(info)
        except Exception:
            continue

    return news_texts, key_infos


def _parse_json_data(inner_json: Dict[str, Any]) -> Tuple[List[str], str]:
    """从反序列化后的 JSON 对象提取可注入信息。"""
    news_texts: List[str] = []
    if (
        inner_json.get("app") == "com.tencent.multimsg"
        and inner_json.get("config", {}).get("forward") == 1
    ):
        news_items = inner_json.get("meta", {}).get("detail", {}).get("news", [])
        if isinstance(news_items, list):
            for item in news_items:
                if not isinstance(item, dict):
                    continue
                text_content = item.get("text")
                if text_content:
                    clean_text = str(text_content).strip().replace("[图片]", "").strip()
                    if clean_text:
                        news_texts.append(clean_text)

    key_info = extract_json_key_info(inner_json)
    return news_texts, key_info

