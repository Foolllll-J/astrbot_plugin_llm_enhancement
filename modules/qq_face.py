from __future__ import annotations

from typing import Any, Optional

import astrbot.api.message_components as Comp


QQ_FACE_ID_TO_NAME: dict[int, str] = {
    4: "得意",
    5: "流泪",
    8: "睡",
    9: "大哭",
    10: "尴尬",
    12: "调皮",
    14: "微笑",
    16: "酷",
    21: "可爱",
    23: "傲慢",
    24: "饥饿",
    25: "困",
    26: "惊恐",
    27: "流汗",
    28: "憨笑",
    29: "悠闲",
    30: "奋斗",
    32: "疑问",
    33: "嘘",
    34: "晕",
    38: "敲打",
    39: "再见",
    41: "发抖",
    42: "爱情",
    43: "跳跳",
    49: "拥抱",
    53: "蛋糕",
    60: "咖啡",
    63: "玫瑰",
    66: "爱心",
    74: "太阳",
    75: "月亮",
    76: "赞",
    78: "握手",
    79: "胜利",
    85: "飞吻",
    89: "西瓜",
    96: "冷汗",
    97: "擦汗",
    98: "抠鼻",
    99: "鼓掌",
    100: "糗大了",
    101: "坏笑",
    102: "左哼哼",
    103: "右哼哼",
    104: "哈欠",
    106: "委屈",
    109: "左亲亲",
    111: "可怜",
    116: "示爱",
    118: "抱拳",
    120: "拳头",
    122: "爱你",
    123: "NO",
    124: "OK",
    125: "转圈",
    129: "挥手",
    144: "喝彩",
    147: "棒棒糖",
    171: "茶",
    173: "泪奔",
    174: "无奈",
    175: "卖萌",
    176: "小纠结",
    179: "doge",
    180: "惊喜",
    181: "骚扰",
    182: "笑哭",
    183: "我最美",
    201: "点赞",
    203: "托脸",
    212: "托腮",
    214: "啵啵",
    219: "踩一踩",
    222: "抱抱",
    227: "拍手",
    232: "佛系",
    240: "喷脸",
    243: "甩头",
    246: "加油抱抱",
    262: "脑阔疼",
    264: "捂脸",
    265: "辣眼睛",
    266: "哦哟",
    267: "头秃",
    268: "问号脸",
    269: "暗中观察",
    270: "emm",
    271: "吃瓜",
    272: "呵呵哒",
    273: "我酸了",
    277: "汪汪",
    278: "汗",
    281: "无眼笑",
    282: "敬礼",
    284: "面无表情",
    285: "摸鱼",
    287: "哦",
    289: "睁眼",
    290: "敲开心",
    293: "摸锦鲤",
    294: "期待",
    297: "拜谢",
    298: "元宝",
    299: "牛啊",
    305: "右亲亲",
    306: "牛气冲天",
    307: "喵喵",
    314: "仔细分析",
    315: "加油",
    318: "崇拜",
    319: "比心",
    320: "庆祝",
    322: "拒绝",
    324: "吃糖",
    326: "生气",
}


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def normalize_qq_face_name(name: Any) -> str:
    text = str(name or "").strip()
    if text.startswith("/"):
        text = text[1:].strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text


def _extract_face_data(segment: Any) -> tuple[Optional[int], Any]:
    if isinstance(segment, Comp.Face):
        return _safe_int(getattr(segment, "id", None)), None
    if isinstance(segment, dict):
        if str(segment.get("type") or "").strip().lower() != "face":
            return None, None
        data = segment.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        return _safe_int(data.get("id")), data.get("raw")
    return None, None


def _extract_raw_face_names(raw_message: Any) -> list[str]:
    raw_segments: list[Any] = []
    if isinstance(raw_message, dict):
        raw_segments = list(raw_message.get("message", []) or [])
    elif hasattr(raw_message, "get"):
        try:
            raw_segments = list(raw_message.get("message", []) or [])
        except Exception:
            raw_segments = []

    names: list[str] = []
    for segment in raw_segments:
        face_id, raw = _extract_face_data(segment)
        if face_id is None:
            continue

        face_name = ""
        if isinstance(raw, dict):
            face_name = normalize_qq_face_name(raw.get("faceText"))
        if not face_name:
            face_name = normalize_qq_face_name(QQ_FACE_ID_TO_NAME.get(face_id, ""))
        names.append(face_name)
    return names


def resolve_qq_face_name(segment: Any) -> str:
    face_id, raw = _extract_face_data(segment)
    if face_id is None:
        return ""

    if isinstance(raw, dict):
        face_text = normalize_qq_face_name(raw.get("faceText"))
        if face_text:
            return face_text

    mapped = normalize_qq_face_name(QQ_FACE_ID_TO_NAME.get(face_id, ""))
    if mapped:
        return mapped

    return f"QQ官方表情(id={face_id})"


def build_qq_face_text(segment: Any) -> str:
    face_name = resolve_qq_face_name(segment)
    if not face_name:
        return ""
    return f"[QQ官方表情:{face_name}]"


def _resolve_message_chain_face_texts(
    message_chain: Any,
    raw_message: Any = None,
) -> list[str]:
    raw_face_names = _extract_raw_face_names(raw_message)
    raw_face_index = 0
    resolved_texts: list[str] = []

    for segment in message_chain or []:
        face_id, raw = _extract_face_data(segment)
        if face_id is None:
            continue

        resolved = ""
        if isinstance(raw, dict):
            resolved = normalize_qq_face_name(raw.get("faceText"))

        if not resolved and raw_face_index < len(raw_face_names):
            resolved = raw_face_names[raw_face_index]
            raw_face_index += 1

        if not resolved:
            resolved = resolve_qq_face_name(segment)

        if resolved:
            resolved_texts.append(f"[QQ官方表情:{resolved}]")

    return resolved_texts


def has_qq_face_segment(message_chain: Any, raw_message: Any = None) -> bool:
    return bool(_resolve_message_chain_face_texts(message_chain, raw_message=raw_message))


def build_message_text_with_qq_faces(
    message_chain: Any,
    fallback_text: str = "",
    raw_message: Any = None,
) -> str:
    resolved_face_texts = _resolve_message_chain_face_texts(
        message_chain,
        raw_message=raw_message,
    )
    if not resolved_face_texts:
        return str(fallback_text or "").strip()

    parts: list[str] = []
    face_index = 0
    for segment in message_chain or []:
        if isinstance(segment, Comp.Plain):
            text = str(getattr(segment, "text", "") or "").strip()
            if text:
                parts.append(text)
            continue

        if isinstance(segment, dict):
            seg_type = str(segment.get("type") or "").strip().lower()
            data = segment.get("data") or {}
            if seg_type == "text":
                text = str((data or {}).get("text") or "").strip()
                if text:
                    parts.append(text)
                continue

        face_id, _raw = _extract_face_data(segment)
        if face_id is not None and face_index < len(resolved_face_texts):
            parts.append(resolved_face_texts[face_index])
            face_index += 1

    if parts:
        return " ".join(parts).strip()

    return str(fallback_text or "").strip()
