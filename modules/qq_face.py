from __future__ import annotations

from typing import Any, Optional

from .runtime_helpers import _safe_int

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
    114: "篮球",
    116: "示爱",
    118: "抱拳",
    120: "拳头",
    122: "爱你",
    123: "NO",
    124: "OK",
    125: "转圈",
    129: "挥手",
    137: "鞭炮",
    144: "喝彩",
    147: "棒棒糖",
    171: "茶",
    173: "泪奔",
    174: "无奈",
    175: "卖萌",
    176: "小纠结",
    179: "doge",
    180: "惊喜",
    181: "戳一戳",
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
    311: "打call",
    312: "变形",
    314: "仔细分析",
    315: "加油",
    317: "菜汪",
    318: "崇拜",
    319: "比心",
    320: "庆祝",
    322: "拒绝",
    324: "吃糖",
    325: "惊吓",
    326: "生气",
    333: "烟花",
    334: "虎虎生威",
    337: "花朵脸",
    338: "我想开了",
    339: "舔屏",
    341: "打招呼",
    342: "酸Q",
    343: "我方了",
    344: "大冤种",
    345: "红包多多",
    346: "你真棒棒",
    347: "大展宏兔",
    349: "坚强",
    350: "贴贴",
    351: "敲敲",
    356: "666",
    392: "龙年快乐",
    395: "略略略",
    415: "划龙舟",
    419: "火车",
    424: "续标识",
    425: "求放过",
    426: "玩火",
    427: "偷感",
    428: "收到",
    429: "蛇年快乐",
    458: "我吗",
    459: "优雅",
    460: "硬撑",
    461: "宕机",
    462: "无语",
    464: "马上到",
    466: "羞羞哒",
    470: "马到成功",
    472: "心动",
    474: "给你一拳",
    475: "干饭",
    476: "不是哥们",
    477: "你懂的",
    478: "对的对的",
    479: "不对不对",
    480: "散味儿",
    481: "学习",
    482: "热化了",
    483: "略",
    484: "比爱心",
}


def normalize_qq_face_name(name: Any) -> str:
    """标准化 QQ 表情名称，去除前导斜杠和方括号。"""
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
    """解析 QQ 表情组件的名称，优先使用 raw 数据中的 faceText。"""
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
    """构建 QQ 表情的文本表示 [QQ官方表情:名称]。"""
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
    """判断消息链中是否包含 QQ 表情组件。"""
    return bool(
        _resolve_message_chain_face_texts(message_chain, raw_message=raw_message)
    )


def build_message_text_with_qq_faces(
    message_chain: Any,
    fallback_text: str = "",
    raw_message: Any = None,
) -> str:
    """构建包含 QQ 表情文本表示的消息文本，无表情时回退到 fallback_text。"""
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
