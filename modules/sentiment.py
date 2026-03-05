import asyncio
import math
import jieba
import re
from astrbot.api import logger

# 扩充 jieba 词典，优先补充分词易拆散的高价值词
CUSTOM_JIEBA_WORDS = (
    # insult
    "傻逼",
    "神经病",
    "草泥马",
    # shutup
    "闭嘴",
    "安静",
    # bored
    "无聊",
    "没劲",
    "没意思",
    "有人吗",
    # ask
    "请问",
    "求教",
    "求助",
    "请教",
    "求解",
    "怎么办",
    "如何解决",
    "怎么处理",
    "怎么做",
    "怎么回事",
    "什么原因",
    "可不可以",
    "能不能",
    "会不会",
    "有没有",
    "谁知道",
    "谁懂",
    "有人会",
    # discourse
    "别骂人",
    "不要骂人",
)

for _word in CUSTOM_JIEBA_WORDS:
    jieba.add_word(_word)

class Sentiment:
    """
    高精度语义检测器 - 优化版词表
    """

    # 精简停用词表
    STOP = {
        "的",
        "了",
        "在",
        "是",
        "都",
        "就",
        "也",
        "和",
        "把",
        "我",
        "你",
        "他",
        "她",
        "它",
        "啊",
        "吧",
        "吗",
        "嘛",
    }

    # 闭嘴类关键词 - 按强度分级
    SHUT_WORDS = {
        # 强命令 (权重1.0, 强度1.8-2.0)
        "闭嘴": (1.0, 2.0),
        "住口": (1.0, 1.9),
        "安静": (1.0, 1.8),
        "shut up": (1.0, 2.0),
        "别说话": (1.0, 1.8),
        "别吵": (1.0, 1.8),
        "别出声": (1.0, 1.7),
        "别嚷嚷": (1.0, 1.7),
        # 中度命令 (权重0.9, 强度1.4-1.6)
        "安静点": (0.9, 1.5),
        "小点声": (0.9, 1.4),
        "别吵了": (0.9, 1.6),
        "别闹了": (0.8, 1.4),
        "别烦我": (0.8, 1.5),
        "别打扰": (0.8, 1.4),
        "别插嘴": (0.8, 1.5),
        # 弱表达 (权重0.7, 强度1.2-1.3)
        "太吵了": (0.7, 1.3),
        "吵死了": (0.7, 1.3),
        "好吵": (0.6, 1.2),
        "话多": (0.6, 1.2),
        "话痨": (0.6, 1.2),
        "少说点": (0.5, 1.1),
        "少说话": (0.5, 1.1),
    }

    # 侮辱类关键词 - 按严重程度分级
    INSULT_WORDS = {
        # 严重侮辱 (权重1.0, 强度1.9-2.0)
        "傻逼": (1.0, 2.0),
        "sb": (1.0, 1.9),
        "nmsl": (1.0, 2.0),
        "去死": (1.0, 2.0),
        "草泥马": (1.0, 1.9),
        "cnm": (1.0, 1.9),
        "废物": (1.0, 1.8),
        "垃圾": (1.0, 1.8),
        "脑残": (1.0, 1.8),
        "弱智": (1.0, 1.7),
        "智障": (1.0, 1.7),
        # 中度侮辱 (权重0.9, 强度1.5-1.6)
        "有病": (0.9, 1.6),
        "神经病": (0.9, 1.6),
        "白痴": (0.9, 1.6),
        "蠢货": (0.9, 1.5),
        "滚": (0.9, 1.7),
        "滚开": (0.9, 1.6),
        "滚蛋": (0.9, 1.7),
        "nt": (0.9, 1.6),
        "fw": (0.9, 1.6),
        "菜鸡": (0.8, 1.5),
        # 轻度侮辱 (权重0.7, 强度1.3-1.4)
        "憨憨": (0.7, 1.3),
        "笨": (0.6, 1.2),
        "呆": (0.6, 1.2),
        "猪": (0.7, 1.3),
        "没脑子": (0.8, 1.4),
        "没出息": (0.7, 1.3),
        "low": (0.7, 1.4),
    }

    # 无聊类关键词 - 按表达强度分级
    BORED_WORDS = {
        # 强烈表达 (权重1.0, 强度1.7-1.8)
        "无聊死了": (1.0, 1.8),
        "好无聊": (1.0, 1.7),
        "太无聊": (1.0, 1.7),
        "闷死了": (1.0, 1.7),
        "好没劲": (1.0, 1.6),
        "真没意思": (1.0, 1.6),
        "闲得慌": (1.0, 1.6),
        # 中度表达 (权重0.8, 强度1.4-1.5)
        "无聊": (0.8, 1.5),
        "好闲": (0.8, 1.4),
        "寂寞": (0.8, 1.4),
        "冷清": (0.8, 1.4),
        "空虚": (0.7, 1.3),
        "没人": (0.7, 1.3),
        "冷场": (0.8, 1.5),
        "死群": (0.8, 1.5),
        # 轻度表达 (权重0.6, 强度1.1-1.2)
        "有点闷": (0.6, 1.2),
        "没事做": (0.6, 1.1),
        "打发时间": (0.6, 1.1),
        "求聊天": (0.7, 1.3),
        "有人吗": (0.7, 1.4),
        "在吗": (0.5, 1.0),
        "滴滴": (0.5, 1.0),
    }

    # 提问类关键词 - 按提问明确度分级
    ASK_WORDS = {
        # 明确提问 (权重1.0, 强度1.7-1.8)
        "请问": (1.0, 1.8),
        "求解": (1.0, 1.8),
        "求教": (1.0, 1.7),
        "请教": (1.0, 1.7),
        "如何解决": (1.0, 1.8),
        "怎么处理": (1.0, 1.7),
        "怎么办": (1.0, 1.7),
        "为什么": (1.0, 1.6),
        "什么原因": (1.0, 1.6),
        "怎么回事": (1.0, 1.7),
        "谁能帮": (1.0, 1.7),
        # 一般提问 (权重0.9, 强度1.4-1.5)
        "怎么": (0.9, 1.5),
        "如何": (0.9, 1.5),
        "啥意思": (0.9, 1.5),
        "怎么做": (0.9, 1.6),
        "哪里": (0.8, 1.4),
        "哪个": (0.8, 1.4),
        "哪能": (0.8, 1.4),
        "有什么": (0.8, 1.4),
        "有没有": (0.8, 1.4),
        "会不会": (0.8, 1.4),
        "能不能": (0.8, 1.5),
        "可不可以": (0.8, 1.5),
        # 模糊提问 (权重0.7, 强度1.2-1.3)
        "什么": (0.7, 1.3),
        "啥": (0.7, 1.2),
        "呢": (0.5, 1.1),
        "吗": (0.5, 1.0),
        "谁懂": (0.8, 1.4),
        "谁知道": (0.8, 1.4),
        "有人会": (0.7, 1.3),
    }

    # 否定词表 - 用于降低可信度
    NEGATION_WORDS = {
        "不",
        "没",
        "无",
        "非",
        "否",
        "别",
        "不要",
        "不太",
        "不太想",
        "不想",
        "不至于",
        "算不上",
        "才不",
        "才不会",
    }

    # 反问词表 - 可能改变语义
    RHETORICAL_WORDS = {"难道", "何必", "怎么可以", "怎么可能", "哪能", "岂能", "谁还"}
    QUESTION_CUES = (
        "请问",
        "求教",
        "求助",
        "怎么",
        "如何",
        "为啥",
        "为什么",
        "啥意思",
        "什么",
        "哪",
        "吗",
        "么",
        "嘛",
        "呢",
        "有没有",
        "能不能",
        "可不可以",
        "谁知道",
        "谁懂",
    )
    BORED_WAKE_CUES = (
        "无聊",
        "好无聊",
        "太无聊",
        "好闲",
        "寂寞",
        "冷清",
        "冷场",
        "死群",
        "没人",
        "求聊天",
        "有人吗",
        "在吗",
        "滴滴",
        "打发时间",
        "没事做",
        "闷",
        "闷死了",
        "好没劲",
        "真没意思",
    )
    ASK_STRONG_CUES = (
        "请问",
        "求教",
        "求助",
        "怎么",
        "如何",
        "为什么",
        "啥意思",
        "什么意思",
    )
    SHUT_IMPERATIVE_CUES = (
        "你闭嘴",
        "给我闭嘴",
        "闭嘴",
        "住口",
        "安静",
        "安静点",
        "别说话",
        "别吵",
        "别出声",
        "别嚷嚷",
        "少说点",
        "少说话",
        "小点声",
    )
    SHUT_REPORTING_CUES = (
        "让我闭嘴",
        "叫我闭嘴",
        "让我住口",
        "说我闭嘴",
        "说我话多",
        "不是让你闭嘴",
        "并不是让你闭嘴",
        "我闭嘴",
        "那我闭嘴",
        "我先闭嘴",
        "我不说了",
        "我不说话了",
        "闭嘴了",
    )
    INSULT_DIRECT_CUES = (
        "你这",
        "你个",
        "你是",
        "你真",
        "你也配",
        "给我滚",
        "滚开",
        "滚蛋",
    )
    INSULT_DISCOURSE_CUES = (
        "别骂人",
        "不要骂人",
        "别骂",
        "不要骂",
        "别喷",
        "不要喷",
        "文明点",
    )
    INSULT_REPORTING_CUES = (
        "他说",
        "她说",
        "他说我",
        "她说我",
        "说我",
        "骂我",
        "被骂",
        "有人骂",
        "刚刚骂",
    )

    @classmethod
    async def _seg(cls, text: str) -> list:
        """分词并保留位置信息"""
        text = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", text.lower())
        words = []
        # jieba.lcut 是 CPU 密集型操作，在大文本下可能阻塞事件循环
        if len(text) > 500:
            lcut_res = await asyncio.to_thread(jieba.lcut, text)
        else:
            lcut_res = jieba.lcut(text)
            
        for word in lcut_res:
            if word.strip() and word not in cls.STOP:
                words.append(word)
        logger.debug(f"[sentiment] {words}")
        return words

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        s = str(text or "").lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _keyword_text_fallback_pos(cls, text: str, keyword: str) -> int:
        """分词未命中时的原文短语兜底匹配位置。"""
        kw = str(keyword or "").strip().lower()
        if not kw:
            return -1

        # 单字词不做原文兜底，避免大量误命中（如“吗”“啥”）。
        if len(kw) <= 1:
            return -1

        # 英文/数字关键词使用词边界匹配，避免子串误命中。
        if re.fullmatch(r"[a-z0-9 ]+", kw):
            pattern = rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])"
            m = re.search(pattern, text)
            return m.start() if m else -1

        return text.find(kw)

    @classmethod
    def _has_negation_near_phrase(cls, text: str, phrase_pos: int) -> bool:
        if phrase_pos < 0:
            return False
        # 取短窗口，近似 token 级“前3词”效果。
        left = text[max(0, phrase_pos - 8) : phrase_pos]
        return any(neg in left for neg in cls.NEGATION_WORDS)

    @classmethod
    def _calculate_confidence(cls, words: list, keyword_dict: dict, raw_text: str = "") -> float:
        """计算语义可信度"""
        norm_text = cls._normalize_text(raw_text)

        # 1) 构建 token 索引，确保每个关键词只计一次
        token_index: dict[str, int] = {}
        for i, w in enumerate(words):
            if w not in token_index:
                token_index[w] = i

        # 2) 基础匹配分数
        base_score = 0
        matched_keywords: list[str] = []

        # 反问表达：token 或原文任一命中即视为存在
        has_rhetorical = any(r_word in token_index for r_word in cls.RHETORICAL_WORDS) or any(
            r_word in norm_text for r_word in cls.RHETORICAL_WORDS
        )

        for keyword, (weight, intensity) in keyword_dict.items():
            matched = False
            has_negation = False

            # A. 优先 token 精确命中
            if keyword in token_index:
                matched = True
                kw_idx = token_index[keyword]
                has_negation = any(
                    neg_word in words[max(0, kw_idx - 3) : kw_idx]
                    for neg_word in cls.NEGATION_WORDS
                )
            else:
                # B. 分词未命中时，走原文短语兜底
                phrase_pos = cls._keyword_text_fallback_pos(norm_text, keyword)
                if phrase_pos >= 0:
                    matched = True
                    has_negation = cls._has_negation_near_phrase(norm_text, phrase_pos)

            if not matched:
                continue

            # 否定词降低权重
            if has_negation:
                weight *= 0.3
                intensity *= 0.5
            # 反问句可能反转语义
            elif has_rhetorical:
                weight *= 0.7
                intensity *= 0.8

            base_score += weight * intensity
            matched_keywords.append(keyword)

        # 3) 上下文增强分数
        context_score = 0
        if matched_keywords:
            # 关键词密度增强
            density = len(matched_keywords) / len(words) if words else 0
            context_score += min(1.0, density * 5) * 0.5

            # 关键词组合增强
            if len(matched_keywords) > 1:
                context_score += min(1.0, (len(matched_keywords) - 1) * 0.4)

        # 4) 总分数计算
        total_score = base_score + context_score

        # 5) 应用 Sigmoid 函数转换为概率值
        confidence = 1 / (1 + math.exp(-4 * (total_score - 1.5)))

        # 6) 上限控制
        return min(0.99, confidence)

    @classmethod
    def is_question_like_message(cls, text: str) -> bool:
        """答疑唤醒前置门槛：先判断是否像提问，再进入 ask 评分。"""
        s = str(text or "").strip().lower()
        if not s:
            return False

        if "?" in s or "？" in s:
            return True

        return any(cue in s for cue in cls.QUESTION_CUES)

    @classmethod
    def is_bored_like_message(cls, text: str) -> bool:
        """无聊唤醒前置门槛：过滤明显非无聊语境，减少误唤醒。"""
        s = str(text or "").strip().lower()
        if not s:
            return False

        if len(s) > 30 and not any(cue in s for cue in ("无聊", "寂寞", "冷清", "冷场", "死群", "好闲")):
            return False

        if cls.is_question_like_message(s) and any(cue in s for cue in cls.ASK_STRONG_CUES):
            return False

        return any(cue in s for cue in cls.BORED_WAKE_CUES)

    @classmethod
    def _shut_intent_multiplier(cls, text: str) -> float:
        """闭嘴语境倍率：命令语气增强，转述/自述降权。"""
        s = str(text or "").strip().lower()
        if not s:
            return 1.0

        multiplier = 1.0
        if any(cue in s for cue in cls.SHUT_REPORTING_CUES):
            multiplier *= 0.45
        elif any(cue in s for cue in cls.SHUT_IMPERATIVE_CUES):
            multiplier *= 1.08

        return min(1.2, max(0.2, multiplier))

    @classmethod
    def _insult_intent_multiplier(cls, text: str) -> float:
        """辱骂语境倍率：直接攻击增强，劝阻/转述降权。"""
        s = str(text or "").strip().lower()
        if not s:
            return 1.0

        multiplier = 1.0
        if any(cue in s for cue in cls.INSULT_DISCOURSE_CUES):
            multiplier *= 0.45
        elif any(cue in s for cue in cls.INSULT_REPORTING_CUES):
            multiplier *= 0.65

        if any(cue in s for cue in cls.INSULT_DIRECT_CUES):
            multiplier *= 1.10

        return min(1.25, max(0.2, multiplier))

    # 对外接口
    @classmethod
    async def shut(cls, text: str) -> float:
        """判断是否要闭嘴"""
        words = await cls._seg(text)
        score = cls._calculate_confidence(words, cls.SHUT_WORDS, raw_text=text)
        if score <= 0:
            return score

        score *= cls._shut_intent_multiplier(text)
        return min(0.99, max(0.0, score))

    @classmethod
    async def insult(cls, text: str) -> float:
        """判断是否辱骂"""
        words = await cls._seg(text)
        score = cls._calculate_confidence(words, cls.INSULT_WORDS, raw_text=text)
        if score <= 0:
            return score

        score *= cls._insult_intent_multiplier(text)
        return min(0.99, max(0.0, score))

    @classmethod
    async def bored(cls, text: str) -> float:
        """判断是否无聊"""
        if not cls.is_bored_like_message(text):
            return 0.0
        words = await cls._seg(text)
        return cls._calculate_confidence(words, cls.BORED_WORDS, raw_text=text)

    @classmethod
    async def ask(cls, text: str) -> float:
        """判断是否疑惑"""
        if not cls.is_question_like_message(text):
            return 0.0
        words = await cls._seg(text)
        return cls._calculate_confidence(words, cls.ASK_WORDS, raw_text=text)
