import asyncio
import math
import re
from collections import defaultdict, deque, OrderedDict
from typing import Any

import jieba

class Similarity:
    """
    相关性检测（TF-IDF + 语义降噪）
    """

    def __init__(
        self,
        stopwords=None,
        bot_template_threshold: int = 2,
        early_stop: float = 0.92,
        idf_window_docs: int = 400,
        token_cache_size: int = 1024,
    ):
        """初始化相关性计算参数。"""
        self._GROUP_DATA = defaultdict(
            lambda: {
                "idf": defaultdict(int),
                "total_docs": 0,
                "docs": deque(),
            }
        )

        self.stopwords = stopwords or {
            "的", "了", "吗", "吧", "啊", "哦", "嗯", "恩",
            "你", "我", "他", "她", "它", "这", "那", "就",
            "都", "又",
        }

        self.bot_template_threshold = bot_template_threshold
        self.early_stop = early_stop
        self.idf_window_docs = max(50, int(idf_window_docs))
        self.token_cache_size = max(128, int(token_cache_size))
        self._token_cache: OrderedDict[str, list[str]] = OrderedDict()

    def _to_plain_text(self, msg: Any) -> str:
        """将消息（可能是字符串、列表或字典）转换为纯文本"""
        if isinstance(msg, str):
            return msg
        if isinstance(msg, list):
            texts = []
            for item in msg:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif hasattr(item, "text"): # 处理可能的消息对象
                    texts.append(getattr(item, "text"))
            return " ".join(texts)
        return str(msg)

    def _is_question_like(self, text: str) -> bool:
        s = (text or "").strip().lower()
        if not s:
            return False
        if "?" in s or "？" in s:
            return True
        cues = ("请问", "求教", "求助", "怎么", "如何", "为什么", "啥意思", "什么", "吗", "呢")
        return any(c in s for c in cues)

    def _has_min_semantic_content(self, text: str, tokens: list[str]) -> bool:
        """最小语义门槛：过滤过短/过空泛消息，降低相关性误唤醒。"""
        if len(tokens) >= 2:
            return True

        # 单 token 时，要求至少有 3 个连续中文，避免“在吗/啊/嗯”这类空泛触发
        zh_spans = re.findall(r"[\u4e00-\u9fa5]{3,}", text)
        if zh_spans:
            return True
        return False

    def _query_relevance_multiplier(self, text: str) -> float:
        """查询文本的相关性倍率：短句与问句轻微降权，避免与 ask_wake 抢触发。"""
        s = (text or "").strip().lower()
        if not s:
            return 1.0

        multiplier = 1.0
        if len(s) <= 2:
            multiplier *= 0.60
        elif len(s) <= 4:
            multiplier *= 0.82

        if self._is_question_like(s):
            multiplier *= 0.90

        return min(1.0, max(0.2, multiplier))

    async def _tokenize(self, text: Any):
        text = self._to_plain_text(text)
        text = re.sub(r"[^\w\u4e00-\u9fa5]", " ", text)
        key = text.strip()
        if not key:
            return []

        cached = self._token_cache.get(key)
        if cached is not None:
            self._token_cache.move_to_end(key, last=True)
            return cached

        if len(key) > 500:
            tokens = await asyncio.to_thread(jieba.lcut, key)
        else:
            tokens = jieba.lcut(key)
        filtered = [t for t in tokens if t not in self.stopwords and t.strip()]

        self._token_cache[key] = filtered
        self._token_cache.move_to_end(key, last=True)
        if len(self._token_cache) > self.token_cache_size:
            self._token_cache.popitem(last=False)
        return filtered

    def _is_noise_msg(self, text: Any) -> bool:
        s = self._to_plain_text(text).strip()

        # 空消息
        if not s:
            return True

        # 纯 CQ 码，如 [CQ:reply,id=xxx]
        if re.fullmatch(r"\[CQ:[^\]]+]", s):
            return True

        # 纯 emoji / 符号
        if re.fullmatch(r"[\W_]+", s):
            return True

        # 纯数字和标点（如“123。。。!!”）
        if re.fullmatch(r"[\d\W_]+", s):
            return True

        return False

    async def _preprocess_bot_msgs(self, msgs: list) -> list[str]:
        cleaned = []
        seen = set()

        for m_raw in msgs:
            if not m_raw:
                continue
            
            m = self._to_plain_text(m_raw)

            # 去重
            if m in seen:
                continue
            seen.add(m)

            # 噪音过滤
            if self._is_noise_msg(m):
                continue

            # token 数过滤（模板句过滤）
            tokens = await self._tokenize(m)
            if len(tokens) <= self.bot_template_threshold:
                continue

            cleaned.append(m)

        return cleaned

    def _update_idf(self, group_id: str, tokens: set):
        data = self._GROUP_DATA[group_id]
        docs = data["docs"]  # type: ignore

        while len(docs) >= self.idf_window_docs:
            old_tokens = docs.popleft()
            for t in old_tokens:
                data["idf"][t] -= 1  # type: ignore
                if data["idf"][t] <= 0:  # type: ignore
                    del data["idf"][t]  # type: ignore
            data["total_docs"] = max(0, data["total_docs"] - 1)  # type: ignore

        docs.append(tokens)
        for t in tokens:
            data["idf"][t] += 1 # type: ignore
        data["total_docs"] += 1  # type: ignore

    def _tfidf_vector(self, group_id: str, tokens: list):
        data = self._GROUP_DATA[group_id]
        total_docs = data["total_docs"] or 1

        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        vec = {}
        for t, c in tf.items():
            idf = math.log((total_docs + 1) / (data["idf"][t] + 1)) + 1  # type: ignore
            vec[t] = c * idf

        return vec

    def _cosine(self, v1, v2):
        if not v1 or not v2:
            return 0.0

        dot = sum(v * v2.get(k, 0) for k, v in v1.items())
        norm1 = math.sqrt(sum(v * v for v in v1.values()))
        norm2 = math.sqrt(sum(v * v for v in v2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def context_count_for_query(self, user_msg: str) -> int:
        """根据当前消息长度动态选择相关性上下文条数。"""
        s = self._to_plain_text(user_msg)
        semantic_len = len(re.sub(r"\s+", "", s))
        if semantic_len <= 8:
            return 4
        if semantic_len >= 25:
            return 8
        return 6

    async def similarity(
        self,
        group_id: str,
        user_msg: str,
        bot_msgs: list[str],
        update_history: bool = True,
    ) -> float:
        """
        计算用户消息与一组 bot 消息的最大相似度
        """
        raw_user = self._to_plain_text(user_msg).strip()
        if self._is_noise_msg(raw_user):
            return 0.0

        # 分词
        user_tokens = await self._tokenize(raw_user)
        if (not user_tokens) or (not self._has_min_semantic_content(raw_user, user_tokens)):
            return 0.0

        # 更新历史（可关闭）
        if update_history:
            self._update_idf(group_id, set(user_tokens))

        # 用户向量
        user_vec = self._tfidf_vector(group_id, user_tokens)

        # bot 消息预处理 + 最近优先
        bot_list = (await self._preprocess_bot_msgs(bot_msgs))[::-1]

        best = 0.0
        for idx, bm in enumerate(bot_list):
            btokens = await self._tokenize(bm)
            bvec = self._tfidf_vector(group_id, btokens)
            sim = self._cosine(user_vec, bvec)
            # 最近消息权重更高，降低被旧上下文“误相关”唤醒的概率
            recency_weight = max(0.75, 1.0 - idx * 0.04)
            adjusted = sim * recency_weight
            if adjusted > best:
                best = adjusted
            if best >= self.early_stop:
                break
        return best * self._query_relevance_multiplier(raw_user)
