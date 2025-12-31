import math
import jieba
from collections import defaultdict, deque
import re


class Similarity:
    """按群号隔离的动态话题关联性检测器"""

    # 基础停用词表
    STOP = {
        "的",
        "了",
        "在",
        "是",
        "和",
        "与",
        "或",
        "这",
        "那",
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

    # 每个群维护自己的状态
    _GROUP_DATA = defaultdict(
        lambda: {
            "cache": deque(maxlen=20),
            "weights": defaultdict(float),
        }
    )
    DECAY_FACTOR = 0.95  # 权重衰减因子（全局统一）


    # 内部工具
    @classmethod
    def _state(cls, group_id: str):
        """根据群号拿到该群的独立数据"""
        return cls._GROUP_DATA[group_id]

    @classmethod
    def _update_topic_cache(cls, words, group_id: str):
        """更新指定群的话题缓存和权重"""
        st = cls._state(group_id)

        # 更新缓存
        for word in words:
            if (
                word not in cls.STOP
                and len(word) > 1
                and re.match(r"^[\u4e00-\u9fa5]+$", word)
            ):
                st["cache"].append(word) # type: ignore

        # 计算频率
        freq = defaultdict(int)
        for w in st["cache"]:
            freq[w] += 1

        # 更新权重
        for w, cnt in freq.items():
            decayed = st["weights"].get(w, 0) * cls.DECAY_FACTOR  # type: ignore
            current = cnt * (1.0 + math.log(len(w) or 1))
            st["weights"][w] = max(decayed, current)

    @classmethod
    def _extract_keywords(cls, s: str, group_id: str) -> list:
        """提取关键词并更新指定群的话题缓存"""
        if not isinstance(s, str):
            if isinstance(s, list):
                # 如果是列表，尝试将其中的文本部分提取并拼接
                texts = []
                for item in s:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                s = " ".join(texts)
            else:
                s = str(s)
                
        s = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", s)
        words = [w for w in jieba.lcut(s) if w.strip() and w not in cls.STOP]

        # 合并连续数字/单字
        merged = []
        for w in words:
            if merged and (
                (w.isdigit() and merged[-1][-1].isdigit())
                or (len(merged[-1]) == 1 and len(w) == 1)
            ):
                merged[-1] += w
            else:
                merged.append(w)

        cls._update_topic_cache(merged, group_id)
        return merged


    # 公共 API
    @classmethod
    def _tokens(cls, s: str, group_id: str) -> dict[str, float]:
        """生成带权重的词向量（群内）"""
        words = cls._extract_keywords(s, group_id)
        st = cls._state(group_id)
        tf = defaultdict(float)

        # 一元词
        for w in words:
            weight = 1.0 + st["weights"].get(w, 0)  # type: ignore
            tf[w] += weight

        # 二元词
        for i in range(len(words) - 1):
            bigram = words[i] + words[i + 1]
            tf[bigram] += 1.5

        total = max(sum(tf.values()), 1)
        return {w: c / total for w, c in tf.items()}

    @classmethod
    def cosine(cls, a: str, b: str, group_id: str = "default") -> float:
        """计算同一群内两条文本的相似度"""
        v1 = cls._tokens(a, group_id)
        v2 = cls._tokens(b, group_id)
        all_w = set(v1) | set(v2)

        dot = 0
        for w in all_w:
            x, y = v1.get(w, 0), v2.get(w, 0)
            if x > 0 and y > 0:
                dot += x * y * (2.0 + cls._state(group_id)["weights"].get(w, 0))  # type: ignore
            else:
                dot += x * y

        norm1 = math.sqrt(sum(v * v for v in v1.values()))
        norm2 = math.sqrt(sum(v * v for v in v2.values()))
        raw = dot / (norm1 * norm2 + 1e-8)
        return 1 / (1 + math.exp(-8 * (raw - 0.6)))

    @classmethod
    def get_current_topics(cls, group_id: str = "default", top_n: int = 5):
        """获取指定群当前最重要的 top_n 个话题"""
        st = cls._state(group_id)
        return sorted(st["weights"].items(), key=lambda kv: kv[1], reverse=True)[:top_n]  # type: ignore


    # 管理工具（可选）
    @classmethod
    def clear_group(cls, group_id: str):
        """清空某个群的所有话题数据"""
        cls._GROUP_DATA.pop(group_id, None)

    @classmethod
    def list_groups(cls):
        """返回当前有数据的群号列表"""
        return list(cls._GROUP_DATA.keys())
