import re
from typing import List, Optional, Iterable, Set

import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

_LEM = WordNetLemmatizer()

# 冠词/指示词/一些弱量词（按需扩展）
_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "each", "every"
}

# 常见“泛词”，不建议当 class（按你的数据集可删减）
_GENERIC_NOUNS = {
    "thing", "things", "object", "objects", "item", "items", "stuff",
    "photo", "picture", "image", "scene", "drawing", "sketch", "figure"
}

# 数量词（避免 "two dogs" 变成 class="two dog"）
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "many", "several", "few"
}

def _normalize_text(s: str) -> str:
    s = s.lower()
    # 去标点但保留连字符（fire-hydrant）
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_np(np: str) -> str:
    """
    NP 规范化：
    - 小写、去标点
    - 去开头冠词/数量词
    - 词形还原（主要处理复数名词）
    """
    np = _normalize_text(np)
    words = np.split()
    # 去掉开头的冠词/数量词
    while words and (words[0] in _DETERMINERS or words[0] in _NUMBER_WORDS):
        words = words[1:]

    if not words:
        return ""

    # 对每个词 lemmatize，名词优先（复数->单数）
    words = [_LEM.lemmatize(w, pos="n") for w in words]
    np = " ".join(words).strip()
    return np

def extract_noun_phrases(tokens: List[str]) -> List[str]:
    """
    使用 RegexpParser 抽取名词短语 NP，并直接遍历 NP 子树：
    - 不会漏掉句尾 NP
    - 不会把连续 NP 错拼在一起
    """
    grammar = r"""
        NBAR: {<JJ.*|NN.*>*<NN.*>}      # (adj|noun)* + noun
        NP:   {<NBAR>}
              {<NBAR><IN><NBAR>}       # e.g., "cup of tea"
              {<NBAR><POS><NBAR>}      # e.g., "dog 's tail"（所有格，效果视 pos_tag 而定）
    """
    chunker = nltk.RegexpParser(grammar)
    tagged = nltk.pos_tag(tokens)
    tree = chunker.parse(tagged)

    nps = []
    for st in tree.subtrees(lambda t: isinstance(t, nltk.Tree) and t.label() == "NP"):
        phrase = " ".join(w for w, _ in st.leaves())
        nps.append(phrase)
    return nps

def extract_classes_from_caption(
    caption: str,
    max_classes: int = 3,
    train_classes: Optional[Iterable[str]] = None,
    prefer_vocab_match: bool = True,
    keep_multiword: bool = True,
    drop_generic: bool = True,
) -> List[str]:
    """
    从 caption 提取 class（名词短语为主），用于塞进 template 再编码。
    返回的字符串可直接喂给 model.encode_text。

    - train_classes: 若提供，会优先返回词表中的类名（稳定 label）
    - keep_multiword: True 保留多词类（traffic light）；False 仅取 head noun（light）
    """
    cap = _normalize_text(caption.replace("\n", " "))
    tokens = nltk.word_tokenize(cap)

    # 1) 抽 NP 候选
    nps_raw = extract_noun_phrases(tokens)

    # 2) 规范化 + 去重（保序）
    cands: List[str] = []
    seen: Set[str] = set()
    for np in nps_raw:
        npn = _normalize_np(np)
        if not npn:
            continue
        if not keep_multiword:
            npn = npn.split()[-1]  # head noun
        if drop_generic and npn in _GENERIC_NOUNS:
            continue
        if npn not in seen:
            seen.add(npn)
            cands.append(npn)

    if not cands:
        return []

    # 3) 若有 train_classes：优先对齐词表
    if train_classes is not None:
        # 词表规范化（一次性构建映射：norm -> orig）
        norm_to_orig = {}
        vocab_norm_set = set()
        for orig in train_classes:
            n = _normalize_np(orig)
            if not n:
                continue
            vocab_norm_set.add(n)
            # 同一个 norm 可能对应多个原词，保留更长/更具体的
            if n not in norm_to_orig or len(orig) > len(norm_to_orig[n]):
                norm_to_orig[n] = orig

        matched, rest = [], []
        for c in cands:
            if c in vocab_norm_set:
                matched.append(norm_to_orig.get(c, c))
            else:
                head = c.split()[-1]
                if head in vocab_norm_set:
                    matched.append(norm_to_orig.get(head, head))
                else:
                    rest.append(c)

        cands = matched + rest if prefer_vocab_match else cands

        # 再去重（matched 后可能重复）
        final = []
        seen2 = set()
        for x in cands:
            xn = _normalize_np(x)
            if drop_generic and xn in _GENERIC_NOUNS:
                continue
            if xn not in seen2:
                seen2.add(xn)
                final.append(x)

        return final[:max_classes]

    # 4) 无词表：优先更具体（多词/更长）但尽量不打乱顺序
    # 简单策略：按“词数多 → 更长 → 更早出现”排序
    final = sorted(
        cands,
        key=lambda x: (-(len(x.split())), -len(x))
    )
    return final[:max_classes]
