"""BM25 retrieval using an inverted index."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from math import log1p

from src.tools.tokenizer import tokenize


@dataclass
class BM25Index:
    postings: dict[str, list[tuple[int, int]]]
    lengths: list[int]
    average_length: float


def build_index(docs: list[str]) -> BM25Index:
    '''
    输入一个列表["apple apple banana", "banana orange"]
    输出
    index.postings == {
        "apple": [(0, 2)],
        "banana": [(0, 1), (1, 1)],
        "orange": [(1, 1)],
    }
    index.lengths == [3, 2]
    index.average_length == 2.5
        
    '''
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = []
    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)
        lengths.append(len(tokens))
        for term, frequency in Counter(tokens).items():
            postings[term].append((doc_id, frequency))
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    return BM25Index(dict(postings), lengths, average_length)


def search(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    '''
    输入：
    index: BM25Index：由 build_index 建好的索引。
    query: str：要搜索的文字，使用与文档相同的规则分词。
    k: int：最多返回多少篇文档。
    输出：
    list[tuple[int, float]]
    每个元组是(doc_id, BM25分数)
    '''
    
    if k <= 0 or not index.average_length:
        return []
    scores: dict[int, float] = defaultdict(float)
    n = len(index.lengths)
    for key_word in set(tokenize(query)):
        postings = index.postings.get(key_word)
        if not postings:
            continue
        idf = log1p((n - len(postings) + 0.5) / (len(postings) + 0.5))
        for doc_id, frequency in postings:
            norm = 1.5 * (0.25 + 0.75 * index.lengths[doc_id] / index.average_length)
            scores[doc_id] += idf * frequency * 2.5 / (frequency + norm)
            
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
