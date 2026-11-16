from src.tools.load_docs import load_docs
from src.config.agent_types import BM25Index, SearchResult, DocWindow
from collections import defaultdict, Counter
from src.tools.tokenizer import tokenize
from typing import List
from math import log1p

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

class DocEngine():
    def __init__(self):
        self.docs = load_docs()
        self.index = build_index([doc["text"] for doc in self.docs])
        
    def docs_search(self, query: str, k: int = 5) -> List[SearchResult]:
        if k <= 0 or not self.index.average_length:
            return []

        query_terms = set(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        total_docs = len(self.index.lengths)

        for term in query_terms:
            postings = self.index.postings.get(term, [])
            if not postings:
                continue

            idf = log1p(
                (total_docs - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for position, frequency in postings:
                norm = 1.5 * (
                    0.25
                    + 0.75 * self.index.lengths[position]
                    / self.index.average_length
                )
                scores[position] += idf * frequency * 2.5 / (frequency + norm)

        ranked = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )[:k]

        results: List[SearchResult] = []
        for position, _score in ranked:
            doc = self.docs[position]
            snippet = ""

            for line in doc["text"].splitlines():
                line_terms = set(tokenize(line))
                if query_terms & line_terms:  # 两组词有交集
                    snippet = line.strip()
                    break                    # 只取第一个匹配的行

            snippet = snippet[:200]
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "snippet": snippet,
            })

        return results
    
    def doc_read(self, doc_id: str, start_line: int = 1) -> DocWindow:
        doc = next(
            (item for item in self.docs if item["doc_id"] == doc_id),
            None,
        )
        if doc is None:
            raise LookupError(f"未知文档: {doc_id}")

        lines = doc["text"].splitlines()
        total_lines = len(lines)
        if not 1 <= start_line <= total_lines:
            raise ValueError(f"start_line 必须在 1 到 {total_lines} 之间")

        selected = []
        for line in lines[start_line - 1:]:#文档的行号是从1开始的
            candidate = "\n".join([*selected, line])
            if len(tokenize(candidate)) > 1500:
                break
            selected.append(line)

        if not selected:
            raise ValueError("起始行本身超过 1500 tokens，无法返回完整行")

        return DocWindow(
            doc_id=doc_id,
            start_line=start_line,
            end_line=start_line + len(selected) - 1,
            total_lines=total_lines,
            text="\n".join(selected),
        )