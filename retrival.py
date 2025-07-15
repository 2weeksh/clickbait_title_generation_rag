import argparse
import json
import random
from typing import Any, Dict, List

from bm25s import BM25
from kiwipiepy import Kiwi

parser = argparse.ArgumentParser(description="유사 기사 검색하는 스크립트")
# 경로 설정
parser.add_argument(
    "--clickbait_path",
    type=str,
    default="/home/joohyuk02/joohyuk/project1/data/TL_Part1_Clickbait_Direct_merged.json",
)
parser.add_argument(
    "--non_clickbait_path",
    type=str,
    default="/home/joohyuk02/joohyuk/project1/data/VL_Part1_Clickbait_Direct_merged.json",
)
parser.add_argument(
    "--rag_retrieval_path",
    type=str,
    default="/home/joohyuk02/joohyuk/project1/testtest/rag_retrieval_results.json",
)
# 하이퍼파라미터 설정
parser.add_argument("--min_num_character", type=int, default=100, help="corpus에 포함될 기사 최소 글자 수")  #
parser.add_argument("--max_num_character", type=int, default=3000, help="corpus에 포함될 기사 최대 글자 수")
parser.add_argument("--k", type=int, default=5, help="검색할 유사 기사 개수")
parser.add_argument("--sample_size", type=int, default=200, help="쿼리 샘플 개수")  #
args = parser.parse_args()


# 토크나이저 초기화
kiwi = Kiwi()


# 토크나이저
def tokenize(texts: List[str]) -> List[List[str]]:
    return [[token.form for token in kiwi.tokenize(text)] for text in texts]


# 검색 코퍼스 생성
def build_corpus(clickbait_path: str, min_len: int, max_len: int):
    corpus = []
    corpus_raw = {}

    # 데이터 열기
    with open(clickbait_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    # 테스트용 코드
    # corpus_data = corpus_data[:30]

    for item in corpus_data:
        text = f"{item['news_title']} {item['new_title']} {item['news_content']}"
        if min_len <= len(text) <= max_len:  # 글자 수 필터링
            corpus.append(text)
            corpus_raw[text] = item

    return corpus, corpus_raw


def create_retriever(corpus: List[str]) -> BM25:
    # corpus 토큰화 및 BM25 인덱싱
    corpus_tokens = tokenize(corpus)
    retriever = BM25(corpus=corpus, k1=1.5, b=0.75)
    retriever.index(corpus_tokens)
    return retriever


def retrieve_title(
    non_clickbait_path: str, retriever: BM25, corpus_raw: Dict[str, Dict[str, Any]], k: int, sample_size: int
) -> List[Dict[str, Any]]:
    # 데이터 열기
    with open(non_clickbait_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    random.seed(42)
    sampled_queries = random.sample(query_data, sample_size)

    rag_queries = []

    # 기사마다 검색 수행
    for query in sampled_queries:
        query_text = query["news_title"] + " " + query["news_content"]
        query_tokens = tokenize([query_text])[0]

        # 검색
        docs_list, _ = retriever.retrieve([query_tokens], k=k)
        retrieved_articles = []

        # 검색 문서들 저장
        for doc in docs_list[0]:
            raw_doc = corpus_raw[doc]
            if raw_doc:
                retrieved_articles.append(
                    {
                        "non_clickbait_title": raw_doc["news_title"],
                        "clickbait_title": raw_doc["new_title"],
                        "clickbait_content": raw_doc["news_content"],
                    }
                )

        # 결과 양식
        rag_queries.append(
            {
                "query_title": query["news_title"],
                "query_content": query["news_content"],
                "human_direct_clickbait_title": query["new_title"],
                "retrieved_articles": retrieved_articles,
            }
        )

    return rag_queries


if __name__ == "__main__":
    corpus_list, corpus_map = build_corpus(args.clickbait_path, args.min_num_character, args.max_num_character)
    bm25_retriever = create_retriever(corpus_list)
    rag_queries = retrieve_title(args.non_clickbait_path, bm25_retriever, corpus_map, args.k, args.sample_size)

    # 저장
    with open(args.rag_retrieval_path, "w", encoding="utf-8") as f:
        json.dump(rag_queries, f, ensure_ascii=False, indent=2)
