import json
import random
from bm25s import BM25
from kiwipiepy import Kiwi


# 기사 글자수 제한
MIN_NUM_CHARACTER = 100
MAX_NUM_CHARACTER = 3000

# 파일 경로
CLICKBAIT_PATH = "/home/joohyuk02/joohyuk/project1/data/TL_Part1_Clickbait_Direct_merged.json"
NON_CLICKBAIT_PATH = "/home/joohyuk02/joohyuk/project1/data/VL_Part1_Clickbait_Direct_merged.json"
RAG_RETRIVAL_PATH = "/home/joohyuk02/joohyuk/project1/output/rag_retrieval_results.json"

# 인자값
K1 = 1.5
B = 0.75
SEED = 11
num_samples = 200

# 토크나이저 초기화
kiwi = Kiwi()

# 토크나이저
def tokenize(texts):
    return [[token.form for token in kiwi.tokenize(text)] for text in texts]

# 검색 코퍼스 생성
def build_corpus():
    corpus = []
    corpus_raw = {}

    # 데이터 불러오기
    with open(CLICKBAIT_PATH, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)   

    for item in corpus_data:
        text = item["news_title"] + " " + item["new_title"] + " " + item["news_content"]
        if MIN_NUM_CHARACTER <= len(text) <= MAX_NUM_CHARACTER:  # 글자 수 제한
            corpus.append(text)
            corpus_raw[text] = item

    return corpus, corpus_raw


def create_retriver(corpus):
    # corpus 토큰화 및 BM25 인덱싱
    corpus_tokens = tokenize(corpus)
    retriever = BM25(corpus=corpus, k1=K1, b=B)
    retriever.index(corpus_tokens)
    return retriever


def retrieve_title(retriever, corpus_raw):
    # 데이터 불러오기
    with open(NON_CLICKBAIT_PATH, "r", encoding="utf-8") as f:
        query_data = json.load(f)


    random.seed(SEED)
    sampled_queries = random.sample(query_data, num_samples)

    # RAG용 검색 결과 구성
    rag_queries = []


    for query in sampled_queries:
        query_text = query["news_title"] + " " + query["news_content"]
        query_tokens = tokenize([query_text])[0]

        docs_list, _ = retriever.retrieve([query_tokens], k=5)
        retrieved_articles = []

        for doc in docs_list[0]:
            raw_doc = corpus_raw.get(doc)
            if raw_doc:
                retrieved_articles.append({
                    "nonclickbait_title": raw_doc["news_title"],
                    "clickbait_title": raw_doc["new_title"],
                    "clickbait_content": raw_doc["news_content"],
                })

        rag_queries.append({
            "query_title": query["news_title"],
            "query_content": query["news_content"],
            "human_direct_clickbait_title": query["new_title"],
            "retrieved_articles": retrieved_articles
        })

    # 저장
    with open(RAG_RETRIVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(rag_queries, f, ensure_ascii=False, indent=2)
    print("RAG 검색 결과가 'rag_retrieval_results.json'에 저장되었습니다!")


if __name__ == "__main__":
    corpus_list, corpus_map = build_corpus()

    bm25_retriever = create_retriver(corpus_list)

    retrieve_title(bm25_retriever, corpus_map)