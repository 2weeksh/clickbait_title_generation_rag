import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

# 인자 값
GENERATE_TEMPERATURE = 1.0

# 입,출력 경로
INPUT_NEWS_PATH = "/home/joohyuk02/joohyuk/project1/output/rag_retrieval_results.json"
GENERATED_TITLE_PATH = "/home/joohyuk02/joohyuk/project1/output/merged_all_models_results.json"

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 온라인 뉴스 콘텐츠를 위한 주목을 끌도록 낚시성 제목을 잘 작성하는 헤드라인 전문가입니다.
주어진 기사 제목과 본문을 바탕으로 사람들의 호기심을 자극하고 클릭을 유도할 수 있는 새로운 제목을 만들어야 합니다.
다만, 제목은 자극적이고 끌리는 방식으로 바꾸되 주제와 관련성을 유지해야 합니다.
생성 형식은 {"title": str} 형식의 JSON 객체 하나만 생성해 주세요.
""".strip()

# LLM프롬프트
NAIVE_LLM_USER_PROMPT = """
### 입력 기사
제목:{query_title}
본문:{query_content}
""".strip()

# RAG 프롬프트
RAG_USER_PROMPT = """
### 입력 기사
제목:{query_title}
본문:{query_content}

### 참고 예시
아래는 위 기사와 관련된 기사들에 대해 사람들의 관심을 끌 수 있도록 재작성한 제목 예시입니다.

{retrieved_title}
""".strip()


def generate_title(models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # 최종 결과 저장 리스트
    results = []

    # 기사 파일 열기
    with open(INPUT_NEWS_PATH, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    # 기사마다 반복
    for i, news in enumerate(rag_data):
        merge_title = {
            "index": i,
            "query_title": news["query_title"].strip(),
            "query_content": news["query_content"].strip(),
            "human_direct": news["human_direct_clickbait_title"].strip(),
        }

        query_title = news["query_title"].strip()
        query_content = news["query_content"].strip()
        retrieved_articles = news["retrieved_articles"]

        # RAG 참고 기사 정리
        retrieved_title = ""
        for retrieved in retrieved_articles:
            clickbait_title = retrieved["clickbait_title"].strip()
            retrieved_title += f"- 낚시성 기사 제목: {clickbait_title}\n"

        retrieved_title = retrieved_title.strip()

        # 모델 설정
        for model in models:
            model_name = model["model"]
            use_rag = model["use_rag"]
            output = model["output"]

            if use_rag is True:
                USER_PROMPT = RAG_USER_PROMPT.format(
                    query_title=query_title, query_content=query_content, retrieved_title=retrieved_title
                )
            else:
                USER_PROMPT = NAIVE_LLM_USER_PROMPT.format(query_title=query_title, query_content=query_content)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]

            response = completion(
                model=model_name,
                messages=messages,
                temperature=GENERATE_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            clickbait_title = response["choices"][0]["message"]["content"].strip()
            merge_title[output] = clickbait_title

        results.append(merge_title)

    return results


# 메인 실행
if __name__ == "__main__":

    # 모델 config 값
    models = [
        {"model": "openai/gpt-4o", "use_rag": False, "output": "GPT_LLM"},
        {"model": "openai/gpt-4o", "use_rag": True, "output": "GPT_RAG"},
        {"model": "gemini/gemini-2.0-flash-001", "use_rag": False, "output": "GEMINI_LLM"},
        {"model": "gemini/gemini-2.0-flash-001", "use_rag": True, "output": "GEMINI_RAG"},
    ]

    results = generate_title(models)

    # 생성 결과 저장
    with open(GENERATED_TITLE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
