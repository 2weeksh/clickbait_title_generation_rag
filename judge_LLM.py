import json
import random

from dotenv import load_dotenv
from litellm import completion

# API 키 설정
load_dotenv()

# 제목 셔플 시드
RANDOM_SEED = 42

# 입출력 주소
GENERATED_TITLE_PATH = "/home/joohyuk02/joohyuk/project1/output/merged_all_models_results.json"
EVALUATE_CLICKBAIT_PATH = "/home/joohyuk02/joohyuk/project1/output/evaluate_clickbait_results.json"

# 모델 config 값
MODEL = {
    "model_name": "fireworks_ai/deepseek-v3-0324",
    "judge_llm_temperature": 0.0,
    "response_format": {"type": "json_object"},
}

# 평가 모델 시스템 프롬프트
EVALUATOR_SYSTEM_PROMPT = """
당신은 흥미로운 기사 제목을 선택해서 읽는 신문 구독자입니다.

아래에 제시된 제목들에 대해 가장 흥미롭고 읽고 싶은 제목을 선택해 주세요.
여기서 제목은 무작위 순서로 제시되며, 순서 편향 없이 선택해 주세요.

반드시 아래 형식처럼 JSON으로 출력해 주세요:
{"choice": "선택한 알파벳"}
""".strip()

# 평가 모델 유저 프롬프트
EVALUATOR_USER_PROMPT = """
### 제목 후보
A: {A}
B: {B}
C: {C}
D: {D}
E: {E}
""".strip()

# 원본 제목 키와 평가용 라벨 매핑
KEY_TO_LABEL_MAP = {"human_direct": "A", "GPT_LLM": "B", "GPT_RAG": "C", "GEMINI_LLM": "D", "GEMINI_RAG": "E"}

# 평가 라벨과 모델 이름 매핑
label_to_model = {"A": "Direct_Human", "B": "GPT", "C": "RAG_GPT", "D": "Gemini", "E": "RAG_Gemini"}


# 평가 함수
def evaluation():
    # 파일 오픈
    with open(GENERATED_TITLE_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # 최종 결과 저장 리스트
    evaluation_results = []

    # 기사마다 평가
    for item in all_data:
        # 제목들 라벨과 같이 정리
        candidates_with_labels = []

        # 셔플을 위한 반복문
        for key, label in KEY_TO_LABEL_MAP.items():
            raw_title_json = item[key]
            parsed_title = safe_json_parse(raw_title_json)["title"]
            candidates_with_labels.append((label, parsed_title))

            # 평가 순서 섞기
            random.shuffle(candidates_with_labels)

            # 섞인 순서에 따라 데이터 재구성
            shuffled_candidates_for_prompt = {
                new_label: title for new_label, (_, title) in zip(["A", "B", "C", "D", "E"], candidates_with_labels)
            }
            shuffled_order = {
                new_label: label_to_model[old_label]
                for new_label, (old_label, _) in zip(["A", "B", "C", "D", "E"], candidates_with_labels)
            }

            # 랜덤 적용 프롬프트 생성
            user_prompt = EVALUATOR_USER_PROMPT.format(
                A=shuffled_candidates_for_prompt["A"],
                B=shuffled_candidates_for_prompt["B"],
                C=shuffled_candidates_for_prompt["C"],
                D=shuffled_candidates_for_prompt["D"],
                E=shuffled_candidates_for_prompt["E"],
            )
            messages = [
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            # 평가 모델 호출
            response = completion(
                model=MODEL["model_name"],
                messages=messages,
                temperature=MODEL["judge_llm_temperature"],
                response_format=MODEL["response_format"],
            )

            result_text = response.choices[0].message.content.strip()

            # 저장 양식
            evaluation_results.append(
                {
                    "index": item["index"],
                    "evaluation": result_text,
                    "shuffled_order": shuffled_order,
                    "shuffled_titles": shuffled_candidates_for_prompt,
                }
            )

    # 최종 결과 저장
    with open(EVALUATE_CLICKBAIT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)


# 메인 실행
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    evaluation()
