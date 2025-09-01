# 클릭 유도성 기사 연구

클릭 유도성 기사 제목 생성 프로젝트 (Clickbait Article Headline Generation)
📖 강원대학교 이주혁 학생(강경필 지도교수)의 학부생 논문 **"대형 언어 모델과 검색 증강 생성 기법을 활용한 클릭 유도성 기사 제목 생성"**을 기반으로 한 공식 코드 저장소입니다.

📝 프로젝트 개요
이 프로젝트는 디지털 뉴스 환경에서 독자의 흥미를 유발하는 클릭 유도성 기사 제목(Clickbait Headlines)을 자동으로 생성하는 방법을 연구합니다.

단순히 대형 언어 모델(LLM)을 사용하는 것을 넘어, 검색 증강 생성(Retrieval-Augmented Generation, RAG) 기법을 적용하여 기존의 흥미로운 기사 제목들을 참고함으로써 더욱 자연스럽고 효과적인 제목을 생성하는 것을 목표로 합니다.

✨ 주요 특징
✌️ 두 가지 생성 방식 비교:

Direct: 기사 본문만으로 제목을 생성하는 직접 생성 방식

RAG: 기사 본문과 유사한 Top-5 기사 제목을 함께 참조하여 생성하는 검색 증강 생성 방식

🤖 최신 LLM 활용: GPT-4o와 Gemini-2.0-Flash 모델을 사용하여 제목 생성을 수행합니다.

📊 다각적인 평가: LLM 기반 자동 평가(DeepSeek-V3)와 실제 사용자 설문 평가를 통해 생성된 제목의 품질을 검증합니다.

📈 RAG의 효과 입증: 실험 결과, RAG 방식이 Direct 방식보다 더 흥미롭고(선호도 높음) 본문과 관련 있는(비정합성 낮음) 제목을 생성함을 확인했습니다.

🔧 시스템 아키텍처
본 프로젝트는 아래 그림과 같이 Direct 방식과 RAG 방식으로 나누어 제목 생성을 진행합니다.

그림 1. 제목 생성 방식 도식화

Direct 방식: LLM(GPT-4o, Gemini-2.0-Flash)에 기사 본문을 직접 입력하여 제목을 생성합니다.

RAG 방식:

입력된 기사와 가장 유사한 기존 기사 5개를 BM25 알고리즘으로 검색합니다.

입력 기사 본문 + 검색된 5개 기사 제목을 프롬프트로 구성하여 LLM에 전달합니다.

LLM이 참조 정보를 바탕으로 최종 제목을 생성합니다.

⚙️ 설치 및 실행 방법
1. 저장소 복제 (Clone)
git clone https://github.com/2weeksh/clickbait_title_generation_rag.git
cd clickbait_title_generation_rag

2. 필요 라이브러리 설치
pip install -r requirements.txt

3. API 키 설정
프로젝트 루트 디렉터리에 .env 파일을 생성하고 사용하시는 LLM의 API 키를 입력하세요.

# .env 파일 예시
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="..."

# 실험 결과
LLM 자동 평가 및 사용자 설문 평가 결과, Gemini-2.0-Flash (RAG) 모델이 가장 흥미로운 제목을 생성하는 것으로 나타났습니다. 또한 RAG를 적용한 모델들이 Direct 방식보다 더 높은 선호도와 본문 연관성을 보였습니다.
