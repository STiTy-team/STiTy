# 진행 중 결정·시행착오 메모

- 2026-09-25 gpt-6-luna: OpenAI 모델 목록에 있음. reasoning_effort none/low/미지정 모두 동작, 미지정이면 reasoning 33토큰 → none 사용.
  단가 $0.10/$0.01/$0.50 per 1M (OpenAI 가격표). 판정기 gpt-6-sol $2/$0.2/$10.
- DeepL key 는 free(:fx) → 비용 0, 병렬 금지.
- 번역 실행기 1차 에이전트가 사용자 오클릭으로 중단 → 스모크 전 상태. 고아 스모크 프로세스(PID 1778483) 정리, 새 에이전트가 인계.
- XCOMET-XL: .env 의 HF_TOKEN 으로 gated 접근 성공. GPU 14.1 GB. COMET 뒤에 같은 프로세스로 돌리면 OOM → 모델마다 별도 프로세스.
  XCOMET 은 대명사 수 오류(it vs them)에 관대(0.984), COMET-DA 는 they/she 를 거의 구분 못 함 → 문맥 현상은 판정기 checks 로 본다.
- 판정기 프롬프트에서 인스턴스의 `expected_context_effect`("n=0 은 he 로 번역할 것" 같은 예측)를 뺐다. 판정기에 기대 오류를 미리 알려주면 그 오류를 찾아내려는 쪽으로 기울 수 있어서. JUDGE_VERSION .2.
- 판정기는 (instance, hypothesis) 고유 쌍마다 1회, 모델·조건 비공개. 6쌍 시험 $0.018.
- 2026-09-26 로컬 모델 지연 재측정은 사용자 결정으로 하지 않는다. main 단계 로컬 모델 지연은 다른 세션의 GPU 작업(seamless 서버 등)과 GPU 를 나눠 쓴 상태에서 잰 값이라 정확하지 않다고 보고서에 적는다. (시도하다 CUDA 오류로 0행에서 멈춘 재측정 결과 폴더는 지웠다.)
