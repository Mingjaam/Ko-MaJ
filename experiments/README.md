# 실험 재현

논문의 모든 표와 그림을 만든 파이프라인입니다. GPU 생성은 vLLM 0.29, 기존 기법(SynthID-Text · LUNA · STELA)은 각 공식 저장소의 HuggingFace 구현을 사용했습니다. 데이터와 생성 결과물은 저장소에 포함하지 않습니다(위키백과 CC BY-SA, 뉴스 데이터셋 라이선스).

## 환경 변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `KOMAJ_OUT` | `../out` | 생성·공격·검출 결과 |
| `KOMAJ_DATA` | `../data` | 프롬프트와 사람 글 |
| `KOMAJ_BIG_MODEL` | `Qwen3.8-27B-FP8` | 공격 모델 겸 PPL 측정 모델(로컬 경로 또는 HF ID) |
| `KOMAJ_BASELINES` | `../baselines` | LUNA·STELA 공식 저장소를 받아 둔 위치 |

## 순서

```bash
cd experiments
python prep_data.py ../data                 # 위키백과: dev 200 / test 500 / human 3,000
python prep_news.py ../data                 # 뉴스 도메인: news_test 500 / news_human 3,000

# 1) 생성 (모델 4종 × 설정): none, KGW δ1~3, Unigram δ1~3, Ko-MaJ δ1.5~4, h2/h2R
python gen.py --model Qwen/Qwen2.5-7B-Instruct --name qwen25_7b --split test --configs configs/main.json --out ../out/gen_main
python gen.py --model Qwen/Qwen2.5-7B-Instruct --name qwen25_7b --split test --configs configs/rep.json  --out ../out/gen_main
python gen.py --model Qwen/Qwen2.5-7B-Instruct --name qwen25_7b --split test --configs configs/keys.json --out ../out/gen_main
python gen.py --model Qwen/Qwen2.5-7B-Instruct --name qwen25_7b --split news_test --configs configs/news.json --out ../out/gen_main
python baseline_hf.py --alg STELA --model Qwen/Qwen2.5-7B-Instruct --name qwen25_7b      # SynthID / LUNA / STELA (공식 코드)

# 2) LLM 재작성 공격과 품질(PPL)
python llm_attack.py --names qwen25_7b --split test --tags none,kgw_d2,han_d3,han_d2.5,han_d2,kgw_d3 --attacks style,paraphrase,backtrans
python ppl.py --names qwen25_7b --split test --out ../out --gen_sub gen_main

# 3) 검출 (z 점수만 저장): 문자 수준 · 음절 편집 · LLM 공격, 방어 없음 / 정규화 / 띄어쓰기 복원
python det_rev.py --name qwen25_7b --tok Qwen/Qwen2.5-7B-Instruct --split test \
  --attacks clean,nospace,randspace,zwsp,nfd,nfdzw,jamo,sub10,sub20,del10,del20,excerpt,mix,style,paraphrase,backtrans --defenses none,norm,space
python det_rev.py --name qwen25_7b --tok LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --attacks clean --label cross_exaone_8b   # 교차 토크나이저
python detbl_rev.py --alg STELA --model Qwen/Qwen2.5-7B-Instruct --name qwen25_7b --lens 200 --attacks clean,nospace  # 기존 기법 (GPU)

# 4) 평가와 집계
python eval_rev.py       # 사람 글 보정 1,500 / 시험 1,500, TPR_elig · TPR_all, 신뢰구간, 짝지은 부트스트랩, 같은 품질 보간
python collect_rev.py    # → out/final2.json
```

## 검증 스크립트

- `test_exact.py`: 생성 측 bias(h2/h2R 포함)가 검출 규칙으로 직접 만든 참조 구현과 모든 후보 토큰에서 일치하는지 확인합니다.
- `stats_impl.py`: UTF-8 미완성 단계(상위 64개 후보 근사)와 분해 자모 출력이 실제로 얼마나 나왔는지 셉니다.

## 평가 규칙

- 탐지율은 앞 200 정규화 음절로 잽니다. 200음절에 도달한 출력 기준(`tpr_elig`)과 전체 500개 기준(`tpr_all`)을 함께 기록합니다.
- 임계값은 사람 글 1,500개(보정)의 99번째 백분위수, 오탐률은 나머지 1,500개와 워터마크 없는 모델 출력(같은 공격 적용)에서 잽니다.
- 기존 기법에는 공격받은 원문을 정규화 없이 그대로 넣습니다. 방어(`norm`, `space`)는 별도 조건으로 기록합니다.
