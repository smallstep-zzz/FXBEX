# 실행 경로와 운영 확인

fxbX는 Torch 연산을 새로 해석하지 않는다. 공식 Furiosa-LLM 모델은 공식 빌더/런타임 경로로, TCLMachine의 정적 FXB는 그 실행 계획을 읽는 경로로 처리한다. 운영 점검에서는 파일 생성과 실제 실행을 구분한다.

## 형식을 구분해 로드

확장자를 바꾸어도 형식은 바뀌지 않는다. FXB_EX 패키지 reader는 manifest를 읽고 native FXB v2라면 `NATIVE_FXB_REQUIRES_EXPLICIT_LOADER`와 올바른 호출 경로를 안내한다. 필요한 설정·토크나이저·가중치를 임의로 추측해 다른 엔진을 실행하지 않는다. 기존 `Model(..., fxb=...)`의 fxb 인자는 경로의 별칭일 뿐 native FXB 자동 변환을 의미하지 않는다.

| 파일 내용 | 올바른 실행 API | 외부 정보 |
|---|---|---|
| FXB_EX schema v1 | `fxbx.Model(path)` 또는 `fxbx.LLM(path)` | 패키지의 adapter/가중치 계약 |
| TCLMachine native 실행 계획 / legacy graph plan | `fxbx.GraphFXB(path, weights=...)` | 명시적인 가중치와 텐서 입력 |
| 공식 모델 FXB | `furiosa_llm.LLM(model_dir, fxb=path)` | 같은 모델의 config/tokenizer/가중치 및 공식 옵션 |

```python
from fxbx import GraphFXB, SafeTensorWeights
with GraphFXB('model.fxb', weights=SafeTensorWeights('checkpoint')) as runtime:
    outputs = runtime.predict((input_tensor,))
```

`GraphFXB`에 graph plan이 없는 native FXB를 전달하면 `NATIVE_FXB_WITHOUT_GRAPH_PLAN`로 거부한다. 공식 LLM으로 변환하는 adapter나 모델 등록 정보를 만들지는 않는다. 파일 손상, 잘못된 ABI, SDK 불일치의 기존 검사는 그대로 수행한다.

## legacy 경로의 실패 처리

legacy 단일 EDF 경로도 업로드된 가중치·입력과 EDF 출력이 Furiosa 장치에서 계약을 보존하는지 확인한다. CPU tensor가 shape/dtype만 같다고 성공으로 취급하지 않는다. EDF 실패 뒤 정리/동기화가 추가로 실패해도 최초 적재·실행 오류를 유지하고 정리 오류를 cause로 연결한다. 정리할 때 resident tensor, program 및 NPU lock context를 해제한다. 현대 ExecutionFXB의 엄격한 정책과 맞춘 것이다.

## 반복 실행과 메모리 검증

`tests/hardware/stress_execution_program.py`의 사용법과 실제 측정은 해당 스크립트 및 `docs/validation` 결과를 참고한다. 이 시험은 공개 API를 통해 기존에 검증한 EDF를 재사용하고, 새 Torch 연산 지원을 주장하지 않는다. 고정 fixture에서 입력/상태·reset·다시 적재·가중치 검사·잘못된 요청·공식 trace 및 RSS/HBM 추이를 확인한다.

논리 메모리 예산이 물리 HBM 피크를 증명하지는 않는다. 전체 실제 모델의 EDF를 받으면 그 모델에 대한 생산자 소유의 `build_verified_model` 검증과 별도의 규모/메모리 시험을 계속해야 한다. 작은 모델·반복 그래프의 통과를 9B 모델 지원으로 표시하지 않는다.

## CI

테스트와 배포 workflow는 기존 `weights` extra를 사용해 safetensors를 함께 설치한다. CPU-only Torch와 NumPy, Python 버전 matrix 및 전체 unittest는 유지한다. CI에서 테스트를 생략하거나 skip으로 통과시키지 않는다. 배포는 기존처럼 명시적인 workflow_dispatch에서만 수행한다.
