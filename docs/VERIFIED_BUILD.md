# 생산자가 검증 완료를 책임지는 FXB 생성

TCLMachine의 `build_verified_model`이 있는 소스 환경에서 다음과 같이 호출한다.

```python
from fxbx import build_model
report = build_model(
    model, './model.fxb', example_inputs=(x,), concurrency=2,
    verification={'validation_inputs': [(x + .5,)], 'atol': .02, 'rtol': .02,
                  'repeats': 3, 'replay_timeout': 600},
)
assert report['status'] == 'verified'
```

fxbX는 검증 의미를 구현하지 않고 이 요청을 TCLMachine에 그대로 위임한다. 생산자가 EDF 생성, FXB 포장, 기준 결과 계산, 새 프로세스에서 기존 fxbX를 통한 컴파일 없는 재적재와 수치·상태·NPU trace 검증을 끝내야 성공한다. 가중치와 검증 증거는 `model.fxb.verified/`에 보존한다. 상세 조건·범위는 TCLMachine의 `docs/VERIFIED_BUILD.md`를 따른다.

`verification`을 생략한 기존 `build_model`은 `built_not_runtime_validated` 의미를 유지한다. 공식 Furiosa-LLM 등록 모델 경로와 legacy contract_provider는 이 Torch 전용 검증 옵션을 거부하며, 지원하지 않는 검증을 실행했다고 표시하지 않는다. 낮은 수준의 실행 및 적재 시 계약/환경/무결성 검사는 기존 fxbX의 책임으로 유지된다.
