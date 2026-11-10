# 생산자가 검증 완료를 책임지는 FXB 생성

TCLMachine이 EDF 생성부터 기준 결과, 독립 실행 및 NPU 증거 검증을 책임진다. fxbX는 생산자의 검증 의미를 다시 구현하지 않고, 저장된 실행 계획을 실행하며 선택적으로 검증 영수증의 현재 조합을 확인한다. 아래는 현재 개발 소스 API다. 실제 설치본의 기능과 이번 후보의 최종 검증 결과는 별도 기록을 확인한다.

## 생성과 검증

TCLMachine의 `build_verified_model`이 있는 호환 소스 환경에서 호출한다.

```python
from fxbx import build_model

report = build_model(
    model, './model.fxb', example_inputs=(x,), concurrency=2,
    verification={
        'validation_inputs': [(x + .5,)],
        'atol': .02,
        'rtol': .02,
        'repeats': 3,
        'replay_timeout': 600,
    },
)
assert report['status'] == 'verified'
```

모델은 frozen eval 상태여야 하고 입력·상태·출력은 생산자가 지원하는 고정 계약을 따라야 한다. 적어도 하나의 실제로 바뀐 validation input을 제공한다. tolerance는 검증 전에 명시한다. 성공시키기 위해 실패 뒤 허용 오차를 넓히지 않는다.

생산자는 FXB와 가중치·fixture hash를 기록하고, producer import와 compile 호출을 금지한 새 프로세스에서 fxbX로 재적재한다. 출력 shape/dtype·수치·상태/reset·입력 불변과 호출별 NPU trace를 확인한 뒤 자료가 바뀌지 않았는지 재확인한다. 성공 뒤에만 최종 FXB를 공개한다. 가중치, 기준값, replay와 `qualification.json`은 `model.fxb.verified/`에 보존한다.

`verification`을 생략한 기존 `build_model`은 `built_not_runtime_validated` 의미를 유지한다. 공식 Furiosa-LLM 등록 모델과 legacy contract provider는 이 Torch 전용 verification 옵션을 거부한다. 파일 생성·공식 형식 검사 성공을 수치·NPU 검증 완료로 바꾸지 않는다.

## 검증 영수증을 사용한 재적재

```python
from fxbx import GraphFXB, SafeTensorWeights

with GraphFXB(
    './model.fxb',
    weights=SafeTensorWeights('./model.fxb.verified/weights'),
    qualification='./model.fxb.verified/qualification.json',
) as runtime:
    print(runtime.qualification['status'])
    outputs = runtime.predict((x,))
```

위 예시는 외부 가중치가 있는 모델 기준이다. 외부 가중치가 없는 모델은 `weights`를 생략한다. `qualification=`을 전달하면 fxbX는 적재 시 다음 자료를 확인한다.

- receipt schema/status/route와 검증 scope.
- FXB 및 canonical execution-plan SHA256.
- weight shard·index, target→checkpoint 이름 매핑.
- 실제 import한 fxbX 소비자 소스 지문과 버전. 같은 버전 문자열만으로 일치로 보지 않는다.
- Torch·Furiosa package 버전, compiler identity 및 device configuration.

불일치는 장치 적재 전에 오류로 반환한다. 전체 가중치 hash는 적재 승인 때 검사하며 매 `run`마다 반복하지 않는다. 신뢰된 생산자가 만든 로컬 자료를 전제로 하며, hash는 작성자 인증이나 전자서명이 아니다. 영수증을 읽다가 임의 모듈·명령을 실행하거나 모델을 다운로드하지 않는다.

strict qualification 경로의 외부 가중치는 기록된 `SafeTensorWeights` 체크포인트를 사용한다. 임의 dict tensor가 같은 artifact인지 추측하지 않는다. 같은 shape/dtype의 다른 값이나 이름 매핑을 사용하면 이전 qualification을 승계하지 않는다.

## 현재 검증 범위

현재 receipt는 `tcl_static`, 제공한 고정 shape fixture, `forward`, `output_device='cpu'` 범위다. 영수증이 있는 세션에서 다른 entry나 NPU 출력 장치를 요청하면 범위 밖 오류를 낸다. CPU 출력은 NPU 연산 뒤 결과 복사이며 CPU 모델 fallback이 아니다.

입력·weight·state의 불변성은 제공 fixture에서 검사한다. 모든 가능한 입력의 alias·mutation 안전성을 증명했다는 뜻은 아니다. state commit 보장은 생산자의 functional 입력 계약 아래 단일 entry 호출을 대상으로 하며 여러 `prefill/decode` 호출 전체의 rollback을 제공하지 않는다.

영수증 없이 사용하는 기존 raw `GraphFXB`는 `qualification['status'] == 'unqualified'`다. raw load·실행 성공에 자동으로 verified를 붙이지 않는다. native 실행 계획용 receipt를 legacy graph 또는 공식 LLM에 적용하지 않는다. named multi-entry와 공식 모델의 tokenizer/KV/bucket/생성 조건 검증은 별도 범위다.

## NPU 증거와 SDK 타이밍 설정

검증 worker는 SDK import 전에 `FURIOSA_TORCH_PROFILE=info`를 명시한다. 작은 batch 전체를 하나의 공식 Profiler로 감싸고 각 호출의 고유 span 안에 해당 step span과 실제 NPU Task가 있는지 검사한다. 전체 trace에서 같은 이름을 한 번 발견했다고 모든 호출이 통과하지는 않는다.

`static_value_preparation`으로 명시된 zero-input 준비 단계만 Task 부재를 허용한다. 준비 단계도 host span과 수치 조건이 필요하고 NPU compute 성공으로 집계하지 않는다. zero-input이어도 `npu_compute`이면 실제 Task가 필요하다. CPU route나 필수 이벤트 누락은 수치 성공과 별도로 실패다. 이미 상태를 바꾼 호출을 같은 세션에서 몰래 재시도하지 않는다.

일반 작업에서는 새 프로세스의 SDK import 전에 `FURIOSA_TORCH_PROFILE=warn`을 사용하는 운영이 호스트 타이밍 누적을 줄인 대조 결과가 있다. `Profiler` 객체를 만들지 않는 것만으로 수집이 꺼지지는 않는다. 이 설정을 검증 worker에 사용해 NPU 증거 조건을 낮추지 않는다. 메모리·worker 수명·기존 실기와 남은 한계는 [운영 확인 문서](RUNTIME_READINESS.md)를 참고한다.
