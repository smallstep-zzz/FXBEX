# 실행 경로와 운영 확인

fxbX는 Torch 연산을 새로 해석하지 않는다. 공식 Furiosa-LLM 모델은 공식 빌더·런타임으로, TCLMachine의 정적 FXB는 생산자가 만든 실행 계획으로 처리한다. 이 문서는 현재 개발 소스의 API와 기존 실기 관찰을 설명한다. 수정본의 최종 소스 식별자와 실행 결과는 `docs/validation`의 실제 기록을 기준으로 한다. main 병합이나 PyPI 배포 완료를 뜻하지 않는다.

## 형식에 맞는 로더

| 파일 내용 | 실행 API | 필요한 외부 정보 |
|---|---|---|
| FXB_EX schema v1 | `fxbx.Model(path)` 또는 `fxbx.LLM(path)` | 패키지의 adapter·가중치 계약 |
| TCLMachine native 실행 계획 | `fxbx.GraphFXB(path, weights=...)` | 명시적인 가중치와 텐서 입력 |
| legacy 단일 EDF graph plan | `fxbx.GraphFXB(path, weights=...)` | 명시적인 tensor 가중치와 입력 |
| 공식 모델 FXB | `furiosa_llm.LLM(model_dir, fxb=path)` | 같은 모델의 config/tokenizer/가중치 및 공식 옵션 |

확장자를 바꾸어도 형식은 바뀌지 않는다. FXB_EX reader는 native FXB를 `NATIVE_FXB_REQUIRES_EXPLICIT_LOADER`로 거부하며 올바른 경로를 안내한다. graph plan이 없는 공식 native FXB를 `GraphFXB`에 전달하면 `NATIVE_FXB_WITHOUT_GRAPH_PLAN`로 거부한다. 필요한 모델 등록·tokenizer·가중치를 추측해 다른 엔진을 실행하지 않는다.

`hug(LLM)`은 로컬 manifest와 producer metadata를 제한된 크기로 읽는다. TCL `execution_plan`·`graph_plan` 또는 명시적 consumer 요구가 있으면 기본 자동 경로에서 `TCL_NATIVE_FXB_REQUIRES_GRAPH_LOADER`를 반환한다. 모델 인자뿐 아니라 `LLM(model_dir, fxb=path)`의 `fxb` 경로도 확인한다. 중복·손상·초과 크기 metadata를 공식 LLM으로 조용히 전달하지 않는다. 로컬 판별에서 catalog/HF 요청을 추가하지 않는다.

FXB_EX를 `.fxb`로 이름만 바꿔도 package 경로를 선택한다. 명시적인 `.fxbex` 경로는 package reader의 무결성·로더 안내를 유지한다. 공식 native FXB는 기존 공식 생성자에 전달한다. `fxbex=False`는 기존의 명시적인 native 강제 전달 옵션으로 남으며 자동 판별의 보호를 해제한다. `Model(..., fxb=...)`의 `fxb`는 package 경로의 별칭으로, native 변환 기능이 아니다.

## 생산자 검증 영수증

```python
from fxbx import GraphFXB, SafeTensorWeights

with GraphFXB(
    'model.fxb',
    weights=SafeTensorWeights('model.fxb.verified/weights'),
    qualification='model.fxb.verified/qualification.json',
) as runtime:
    outputs = runtime.predict((input_tensor,))
    print(runtime.qualification)
```

TCLMachine이 생성·수치·입력 변경·상태·호출별 NPU 증거를 검증하고 영수증을 발행한다. fxbX는 적재 시 FXB/plan/체크포인트/target 이름 매핑/실제 소비자 소스/SDK가 그 자료와 일치하는지만 확인한다. 전체 가중치 hash는 적재 승인 때 검사하고 매 추론마다 반복하지 않는다. 영수증을 전달했는데 불일치하면 장치 적재 전에 거부한다.

현재 영수증은 제공한 고정 shape fixture, `forward`, `output_device='cpu'` 범위다. 다른 entry 또는 NPU 출력 요청은 그 영수증 범위 밖이므로 거부한다. CPU 출력은 NPU 모델 계산 후 D2H이며 CPU fallback을 뜻하지 않는다. 입력 불변 검사는 제공 fixture에 대한 결과로, 모든 입력의 alias·mutation 안전성을 증명하지 않는다.

영수증 없는 raw 적재는 `unqualified`다. 성공적으로 로드하거나 호출한 것만으로 `verified`로 승격하지 않는다. 같은 shape/dtype이라도 가중치 값이나 target 매핑이 바뀌면 이전 qualification을 승계하지 않는다. 자세한 경계는 [생산자 검증 사용법](VERIFIED_BUILD.md)을 따른다.

## 세션 수명과 완료 불명

legacy 및 현대 정적 로더는 세션의 적재·실행·종료를 재진입 락으로 직렬화한다. 업로드된 가중치·입력과 EDF 출력은 shape/dtype/stride뿐 아니라 Furiosa 장치 계약을 보존해야 한다. CPU tensor가 메타데이터만 같다고 NPU 성공으로 취급하지 않는다. 부분 적재 중인 Runnable도 이동 전에 세션 소유 대상으로 기록한다.

장치 lease를 확보한 뒤 fusion/mesh 설정과 EDF 준비를 수행한다. lease가 busy면 명시적 SDK 설정 변경·장치 적재로 진행하지 않는다. 스케줄러가 전달한 lease FD는 검증하여 child에 명시적으로 상속하고, child 종료가 부모의 원래 lease를 해제하지 않게 한다. 같은 inherited lease 아래 여러 worker를 병렬 실행해도 서로 배제된다고 가정하면 안 된다.

각 제출 구간에서 공개 current stream을 캡처하고, 다른 스레드에서 close하더라도 그 제출 stream의 완료를 확인한다. 락은 같은 세션의 호출을 직렬화하며 장치 병렬 실행 기능이 아니다.

EDF 실패 뒤 정리도 실패하면 최초 적재·실행 오류를 유지하고 정리 오류를 cause로 연결한다. context 종료 실패 뒤에도 닫힌 stack이 open처럼 남지 않게 하고, request handle과 tensor 참조를 정리한다. Python 필드를 비웠다는 사실과 장치 완료는 구분한다.

완료 동기화가 실패하면 `completion_known`은 false가 되고, 같은 프로세스의 해당 SDK를 사용하는 다음 native 적재·실행은 `NATIVE_COMPLETION_UNKNOWN`으로 거부한다. worker 재시작과 별도 장치 상태 확인이 필요하다. 프로세스 종료나 SMI memory 0만으로 장치 복구를 확정하지 않는다. 같은 실패 호출을 자동 재실행하거나 실패 세션을 reset해 정상으로 간주하지 않는다.

state commit 경계는 단일 `run(entry, ...)`다. 여러 `prefill → decode` 호출 전체를 되돌리는 트랜잭션은 제공하지 않는다.

## FXB_EX worker 제한 시간과 취소

`Model`, `LLM` 및 package facade는 한 worker에 한 요청을 보낸다. 요청 진입 시 만든 절대 monotonic deadline을 락 대기·stdin write·stdout read에 공유한다. facade batch도 같은 deadline을 공유한다. token·progress·일반 로그가 계속 와도 시간을 연장하지 않는다.

기존 host 설정 `~/.config/fxbex/config.json`의 다른 필드를 보존하며 다음 항목을 병합할 수 있다.

```json
{
  "worker_timeouts": {
    "load": 600,
    "request": 300,
    "unload": 30,
    "terminate": 15,
    "kill": 5
  }
}
```

각 값은 초 단위의 유한 양수다. 모델 패키지가 이 값을 바꾸지는 못한다. timeout 뒤 terminate/kill/wait 정리 시간이 추가될 수 있으므로 API 반환 시각은 요청 deadline과 다르다.

transport는 binary nonblocking pipe에서 partial write·부분 UTF-8·newline 없는 응답과 frame 크기를 처리한다. worker가 살아 있어도 응답이 없으면 제한 시간 내 요청을 중단한다. queue에서 만료된 요청은 다른 요청이 사용 중인 worker를 종료하지 않는다.

다른 제어 스레드에서 `model.cancel()`을 호출하면 현재 요청의 취소 event를 설정한다. async facade stream도 자기 요청의 event만 취소한다. 오래된 stream 취소가 새 요청을 종료하지 않게 한다. 요청을 소유한 worker는 terminate, 제한된 wait, 필요 시 kill과 추가 wait로 종료를 회수한다. 종료를 확인하지 못하면 handle을 유지하고 새 load를 거부한다. 종료 실패도 최초 timeout/cancel 오류를 가리지 않는다.

사용자 `on_progress`/`on_token` callback은 짧게 반환해야 한다. 같은 스레드에서 무기한 막힌 사용자 callback까지 transport deadline만으로 중단할 수 있다고 보장하지 않는다. 직접 호출하는 native `GraphFXB`에 timeout/cancel worker API를 추가한 것은 아니다. 강제 중단이 필요한 native 검증은 모델별 subprocess로 실행한다.

## 타이밍 수집과 호스트 메모리

**Python `Profiler` 인스턴스를 만들지 않았다는 사실은 SDK 타이밍 수집이 꺼졌다는 뜻이 아니다.** 이 환경의 SDK 2026.4.0에서 기본 `info` 타이밍 수집과 명시적인 `warn` 설정을 별도 프로세스로 비교했다. 고정 cascade를 32회 재적재한 대조의 RSS 증가는 기본 수집에서 219,910,144 bytes(약 209.72MiB), `warn`에서 14,860,288 bytes(약 14.17MiB)였다. 이 차이는 환경별 관찰이며 SDK 전체의 누수 원인이 모두 해결됐다는 증거는 아니다. 상세 입력·호출 횟수·최종 판정은 해당 실행 JSON을 기준으로 한다.

NPU trace 증거가 필요 없는 일반 작업은 SDK를 import하기 전 새 프로세스에서 공개 환경 변수를 설정하는 운영을 권장한다.

```bash
FURIOSA_TORCH_PROFILE=warn python app.py
```

이미 SDK를 import한 프로세스에서 환경 변수만 바꾸어 동일 효과를 가정하지 않는다. `warn` 실행에서 수치나 장치 배치만 확인하고 NPU Task가 수집됐다고 보고하지 않는다. TCLMachine 검증 worker는 import 전에 `FURIOSA_TORCH_PROFILE=info`를 명시하고 실제 호출별 NPU Task를 요구한다. 메모리를 줄이기 위해 검증 증거 조건을 완화하지 않는다.

여러 모델을 순환하는 일반 검증 작업은 모델 하나를 load하고 여러 호출·결과 저장을 마친 뒤 worker 종료를 회수하는 방식으로 운영한다. 매 inference마다 새 프로세스를 만들지는 않는다. FXB_EX `PackageService`와 달리 raw native `GraphFXB` 자체가 subprocess인 것은 아니므로 작업 실행기가 명시적인 경계를 둬야 한다.

부모와 worker의 RSS를 따로 기록한다. worker 격리로 부모 누적을 제한한 결과와 동일 프로세스 재적재의 근본 원인 해결을 구분한다. 이번 수정본의 새 worker 격리 대조는 최종 실행 결과로 확인해야 하며, 이 문서만으로 통과를 주장하지 않는다.

## 기존 실기 근거와 남은 범위

[이전 반복 실행 결과](validation/runtime-stress-20261110.json)의 작은 고정 `[4,8]` fixture는 128단계 반복 EDF cascade와 prefill/decode에서 16,384회 수치 비교를 끝냈다. 해당 수집 조건에서 워밍업 이후 RSS 증가 약 8.52MiB, 종료 후 NPU 증가 0을 기록했다. 이 숫자만으로 메모리 문제가 해결됐다고 판단하면 안 된다. [후속 메모리 대조](validation/runtime-memory-followup-20261110.json)에서는 같은 프로세스의 반복 적재 증가가 별도로 드러났다.

이전 반복 Profiler 시작·종료에서는 48개 샘플 중 12개가 필수 이벤트 조건에 실패했다. 수치가 맞아도 `runtime_numerical_passed_evidence_incomplete`로 보존했다. 기본 stress 모드는 strict이고 `--continue-on-trace-loss`는 나머지 검사를 계속할 뿐 최종 성공을 반환하지 않는다.

TCLMachine 검증 worker는 이제 제한된 batch를 하나의 Profiler로 감싸고 호출별 고유 span의 시간 범위 안에서 각 step과 NPU Task를 대조한다. 이름을 전체 trace에서 한 번만 모아 다른 호출의 증거로 누락을 가리지 않는다. 명시적 `static_value_preparation`만 Task 면제를 허용하며, zero-input `npu_compute`에는 Task가 필요하다. 이 구현의 최종 수용 결과는 해당 수정본의 실행 자료로 확인한다.

`probe_legacy_runtime.py`의 이전 실기 10회는 CPU/NPU 출력 요청, 수치 비교와 NPU Task를 확인했고 CPU route는 0이었다. fake SDK를 이용한 오류 회귀 검사는 이 실기와 구분한다.

논리 메모리 예산은 물리 HBM 피크가 아니다. EDF workspace·SDK layout/padding·예약과 다른 점유 작업을 모두 증명하지 않는다. 미사용 큰 tensor 선언은 실제 가중치 상주·연산 증거가 아니며, 작은 반복 그래프 통과를 9B 모델 지원으로 표시하지 않는다. 공식 Furiosa-LLM과 named multi-entry의 qualification도 현재 single-forward 영수증과 별도다.

```bash
# SDK 환경과 수정본 소스에서 사용법을 확인한다. 기존 결과는 덮어쓰지 않는다.
python tests/hardware/stress_execution_program.py --help
python tests/hardware/stress_request_threads.py --help
python tests/hardware/probe_session_profile.py --help
python tests/hardware/probe_legacy_runtime.py --help
```

## CI와 배포

테스트와 배포 workflow는 기존 `weights` extra로 safetensors를 설치한다. CPU-only Torch와 NumPy, Python matrix 및 전체 unittest를 유지한다. CI에서 테스트를 생략하거나 skip으로 성공시키지 않는다. 소스의 수정·실기·main 병합·PyPI 배포는 서로 다른 단계이며, 배포는 기존처럼 명시적인 workflow_dispatch로 수행한다.
