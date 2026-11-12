# fxbX

**0.8.1 개발 소스:** [TCLMachine 공통 실행 계약](docs/EXECUTION_PLAN.md)으로 다중 EDF와 상주 상태를 연결합니다. 모델명 분기 없이 연산 계약으로 내보내며, 컴파일러 없는 새 프로세스 NPU 재적재를 검증했습니다. 구형 `fxbex`가 새 구현을 가리면 명확한 오류로 중단합니다. PyPI0.7.0과 구분하세요.

**FXBEX에서 fxbX로 이름이 바뀌었습니다.** 배포 패키지·CLI·공개 import는 `fxbx`이며, 이름 전환 릴리스는 0.7.0입니다.
새 fxbX 패키지에는 `.fxbx` 확장자를 사용합니다. 기존 `.fxbex` 모델 파일은 계속 읽을 수 있습니다. [PyPI 0.7.0](https://pypi.org/project/fxbx/0.7.0/)에서 배포합니다.
설치는 `python -m pip install --upgrade fxbx`를 사용하세요.
기존 `fxbex` 배포와 엔진 이름 공간을 공유하므로, 같은 환경으로 이전할 때는 기존 배포를 먼저 제거해야 합니다. 실행 중인 서비스가 있다면 별도 환경에서 이전을 검증하세요.

Furiosa RNGD에서 사전 컴파일한 실행 파일을 패키징하고 실행하는 Linux CLI입니다. `.fxbx`와 기존 `.fxbex`는 fxbX가 읽는 패키지입니다. Furiosa 공식 번들은 `.fxb`를 사용합니다. TCLMachine 커스텀 실행 계획도 `.fxbx`로 저장할 수 있으며, 이 산출물은 `ExecutionFXB`로 엽니다.

## Python 및 Furiosa SDK 호환성

FXBEX 코어는 Python 3.10–3.14를 지원합니다. Furiosa SDK 2026.4 문서의 Python 지원 범위도 3.10–3.14입니다. 실제 NPU 실행에는 대상 호스트에 설치된 SDK, PyTorch, 모델 어댑터와 호환되는 패키지가 필요합니다.

실행 계획은 Python의 마이너 버전이 달라도 재사용할 수 있습니다. 재사용 시 운영체제, PyTorch 버전, Furiosa SDK 버전과 SDK 소스 지문은 계속 일치해야 하며, 요청 그래프도 기록된 실행 계획과 정확히 일치해야 합니다. 실행 계획이 기록되지 않았거나 요청 그래프가 달라지면 런타임은 컴파일을 시도하지 않고 오류를 반환합니다. Python 주 버전이 다르면 재사용하지 않습니다.

## 설치

```bash
python -m pip install fxbx
fxbx --help
```

체크아웃한 소스로 개발할 때는 다음을 사용합니다.

```bash
python -m pip install -e .
```

기존 SDK 패키지를 바꾸지 않고 설치하려면 사용할 인터프리터를 지정합니다.

```bash
PYTHON=python3.11 bash scripts/install.sh
```

이 패키지는 Furiosa SDK, PyTorch, 드라이버를 설치하거나 업그레이드하지 않습니다. 빌드, 검사, 무결성 전용 검증은 표준 라이브러리만 사용하지만 NPU 실행에는 모델과 호환되는 SDK 및 의존성이 필요합니다. 현재 배포는 [fxbx 0.7.0](https://pypi.org/project/fxbx/0.7.0/)입니다. 재현 가능한 설치에는 `python -m pip install fxbx==0.7.0`을 사용하세요. SDK가 지원되는 Python 환경에서 실행하세요.

소스 버전 0.6.8부터 패키지를 추출하면서 각 파일의 SHA-256을 동시에 검증합니다. 모든 검증이 끝나기 전에는 모델을 실행하지 않습니다. 설치된 PyPI 버전에 이 변경이 있는지는 해당 릴리스를 확인하세요.

TPLMachine/FHA 최적화 후보는 실제 수치·NPU trace·반복 실행 검증을 거친 뒤 새 패키지로 만듭니다. FXBEX가 실행 중 임의의 스케줄이나 커널을 선택하지는 않습니다. 현재 소스에서는 실제 적재에도 정적 preflight를 적용하고, `generate` 실행 기록을 남기며, 일반 DAG의 정수·불리언 출력 타입을 보존합니다. 변경 범위와 아직 검증하지 않은 항목은 [연계 검토](docs/TPLMACHINE_REVIEW.md)를 참고하세요.

## 명령어

```bash
fxbx build ./compiled_model -o model.fxbx
fxbx build ./compiled_model -o model-standalone.fxbx --standalone
fxbx inspect model.fxbx
fxbx validate model.fxbx
fxbx run model.fxbx
fxbx chat model.fxbx
fxbx serve model.fxbx --port 8000
fxbx doctor
```

`run`은 채팅 어댑터를 실행합니다. 텐서 모델은 `--input inputs.json -o result.json`을 받습니다. `chat`에서는 `/reset`, `/quit`을 사용할 수 있습니다. `serve`는 채팅 어댑터용 비스트리밍 `/v1/chat/completions`와 텐서 어댑터용 `/predict`를 localhost에 제공합니다. `fxbx ui --port 7861`은 선택 기능이며 Gradio가 필요합니다. 모든 명령은 `--help`를 지원합니다.

### OpenAI Python 클라이언트와 사용하기

```bash
fxbx serve Qwen3.5-9B-FP8-dynamic --port 8000
python -m pip install openai
```

다른 터미널에서 실행합니다. `model`은 서버가 로드한 패키지의 원본 모델 ID입니다.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="RedHatAI/Qwen3.5-9B-FP8-dynamic",
    messages=[{"role": "user", "content": "안녕!"}],
    max_tokens=32,
)
print(response.choices[0].message.content)
```

현재 서버는 단일 모델·비스트리밍 채팅과 greedy 생성만 지원합니다. 지원하지 않는 생성 옵션은 오류로 반환합니다. 공식 Furiosa 서버의 모든 엔드포인트와 기능을 제공하는 것은 아닙니다. 패키지의 정밀도는 생성 시 고정되며 명시한 `dtype`이 패키지의 검증된 메타데이터와 다르면 오류가 발생합니다.

`validate`는 체크섬, 외부 가중치, SDK 버전, 어댑터 계약과 RNGD 장치를 확인하지만 **추론을 실행하지 않습니다.** SDK나 NPU가 없는 시스템에서는 다음처럼 무결성만 검사할 수 있습니다.

```bash
fxbx validate model.fxbx --integrity-only
```

`--weights-dir /path/to/snapshot`은 해시 검증을 유지하면서 외부 가중치 위치를 지정합니다.

빌더는 이미 컴파일된 아티팩트를 패키징하며 임의의 Hugging Face 모델을 컴파일하지 않습니다. 실행 파일이 없거나 SDK와 맞지 않으면 오류를 반환하고 자동 재컴파일하지 않습니다. 신뢰할 수 있는 패키지와 설치된 플러그인만 사용하세요.

## Model Forge 연동

FXBEX 코어는 `compiler_engine`에 의존하지 않습니다. 일반 `runnable_dag` 패키지는 코어 런타임에서 실행됩니다. 모델별 코드는 별도로 설치하는 어댑터 플러그인에 둡니다.

```bash
cd /root/rngd-model-forge
bash tests/install_fxbex.sh /root/FXBEX
```

설치 스크립트는 `fxbex.adapters` 진입점을 통해 `qwen35_replay` 플러그인을 설치하고, 기존 Model Forge 웹 작업자와 공유하는 NPU 잠금 경로를 `~/.config/fxbex/config.json`에 기록합니다. Model Forge의 **FXBEX package** 탭은 설치된 런타임을 별도 프로세스로 호출하며, 모델을 내리면 NPU 리소스도 반환합니다.

플러그인 클래스는 `(extracted_root, report_dir, ExitStack, weights_dir=None)`를 받고 `chat` 및/또는 `predict`를 제공합니다. `preflight(root, weights_dir=None)`는 정적 검사를 수행합니다. 패키지는 임의의 import 경로나 내장 Python 프로그램 대신 설치된 진입점 이름을 지정합니다. 필요한 진입점이 없거나 중복되면 명시적으로 실패합니다.

기본 보고서 경로는 `~/.cache/fxbex/runs`입니다. `~/.config/fxbex/config.json`의 `reports_dir`, `npu_lock` 설정으로 변경할 수 있으며 `FXBEX_CONFIG`, `FXBEX_NPU_LOCK` 환경 변수도 사용할 수 있습니다. Model Forge 연동 설치기는 기존 잠금 파일을 공유합니다.

## 범위

현재 내장 런타임은 RNGD fusion8의 고정 텐서 DAG를 지원합니다. Qwen 플러그인은 BF16 텍스트 입력을 사용하며 컨텍스트는 패키지의 `context.max_tokens`를 따릅니다. 기존 패키지는512토큰이고, 새 길이는 해당 모델 릴리스에서 실제 검증한 범위를 확인하세요. 이전 검증에서 Qwen 패키지를 다시 불러올 때 컴파일 호출은 0회였으며, 최초 로드에는 가중치 준비와 그래프 복원에 약 3분이 걸렸습니다. 다른 모델 어댑터와 자동 메모리 계획은 구현되어 있지 않습니다. 저장소 분리 이전의 과거 검증 자료는 현재 분리된 코드베이스의 검증 기록과 다를 수 있습니다.

테스트 명령은 `python -m unittest discover -s tests`입니다.

## 재사용 가능한 Gradio 모델 페이지

`examples/model_page.py`를 복사하고 `PACKAGE = Path('fxbex/model.fxbx')`를 설정한 뒤 SDK가 설치된 호스트에서 실행합니다. 전달할 패키지는 애플리케이션의 `fxbex/` 폴더에 둡니다.

```bash
fxbx build ./compiled_model -o fxbex/model.fxbx
```

페이지는 저장된 패키지 목록, 기존 파일을 덮어쓰지 않는 업로드, Load/Unload, 요청 시간 측정과 선택한 파일명을 포함한 Python 예제를 제공합니다. `mode='chat'` 또는 `mode='predict'`를 사용합니다. 이미지 입력·출력 같은 모델별 처리에는 `controls(run)` 콜백을 제공할 수 있습니다. 콜백은 `run('predict', named_inputs)`를 호출하고 `(result, timing)`을 받습니다. 입력 변환과 출력 렌더링은 검증된 모델 어댑터 계약에 맞춰야 합니다. 템플릿은 새 모델 지원을 추가하지 않습니다. 기존 Model Forge가 동일한 페이지 골격과 공유 서비스를 사용합니다. 패키지 파일은 Git에 포함하지 않습니다.

## Python 언어 모델 API

```python
from fxbx import LLM

package = '/root/rngd-model-forge/fxbex/RedHatAI--Qwen3.5-9B-FP8-dynamic__text-bf16__fusion8__ctx512__sdk2026.4.0.fxbex'
with LLM(package) as llm:
    print(llm.generate('2+2는 얼마인가요? 짧게 답하세요.')['text'])
    print(llm.converse('2+2는 얼마인가요?')['text'])
    print(llm.converse('그 답에 3을 곱해 주세요.')['text'])
    llm.reset()
```

`LLM(package)` 또는 `LLM(fxbex=package)`처럼 패키지 경로 하나만 전달합니다. `fxb=package`도 별칭으로 사용할 수 있습니다. 모델 이름은 manifest의 `llm.model`에서 가져오므로 별도 인자는 받지 않습니다. `chat(messages)`는 명시적인 대화 이력을 받습니다. 결과는 `text`와 사용 가능한 런타임 지표를 담은 딕셔너리입니다. `with` 블록을 쓰지 않으면 `llm.close()` 또는 `unload()`를 호출하세요. 실행은 별도 프로세스에서 이뤄지며 종료 시 NPU 메모리를 반환합니다.

이 API는 파일 이름이 `.fxb`여도 FXBEX 콘텐츠를 읽을 수 있지만, **Furiosa 공식 FXB 파일은 지원하지 않습니다.** 공식 FXB에는 `furiosa_llm.LLM`을 사용하세요. 호환 SDK, 원본 가중치와 모델 어댑터 플러그인은 별도로 필요합니다. 모델을 다운로드하거나 컴파일하지 않으며 CPU로 대체 실행하지 않습니다.

### Wheel 설치

```bash
python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install dist/fxbx-0.7.1-py3-none-any.whl
```

`pip install fxbx`는 공개된 Python 코어를 설치합니다. 공개 이름 전환 릴리스는 0.7.0입니다. PyPI 배포는 소스를 공개하므로 다음 릴리스 게시에는 관리자의 인증이 필요합니다. [PyPI 릴리스 준비와 GitHub OIDC 인증](docs/PYPI_RELEASE.md)을 참고하세요. Python 코어는 Apache-2.0 라이선스이며 Furiosa SDK와 모델 파일은 포함하지 않습니다.

기존 문서의 추가 NPU 검증 생략 기록은 해당 시점의 검증 범위를 뜻합니다. API 단위 검사와 portable 무결성·소스·어댑터 사전 검사는 NPU 추론 검증을 대신하지 않습니다.

## 단일 파일 모델 전달

작은 참조 패키지는 완전한 모델이 아닐 수 있습니다. 외부 가중치나 설치된 어댑터가 필요할 수 있습니다. `build --standalone`은 인덱싱된 가중치, 토크나이저와 설정 파일을 포함하지만 외부 어댑터는 별도일 수 있습니다. portable 패키지는 `runtime.bundled_python`이 지정한 정확한 Python 어댑터 코드를 `python/`에 추가합니다. 불러오는 과정에서 이 코드가 실행되므로 신뢰할 수 있는 패키지만 사용하세요. 체크섬은 손상을 감지하지만 작성자를 인증하지 않습니다.

검증된 Model Forge Qwen 소스에서는 다음처럼 만듭니다.

```bash
cd /root/rngd-model-forge
python -m tools.export_portable_qwen fxbex/model.fxbx -o /large-disk/model-portable.fxbex
```

이 명령은 고정된 프로젝트 소스를 확인하고 체크포인트 파일, 토크나이저, 컴파일된 커널과 Python 어댑터를 포함합니다. 다운로드나 별도의 Model Forge 설치는 필요하지 않지만 호환 RNGD 드라이버, Furiosa SDK와 Python 의존성은 호스트에 있어야 합니다. 따라서 `pip install fxbx`와 파일 하나만으로 실행 가능한 조건은 **이미 호환되는 RNGD 호스트에서 FXBEX 0.3.0 이상을 사용하는 경우**입니다. PyPI 0.2.1은 bundled adapter를 지원하지 않습니다. 완전한 portable 배포라고 안내하기 전에 다른 환경에서 실제 추론을 검증해야 합니다. 임시 디스크에는 패키지와 압축 해제된 파일을 함께 둘 공간이 필요합니다.

## 패키지 우선 모델 클래스 (0.4.0 개발 버전)

```python
from fxbx import Model, LLM, YOLO, ImageGeneration

with LLM('chat.fxbex') as model:
    print(model.generate('안녕하세요')['text'])

with YOLO('detector.fxbex') as model:
    result = model.predict('/absolute/path/image.jpg')
    print(result, model.last_timing)

with ImageGeneration('generator.fxbex') as model:
    result = model.generate('새 한 마리', seed=42)

with Model('tensor.fxbex') as model:
    result = model.predict({'x': [[1, 2]]})
```

`ObjectDetection`은 `YOLO`의 별칭입니다. 모든 클래스는 같은 load/close/context 관리와 `load_seconds`, `last_timing`을 사용합니다. `last_timing`은 전송·대기 시간을 포함한 호스트 요청 시간이며 NPU만의 시간은 아닙니다. 입력과 결과는 패키지 어댑터 계약을 따르고 JSON으로 변환할 수 있어야 합니다. YOLO는 `image`, 이미지 생성은 `prompt`와 옵션을 전달합니다. 전처리, 디코딩·샘플링과 결과 파일 경로는 어댑터가 구현합니다. 지원하지 않는 모델을 변환하거나 전처리·어댑터를 자동으로 제공하지 않습니다. 현재 YOLO 어댑터는 이미지 입력만 받습니다. 0.4.0 API는 Git 소스에서 사용할 수 있으며 PyPI 0.3.0에는 이전 API가 있습니다. `LLM(model_name, fxbex=path)` 대신 `LLM(path)`를 사용하세요.

## 실행 영수증과 모델 검증 경계

소스 버전은 worker의 `predict`, `chat` 응답에 `fxbex_execution` 필드를 추가합니다. 어댑터 결과와 패키지 형식은 바뀌지 않습니다.

```python
with YOLO('detector.fxbex') as model:
    result = model.predict('image.jpg')
    receipt = result['fxbex_execution']
    print(receipt['status'], receipt['wall_seconds'], receipt['report_path'])
```

각 세션은 보고서 폴더의 `executions.jsonl`에 manifest 해시, 작업 순서, 성공 여부, 호스트 경과 시간과 일부 어댑터 보고 지표를 기록합니다. 프롬프트, 대화 내용, 텐서 결과와 예외 본문은 저장하지 않습니다. 기록에 실패하면 `persisted: false`로 표시하며 모델 예외를 숨기지 않습니다.

`npu_verdict`는 `unknown`으로 유지됩니다. Python 호출 성공이나 어댑터의 이전 `validated_audit` 기록만으로 해당 요청이 NPU에서 실행됐다고 볼 수 없습니다. 지원 판정을 내리거나 최적 구성을 선택하려면 현재 실행에서 수집한 공식 trace, 수치 정확도, 전체 그래프 범위, 입력·환경 계약과 프로파일링되지 않은 반복 시간까지 검증해야 합니다. 어댑터가 보고한 값은 검증 전까지 증거로 간주하지 않습니다. 호스트 시간에는 실행 요청·전송·대기가 포함되고 완료 기준은 어댑터별로 다릅니다. `Model.last_timing.status`가 실패한 요청은 유효한 시간 표본에서 제외됩니다.

ChunSik, 커널 매칭, 후보 생성과 튜닝 정책은 Furiosa-Hug-Anything이 담당합니다. FXBEX는 패키징, 실행과 보고를 맡습니다. 기존 `.fxbex` 파일은 실행 영수증을 위해 다시 만들 필요가 없습니다. 이 소스 변경은 PyPI 새 릴리스가 아닙니다.

### 모델 입력과 출력 확인하기 — 소스 버전

```bash
fxbx describe model.fxbx
fxbx sample model.fxbx > inputs.json
fxbx run model.fxbx --input inputs.json -o result.json
```

내보내기 도구는 선택적으로 `manifest.io_contract`를 제공합니다. 여기에는 `schema_version=1`, 실행 방식(`chat`/`predict`), `inputs`, `outputs`, `example_input`, 실제 검증 범위를 반영한 `expected_result` 설명이 들어갑니다. 메타데이터가 없으면 누락 사실을 알리며 추측하지 않습니다.
Python에서는 `from fxbx import describe, example_inputs`를 사용할 수 있습니다. 편집기 연동에 사용할 JSON 사전을 반환하며, IDE 자동 완성은 아직 구현되지 않았습니다. 이 정보를 읽는 동안 패키지 코드는 실행하지 않습니다.
소스 버전에서 추가된 명령의 설치본 제공 여부는 해당 PyPI 릴리스를 확인하세요.

### 카탈로그의 모델 이름으로 실행하기 — 소스 버전

호환되는 RNGD SDK와 의존성이 설치된 호스트에서 실행합니다.

```bash
pip install "git+https://github.com/smallstep-zzz/fxbX.git@main" huggingface_hub
fxbx run "SOURCE_ORG/MODEL@RELEASE_VERSION"
```

FuryAwesome에 표시된 원본 모델 ID와 버전을 사용하세요. fxbX는 카탈로그에 고정된 Hugging Face 커밋을 조회하고, HF 다운로드 캐시를 재사용하며, 적재 전에 SHA256을 검증합니다. 이름이 모호하면 `@version`이 필요합니다. 다운로드 크기가 수 GB일 수 있고 실행 가능한 어댑터 코드가 포함되므로 신뢰하는 배포자의 패키지를 선택하세요.

`--input` 없는 `run`은 패키지의 `io_contract.example_input` 예제를 한 번 실행하고 결과·보고서 위치를 출력합니다. `-o result.json`은 응답을 저장합니다. `chat MODEL_ID`는 대화형으로 동작합니다. 로컬 `.fxbex` 경로에도 같은 규칙을 적용합니다.
입출력 메타데이터가 없는 구형 패키지는 누락된 계약을 알립니다. 구형 LLM은 `chat`, 이미지 모델은 해당 HF 안내의 `predict --input inputs.json`을 사용하세요.

내보내기 도구는 샘플 이미지를 아카이브에 넣고 `example_input`에서 `{"$asset":"samples/example.png"}`로 참조할 수 있습니다. 압축 해제된 패키지 내부에서 무결성 목록에 등록된 파일만 해석합니다. 일반 문자열은 텍스트로 유지합니다. 메타데이터에 있는 임의의 셸 명령이나 샘플 생성 코드는 실행하지 않습니다.

일치하는 로컬 파일이 없으면 카탈로그 ID 앞의 `./`도 허용합니다.

```bash
fxbx run ./RedHatAI/Qwen3.5-9B-FP8-dynamic
```

기존 로컬 파일이 우선합니다. 존재하지 않는 `.fxbex` 파일 경로는 파일 오류로 처리합니다. 다운로드에는 HF 캐시를 사용합니다. 예제 메타데이터가 없는 구형 패키지는 앞서 설명한 `chat` 또는 명시적 입력이 필요합니다.

## 과거 릴리스: 0.4.0

당시 Python 3.10–3.14에서 `python -m pip install --upgrade fxbex==0.4.0`으로 설치했습니다. 이 릴리스는 카탈로그 모델 ID 다운로드(`./org/model` 포함), 무결성 검사, 입출력 설명과 샘플 실행을 추가했습니다. `huggingface_hub`는 자동 설치됩니다. 샘플 메타데이터가 없는 구형 모델 패키지는 문서화된 입력 또는 `chat` 명령이 필요합니다. 현재 설치는 문서 상단의 `fxbx` 안내를 따르세요.

### 과거 릴리스: 0.4.1 다운로드 위치 수정

이 버전에서는 `./` 유무와 관계없이 모델 ID로 현재 작업 디렉터리 아래 폴더를 만들고 `model.fxbx`를 저장합니다. HF 전송과 로컬 복사 진행률을 표시하며 기존 파일은 재사용합니다. HF 캐시와 로컬 복사본이 각각 디스크 공간을 차지할 수 있습니다. 당시 업그레이드 명령은 `python -m pip install --upgrade fxbex==0.4.1`이었습니다.

### Qwen 런타임 호환성 조사 기록

Qwen3.5-9B portable을 SDK 2026.4.0 / PyTorch 2.10.0+cpu / Python 3.11 / transformers 5.22.1에서 적재·검증할 때 간헐적인 네이티브 충돌이 발생했습니다. CPU 스레드 제한과 CPU 프로파일링 제외로는 해결되지 않았습니다. Python 3.11을 유지하고 transformers만 5.20.2로 바꾼 비교에서는 수치·NPU 검증과 두 번의 채팅(4, Hello)을 통과했습니다. 당시 비교 조합의 명령은 `python -m pip install --no-deps transformers==5.20.2`입니다. 다만 네이티브 SDK 문제의 원인은 특정되지 않았고, 한 번의 성공이 안정성을 보장하지 않습니다. 같은 주 버전 내 transformers 차이도 경고 정책을 유지합니다.

## 공식 Furiosa LLM API에 fxbX 연결하기

```python
from furiosa_llm import LLM, SamplingParams
from fxbx import hug

LLM = hug(LLM)
# 공식 모델은 원래 생성자·옵션·메서드·객체를 유지합니다.
# llm = LLM("your-official-model", ...)
llm = LLM(model="./model.fxbx", dtype="bf16")  # 패키지가 BF16인 경우
try:
    output = llm.generate("The numbers are one, two,", SamplingParams(temperature=0, max_tokens=16))
    print(output.outputs[0].text)
    answers = llm.chat([{"role": "user", "content": "Hello!"}],
                       SamplingParams(temperature=0, max_tokens=16))
    print(answers[0].outputs[0].text)
finally:
    llm.shutdown()
```

래퍼는 공식 클래스를 전역 수정하지 않습니다. 공식 모델 생성 시 원래 객체를 반환하고, 래핑된 클래스도 상속한 클래스 도우미와 인스턴스 검사를 유지합니다. 로컬 `.fxbex` 경로, `model.fxbx`가 들어 있는 폴더, 정확한 FuryAwesome ID는 fxbX 경로를 선택합니다.
`fxbex=True`는 이 경로를 요구하고, `fxbex=False`는 공식 엔진 경로를 강제합니다. 같은 모델 ID가 두 곳에 있을 때 유용합니다. 오프라인에서 알 수 없는 카탈로그 ID는 경고 후 공식 생성자로 전달합니다. 카탈로그 ID에는 기존 검증·다운로드·캐시를 사용합니다. 공식 엔진의 가중치·SDK·라이선스는 별도로 필요하며, 래퍼가 해당 엔진의 미지원 모델을 지원하게 만들지는 않습니다.

### 실시간 스트리밍

```python
import asyncio

async def show(llm):
    async for text in llm.stream_generate(
        "Once upon a time", SamplingParams(temperature=0, max_tokens=32)
    ):
        print(text, end="", flush=True)
    print()

# 모델 적재 후 종료 전에 호출합니다:
# asyncio.run(show(llm))
```

fxbX는 격리된 작업 프로세스에서 실제 디코더 콜백을 스트리밍합니다. 완성된 응답을 나눠 스트리밍처럼 보이게 하지 않습니다. 유니코드 바이트 조각은 버퍼에 모아 처리합니다.

### 0.5.0 당시의 호환 범위

| 경로 | 지원 동작 |
|---|---|
| 공식 모델 / 공식 FXB | 원래 생성자·엔진을 사용하며 기존 기능을 전달 |
| Qwen `generate` | 문자열 또는 문자열 목록, 실제 공식 RequestOutput/CompletionOutput과 토큰 ID 반환 |
| Qwen `chat` | 단일 또는 순차 처리하는 텍스트 메시지 배치, 공식 RequestOutput 목록 반환 |
| Qwen `stream_generate` | greedy 디코더의 비동기 텍스트 조각, 제한된 길이 생성 |
| 샘플링 | Greedy(`temperature=0`), `max_tokens` 1–256, 기본 생성 길이 16 |
| 컨텍스트 | 당시 512토큰, `max_model_len=512`만 허용 |
| 미지원 옵션 | 기본값과 다른 샘플링, logprobs, 도구·템플릿, 임베딩·점수 계산, 사용자 토크나이저, 토큰화된 입력, 장치·병렬화·스케줄러 옵션은 오류 반환 |

배치는 공식 엔진 배칭이 아닌 순차 처리입니다. 공식 모델의 처리량과 스트림 스케줄링은 바꾸지 않습니다. 최초 hug 어댑터의 대상은 furiosa_llm 2026.4.0과 검증된 Qwen 재생 런타임입니다. Diffusers, Ultralytics, ONNX Runtime은 당시 hug 어댑터 대상이 아니었습니다. 패키지 중심의 `from fxbx import LLM` API는 기존 사전 반환 방식으로 계속 사용할 수 있습니다.

네이티브 충돌 문제가 발생하는 환경에서는 애플리케이션 인터프리터를 유지하고, 별도로 설치·검증한 작업 프로세스용 인터프리터를 명시적으로 선택하세요.

```bash
export FXBEX_WORKER_PYTHON=/usr/bin/python3.10
```

당시 작업 인터프리터에는 FXBEX 0.5.0과 호환 SDK·어댑터 의존성이 필요했습니다. 부모 프로세스는 공식 Python 출력 클래스를 반환합니다. 인터프리터를 자동 전환하거나 충돌을 숨기지 않습니다. 이후 Python 3.11 + transformers 5.20.2의 최초 적재에서도 다시 충돌했으므로 버전 고정만으로 해결된 것은 아닙니다. 진행 중인 네이티브 런타임 조사는 저장소의 이슈 #2를 참고하세요.

### 실제 모델 검증 기록

`docs/HUG_NPU_VALIDATION.json`은 Python 3.10 시험, `docs/HUG_CROSS_ENV_VALIDATION.json`은 conda Python 3.11 부모와 Python 3.10 작업 프로세스 시험을 기록합니다. 두 시험 모두 기존 Qwen portable2를 적재하고 수치·NPU 검증을 통과했습니다(cosine 0.999929, top1 일치, 공식 NPU Task 1,562개, CPU 실행 경로·미확인 그래프 0개). Generate는 실제 토큰 ID를 담은 공식 출력 객체를 반환했고, chat은 `4`를 답했으며, 스트리밍은 완료 전에 연속 조각을 전달했습니다. 교차 환경 시험의 첫 조각은 1.18초, 마지막 조각은 2.82초에 도착했습니다.

`generate`는 공식 API와 같은 이어쓰기 기능입니다. 지시형 질문에는 `chat`을 사용하세요. 이 시험은 일반적인 응답 품질이나 다른 공식 지원 모델의 벤치마크를 뜻하지 않습니다. 공식 생성자·기능 보존은 전달·클래스 의미 회귀 테스트로 확인합니다. Python 3.11 작업 프로세스의 네이티브 충돌은 재현됐고 hug가 해결하지 못했으므로 검증된 작업 환경을 명시적으로 선택해야 합니다. 실행 예제는 `examples/hug_llm.py`를 참고하세요.

## 버전별 로컬 모델 관리 — 0.5.1

새 다운로드에는 읽기 쉬운 버전별 경로를 사용합니다.

```text
RedHatAI/Qwen3.5-9B-FP8-dynamic/
  2026.4.0-portable2/
    Qwen3.5-9B-FP8-dynamic.fxbex
  opt3/
    Qwen3.5-9B-FP8-dynamic.fxbex
```

```bash
fxbx chat RedHatAI/Qwen3.5-9B-FP8-dynamic@2026.4.0-portable2
# 향후 설치하거나 게시한 최적화 버전을 명시적으로 선택할 수 있습니다:
# fxbex chat RedHatAI/Qwen3.5-9B-FP8-dynamic@opt3
```

릴리스 이름은 명시적으로 지정합니다. 위 `opt3`는 설명용 예시이며 공개 모델이 아닙니다. 로컬 버전은 오프라인에서 동작합니다. 여러 로컬 버전이 있으면 `@VERSION` 또는 정확한 파일 경로가 필요하며, 문자열 정렬로 최신·최적 버전을 추측하지 않습니다. 기존 `model.fxbx`는 이름을 바꾸거나 삭제하지 않습니다. 버전을 지정하면 버전 없는 구형 패키지를 잘못 재사용하지 않습니다. hug 어댑터도 같은 해석 규칙을 사용합니다. HF 캐시와 로컬 복사본은 별개입니다.

## 원본 모델 출처 기록 — 0.5.2

```bash
fxbx build ./compiled_model -o model.fxbx --standalone --source-model stabilityai/TripoSR --source-revision ORIGINAL_COMMIT
fxbx inspect model.fxbx
```

매니페스트에는 `provenance.source_url`, 선택적인 `revision`, `status`를 기록합니다. Python 빌더는 `build()`에 `source_url=`과 `source_revision=`을 전달합니다. 명시적인 기존 출처 정보는 유지합니다. 출처가 없으면 `unknown`으로 기록하고, 표시 이름을 출처 근거로 사용하지 않습니다. FHA는 모델 준비·다운로드 기록에서 출처를 채우고 게시 전에 확인합니다. 이 필드가 없는 구형 패키지도 적재할 수 있습니다. 메타데이터를 수정하면 새 아티팩트 해시와 릴리스가 필요하며 기존 파일을 덮어쓰지 않습니다.

## 짧은 모델 이름과 최신 로컬 파일 — 0.5.3

```bash
fxbx chat Qwen3.5-9B-FP8-dynamic
fxbx run yolo11-onnx
fxbx run ./models
```

원본 저장소 이름의 마지막 부분이 유일하면 짧은 모델 ID로 사용할 수 있습니다. 배포자 이름 등이 겹치면 전체 ID가 필요합니다. 폴더는 하위 폴더까지 탐색해 수정 시각이 가장 최근인 `.fxbex`를 선택하고 경로를 출력합니다. 시각이 같으면 경로의 문자열 순서를 사용합니다. 다른 모델이 들어 있는 폴더도 포함됩니다. 정확한 파일 경로와 `@version`은 지정한 대상을 선택합니다. 파일 복사로 수정 시각이 바뀔 수 있으므로 최근 파일 선택은 편의 기능이며 성능 순위가 아닙니다. 원격 릴리스는 카탈로그에 기록된 등록 순서를 사용합니다. 등록 순서가 없는 구형 카탈로그에 여러 버전이 있으면 버전을 명시해야 합니다. 기존 파일은 원격 릴리스보다 우선합니다.


### 패키지가 선언하는 컨텍스트와 메모리 — 0.6

`hug(LLM)`은 검증된 매니페스트의 `context.max_tokens`에서 컨텍스트 한도를 읽습니다. 구형 패키지의 기본값은 512입니다. 출력 허용과 작업 프로세스 전송에는 적재한 런타임의 용량을 적용하며, 실제 토큰화한 입력과 요청 출력의 합이 한도 안에 있어야 합니다. 이 기능이 기존 512토큰 패키지의 컨텍스트를 늘려 주지는 않습니다. 생성자에서 더 작은 한도로 덮어쓰는 기능도 명시적으로 지원하지 않습니다.

런타임 보고서는 원시 SMI 정보를 `memory.jsonl`에 저장하고, 어댑터가 제공하면 논리 텐서 집계를 `memory_latest.json`에 저장합니다. UI에는 논리 가중치·KV·순환 상태의 바이트 수를 구분해 표시합니다. SMI의 0 표시는 HBM 사용량이 0이라는 증거가 아닙니다. SDK에 공개 조회 기능이 없으면 물리 할당량은 미확인으로 둡니다. SDK·드라이버를 업그레이드하거나 사용률을 만들어 내지 않습니다.

### 논리 메모리 계획과 안전한 텐서 수명 — 0.6.2

`fxbx build`는 `runnable-dag-v1`의 작은 그래프·shape·인덱스 JSON 문서에서 연결 관계가 고정된 메모리 계획을 만듭니다. 중간 텐서 shape와 dtype은 `runtime_config.tensor_specs`, 가중치 메타데이터는 `weights_index`에 선언하세요. 누락되거나 동적인 크기는 미확인으로 둡니다. `fxbx inspect model.fxbx --memory`는 NPU 적재 없이 계획을 표시하며, 전체 inspect는 무결성도 검사합니다. 메타데이터만 분석하려면 `fxbex.memory.inspect_archive(path)`를 사용합니다.

가중치를 올리기 전에 오래된 연결 정보 또는 명시한 `memory_plan.policy.budget_bytes` 초과를 거부합니다. 논리 추정치가 맞더라도 물리 HBM 용량을 보장하지는 않습니다. 별칭, 패딩, 네이티브 실행 파일의 작업 공간과 할당기 여유분은 측정하지 않습니다. 구형 사용자 어댑터는 `fxbex.memory.stateful_plan`으로 실제 shape를 선언할 때까지 `unplanned`로 남습니다.

내장 DAG 실행은 상수와 최종 출력을 유지하며, 공개된 현재 스트림 완료 장벽을 지난 뒤 수명이 끝난 중간 참조를 묶어서 해제합니다. 기본 기준은 64MiB 또는 참조 64개입니다. 완료 API가 없으면 기존 보존 방식을 유지합니다. `release_threshold_bytes=0`은 적극적인 해제를 시험하며 해당 단계마다 장벽을 추가하므로, 배포 전에 실측 지연으로 조정하세요. 저장소 할당·재사용은 SDK가 담당합니다. 참조 해제가 물리 메모리 반환을 증명하지 않습니다. 수동 메모리 영역, 버퍼 별칭, CPU 모델 폴백은 추가하지 않습니다. 세션의 `memory_plan.json`과 결과의 `logical_memory`에는 논리 최대 사용량·해제 횟수·장벽을 기록하고 물리 최대 사용량은 미확인으로 둡니다.

### 최신 카탈로그 릴리스와 SDK 환경 — 0.6.3

`fxbx run MODEL`과 `fxbx chat MODEL`은 모델 이름 캐시를 재사용하기 전에 현재 FuryAwesome 카탈로그를 확인합니다. 짧은 이름과 전체 원본 ID는 가장 최근에 등록된 릴리스를 사용하며, 오래된 로컬 파일의 수정 시각이 더 최근이어도 마찬가지입니다. `MODEL@VERSION`은 릴리스를 고정합니다. 명시적인 파일, `./existing-folder`, 절대 경로와 Python `Path` 인자는 로컬 선택을 유지하고, 폴더는 수정 시각이 가장 최근인 파일을 선택합니다. 오프라인 대체 경로는 최신 여부가 미확인임을 경고합니다. 일치하는 설치본은 해시를 확인한 뒤 다운로드 없이 재사용합니다.

실제 명령은 `fxbx --version`으로 확인하세요. CLI와 SDK의 환경이 다르면 사용자 설정 `~/.config/fxbex/config.json`의 `runtime_python`에 SDK 인터프리터 절대 경로(예: `/usr/bin/python3.10`)를 지정합니다. 두 환경에는 같은 fxbX 릴리스를 설치하세요. 런타임 명령과 작업 프로세스가 해당 인터프리터를 사용하며, `FXBEX_WORKER_PYTHON`이 우선합니다. SDK를 설치하거나 CPU 폴백을 켜는 기능이 아닙니다. 다른 설정 키도 보존하세요.

`run`은 채팅 패키지에서 대화형 채팅을 엽니다. `run --sample`은 패키지 예제를 한 번 실행합니다. 예측 패키지의 `run`은 내장 예제를 사용하고, `run --image PATH`는 선언된 이미지 입력을 바꿉니다. `chat`은 `>>>` 대화창을 엽니다. 기본 단발 JSON은 모델 출력·지표·보고서 경로를 유지하고, 원시 메모리·컴파일·재생 진단은 보고된 `result.json`에 저장합니다. `--verbose`로 표시하거나 `-o result.json`으로 별도 저장할 수 있습니다.

### 공유 다운로드와 새로 받기

모델 ID의 다운로드 위치는 `~/.cache/fxbex/models/PUBLISHER/MODEL/VERSION/`이며, `$XDG_CACHE_HOME`을 지정하면 그 아래 `fxbex/models`를 사용합니다. 명시한 로컬 경로는 로컬 대상으로 유지합니다.

```bash
fxbx fetch Qwen3.5-9B-FP8-dynamic --refresh
fxbx cache list
fxbx cache clear --yes
```

새로 받기는 HF 다운로드를 강제하고 카탈로그 SHA256을 확인한 뒤 관리 파일을 교체합니다. 캐시 비우기는 fxbX가 관리하는 모델 다운로드만 지우며 명시적인 로컬 파일이나 HF 캐시는 지우지 않습니다. 비우기 전에 실행 중인 모델을 해제하세요.

### TCLMachine BuildBundle — 소스 0.6.10

TCLMachine 0.3은 지원하는 소형 그래프 모델을 FHA 없이 전달할 수 있습니다. 아카이브와 런타임의 소유자는 fxbX입니다. 선택적인 `build_bundle.json`은 컴파일 산출물, 가중치, 설정, 어댑터·입출력·상태 메타데이터와 독립 실행 검증을 하나의 콘텐츠 ID에 연결합니다. 내용이 달라지면 업로드 전에 거부합니다. BuildBundle이 없는 구형 패키지는 기존 경로를 유지합니다. 새 경로에는 일치하는 소스 버전이 필요하며 새로운 PyPI 배포를 뜻하지 않습니다.

요청별 SMI 정보는 5초 동안 재사용하고 정보의 나이를 표시합니다. 적재·해제 시에는 항상 새로 조회합니다. `MemoryTelemetry.sample(...,force=True)`로 강제 조회할 수 있습니다. 실행 영수증과 논리 모델 메모리는 계속 갱신하며 물리 HBM과 SMI의 0 표시에 대한 불확실성은 유지합니다. 작은 그래프 시간은 이런 호스트 관리 비용과 상주 커널의 완료 시간을 구분해야 합니다.

### 선택적으로 사용하는 텐서·PyTorch 모듈 API

입출력 텐서 명세가 선언된 검증된 RunnableDAG 패키지는 Python 목록으로 변환하지 않고 텐서를 직접 사용할 수 있습니다.

```python
from fxbx import as_torch

with as_torch('./model.fxbx', output_device='cpu') as model:
    outputs = model(input_tensor)  # 컴파일된 shape와 dtype을 정확히 사용합니다.
```

`Session.predict_tensors({'input_name': input_tensor})`도 텐서 출력을 반환합니다. `output_device='furiosa'`는 출력을 장치에 유지하며 공개 스트림 완료 API가 필요합니다. 추론 전용 래퍼이므로 학습, 컴파일된 가중치 변경, dtype·장치 이동과 선언되지 않은 입력·옵션은 거부합니다. 구조화된 출력에는 명시적인 `output_tree`가 필요하며 허용된 라이브러리 반환 클래스만 사용합니다. 임의의 Transformers·Diffusers 파이프라인에 자동 호환되는 기능은 아닙니다. 기존 JSON·LLM API의 동작은 유지합니다.

이전에 검증된 작은 reduction 패키지는 SDK 2026.4.0에서 보존된 JSON 출력과 텐서 호출 4회가 일치했고, 프로파일링한 호출에서 공식 NPU Task 2개와 CPU 모델 실행 경로 0개를 확인했습니다. 소형 텐서 전송 경로의 검증이며 임의의 전체 모델 지원을 뜻하지 않습니다. 입출력 명세가 없거나 잘못된 구형 패키지는 엄격한 텐서 API를 사용하기 전에 다시 만들어야 합니다.

## 공식 FXB 형식으로의 확장

가중치를 포함하지 않는 공식 호환 FXB 작성기는 [NATIVE_FXB.md](docs/NATIVE_FXB.md)에 설명돼 있습니다. 기존 `.fxbex` 패키지도 계속 지원합니다. 검증된 작성기는 실제 공식 매니페스트·EDF 계약을 유지합니다. 임의의 Torch Runnable을 네이티브 모델로 내보내는 기능은 별도의 연결 과제입니다. 작성기로 생성한 파일로 공식 Qwen3-8B 적재와 반복 생성을 통과했으며 [실행 근거](docs/validation/native-fxb-v1/runtime.json)를 참고하세요. 이 LLM 시험에서는 네이티브 연산별 NPU 추적 범위를 수집하지 않았습니다.



### 정적 그래프 자동 FXB — 소스 0.7.1

[자동 계약·실행 연결 검증](docs/STATIC_GRAPH_BRIDGE.md)을 추가했습니다.
`from fxbx import build_model, GraphFXB`로 실제 그래프·EDF·ABI·가중치 연결을
담은 native FXB를 만들고 다시 실행할 수 있습니다. 현재 자동 EDF 내보내기
검증 범위는 정적 연속 BF16 2D mm/linear(선택적 bias)입니다. 공식 show/inspect,
CPU 비교, 실제 RNGD trace와 반복 실행을 통과했습니다.

다른 모델의 등록정보를 사용하지 않습니다. 공식 Furiosa-LLM의 새 아키텍처
구현·KV 스케줄러 등록과 임의의 구형 Runnable 변환은 별도 과제입니다.
**이 기능은 0.7.1 개발 소스에 있으며, PyPI 0.7.0에는 없습니다.**

```bash
# 호환 SDK 환경에서 현재 소스 설치; SDK/의존성 자동 변경 없음
python -m pip install -e . --no-deps
python examples/native_static_graph.py build ./linear.fxb
python examples/native_static_graph.py run ./linear.fxb
```

### 공유 가중치와 요청별 상태

`build_program`과 `tclmachine.export_program`은 이름을 붙인 여러 실행 함수를 하나의 번들로 묶습니다. `GraphFXB.new_state()`로 가중치를 공유하면서 요청별 상태를 분리할 수 있습니다. 가중치는 별도로 공급하며, 함수의 상태와 페이지 기반 KV 캐시는 서로 다른 계약입니다. [공식 구성 재사용과 NPU 검증 범위](docs/OFFICIAL_STATE_REUSE.md)

`SafeTensorWeights`는 전체 CPU 모델을 만들지 않고 체크포인트 헤더부터 확인해 가중치를 읽습니다. [가중치 적재 방식](docs/CHECKPOINT_WEIGHTS.md)

### 생성부터 독립 실행 검증까지

TCLMachine이 생성과 검증 완료를 책임지고 fxbX의 기존 런타임으로 새 프로세스에서 재적재·실행합니다. [단일 호출 사용법과 검증 범위](docs/VERIFIED_BUILD.md)를 참고하세요.
