# FXBEX

Furiosa RNGD에서 사전 컴파일한 실행 파일을 패키징하고 실행하는 Linux CLI입니다. `.fxbex`는 FXBEX가 정의한 독립 형식이며, **Furiosa 공식 FXB 형식이 아닙니다.**

## Python 및 Furiosa SDK 호환성

FXBEX 코어는 Python 3.10–3.14를 지원합니다. Furiosa SDK 2026.4 문서의 Python 지원 범위도 3.10–3.14입니다. 실제 NPU 실행에는 대상 호스트에 설치된 SDK, PyTorch, 모델 어댑터와 호환되는 패키지가 필요합니다.

실행 계획은 Python의 마이너 버전이 달라도 재사용할 수 있습니다. 재사용 시 운영체제, PyTorch 버전, Furiosa SDK 버전과 SDK 소스 지문은 계속 일치해야 하며, 요청 그래프도 기록된 실행 계획과 정확히 일치해야 합니다. 실행 계획이 기록되지 않았거나 요청 그래프가 달라지면 런타임은 컴파일을 시도하지 않고 오류를 반환합니다. Python 주 버전이 다르면 재사용하지 않습니다.

## 설치

```bash
python -m pip install fxbex
fxbex --help
```

체크아웃한 소스로 개발할 때는 다음을 사용합니다.

```bash
python -m pip install -e .
```

기존 SDK 패키지를 바꾸지 않고 설치하려면 사용할 인터프리터를 지정합니다.

```bash
PYTHON=python3.11 bash scripts/install.sh
```

이 패키지는 Furiosa SDK, PyTorch, 드라이버를 설치하거나 업그레이드하지 않습니다. 빌드, 검사, 무결성 전용 검증은 표준 라이브러리만 사용하지만 NPU 실행에는 모델과 호환되는 SDK 및 의존성이 필요합니다. `fxbex` 0.3.0이 [PyPI에 공개](https://pypi.org/project/fxbex/0.3.0/)되어 있습니다. 재현 가능한 설치에는 `python -m pip install fxbex==0.3.0`처럼 버전을 고정하세요. Python 3.13·3.14를 포함한 최신 소스는 GitHub의 현재 브랜치를 사용하세요.

## 명령어

```bash
fxbex build ./compiled_model -o model.fxbex
fxbex build ./compiled_model -o model-standalone.fxbex --standalone
fxbex inspect model.fxbex
fxbex validate model.fxbex
fxbex run model.fxbex
fxbex chat model.fxbex
fxbex serve model.fxbex --port 8000
fxbex doctor
```

`run`은 채팅 어댑터를 실행합니다. 텐서 모델은 `--input inputs.json -o result.json`을 받습니다. `chat`에서는 `/reset`, `/quit`을 사용할 수 있습니다. `serve`는 채팅 어댑터용 비스트리밍 `/v1/chat/completions`와 텐서 어댑터용 `/predict`를 localhost에 제공합니다. `fxbex ui --port 7861`은 선택 기능이며 Gradio가 필요합니다. 모든 명령은 `--help`를 지원합니다.

`validate`는 체크섬, 외부 가중치, SDK 버전, 어댑터 계약과 RNGD 장치를 확인하지만 **추론을 실행하지 않습니다.** SDK나 NPU가 없는 시스템에서는 다음처럼 무결성만 검사할 수 있습니다.

```bash
fxbex validate model.fxbex --integrity-only
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

현재 내장 런타임은 RNGD fusion8의 고정 텐서 DAG를 지원합니다. Qwen 플러그인은 BF16 텍스트 입력, 최대 512 토큰을 지원합니다. 이전 검증에서 Qwen 패키지를 다시 불러올 때 컴파일 호출은 0회였으며, 최초 로드에는 가중치 준비와 그래프 복원에 약 3분이 걸렸습니다. 다른 모델 어댑터와 자동 메모리 계획은 구현되어 있지 않습니다. 저장소 분리 이전의 과거 검증 자료는 현재 분리된 코드베이스의 검증 기록과 다를 수 있습니다.

테스트 명령은 `python -m unittest discover -s tests`입니다.

## 재사용 가능한 Gradio 모델 페이지

`examples/model_page.py`를 복사하고 `PACKAGE = Path('fxbex/model.fxbex')`를 설정한 뒤 SDK가 설치된 호스트에서 실행합니다. 전달할 패키지는 애플리케이션의 `fxbex/` 폴더에 둡니다.

```bash
fxbex build ./compiled_model -o fxbex/model.fxbex
```

페이지는 저장된 패키지 목록, 기존 파일을 덮어쓰지 않는 업로드, Load/Unload, 요청 시간 측정과 선택한 파일명을 포함한 Python 예제를 제공합니다. `mode='chat'` 또는 `mode='predict'`를 사용합니다. 이미지 입력·출력 같은 모델별 처리에는 `controls(run)` 콜백을 제공할 수 있습니다. 콜백은 `run('predict', named_inputs)`를 호출하고 `(result, timing)`을 받습니다. 입력 변환과 출력 렌더링은 검증된 모델 어댑터 계약에 맞춰야 합니다. 템플릿은 새 모델 지원을 추가하지 않습니다. 기존 Model Forge가 동일한 페이지 골격과 공유 서비스를 사용합니다. 패키지 파일은 Git에 포함하지 않습니다.

## Python LLM API

```python
from fxbex import LLM

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
python -m pip install dist/fxbex-0.3.0-py3-none-any.whl
```

`pip install fxbex`는 공개된 Python 코어를 설치합니다. 저장소의 현재 GitHub 소스는 0.4.0 개발 버전입니다. PyPI 배포는 소스를 공개하므로 다음 릴리스 게시에는 관리자의 인증이 필요합니다. [PyPI 릴리스 준비와 GitHub OIDC 인증](docs/PYPI_RELEASE.md)을 참고하세요. Python 코어는 Apache-2.0 라이선스이며 Furiosa SDK와 모델 파일은 포함하지 않습니다.

기존 문서의 추가 NPU 검증 생략 기록은 해당 시점의 검증 범위를 뜻합니다. API 단위 검사와 portable 무결성·소스·어댑터 사전 검사는 NPU 추론 검증을 대신하지 않습니다.

## 단일 파일 모델 전달

작은 참조 패키지는 완전한 모델이 아닐 수 있습니다. 외부 가중치나 설치된 어댑터가 필요할 수 있습니다. `build --standalone`은 인덱싱된 가중치, 토크나이저와 설정 파일을 포함하지만 외부 어댑터는 별도일 수 있습니다. portable 패키지는 `runtime.bundled_python`이 지정한 정확한 Python 어댑터 코드를 `python/`에 추가합니다. 불러오는 과정에서 이 코드가 실행되므로 신뢰할 수 있는 패키지만 사용하세요. 체크섬은 손상을 감지하지만 작성자를 인증하지 않습니다.

검증된 Model Forge Qwen 소스에서는 다음처럼 만듭니다.

```bash
cd /root/rngd-model-forge
python -m tools.export_portable_qwen fxbex/model.fxbex -o /large-disk/model-portable.fxbex
```

이 명령은 고정된 프로젝트 소스를 확인하고 체크포인트 파일, 토크나이저, 컴파일된 커널과 Python 어댑터를 포함합니다. 다운로드나 별도의 Model Forge 설치는 필요하지 않지만 호환 RNGD 드라이버, Furiosa SDK와 Python 의존성은 호스트에 있어야 합니다. 따라서 `pip install fxbex`와 파일 하나만으로 실행 가능한 조건은 **이미 호환되는 RNGD 호스트에서 FXBEX 0.3.0 이상을 사용하는 경우**입니다. PyPI 0.2.1은 bundled adapter를 지원하지 않습니다. 완전한 portable 배포라고 안내하기 전에 다른 환경에서 실제 추론을 검증해야 합니다. 임시 디스크에는 패키지와 압축 해제된 파일을 함께 둘 공간이 필요합니다.

## 패키지 우선 모델 클래스 (0.4.0 개발 버전)

```python
from fxbex import Model, LLM, YOLO, ImageGeneration

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

### Discover model inputs and outputs (source version)

```bash
fxbex describe model.fxbex
fxbex sample model.fxbex > inputs.json
fxbex run model.fxbex --input inputs.json -o result.json
```

Exporters provide an optional `manifest.io_contract` with schema_version=1,
method (`chat`/`predict`), `inputs`, `outputs`, `example_input`, and an honest
`expected_result` description. Missing metadata is reported, never guessed.
Python users can call `from fxbex import describe, example_inputs`. These return
JSON dictionaries suitable for editor integrations; automatic IDE completion is
not yet implemented. No package code executes while reading these hints.
These new commands are available from this repository's source version; a new
PyPI release is required before they can be assumed in `pip install fxbex`.

### Run a catalog model by name (source version)

On a host with the matching RNGD SDK and dependencies:

```bash
pip install "git+https://github.com/smallstep-zzz/FXBEX.git@main" huggingface_hub
fxbex run "SOURCE_ORG/MODEL@RELEASE_VERSION"
```

Use the source model ID and version listed in FuryAwesome. FXBEX resolves the
catalog's immutable HF commit, reuses the HF download cache, and verifies SHA256
before loading. Ambiguous names require `@version`. Downloading can consume many
GB; the package includes executable adapter code, so select a trusted publisher.

`run` without `--input` executes one `io_contract.example_input` and prints the
result/report location; `-o result.json` saves the response. `chat MODEL_ID` remains
interactive. The same behavior applies to local `.fxbex` paths. Legacy packages
without I/O metadata report the missing contract; use `chat` for a legacy LLM or
`predict --input inputs.json` for an image model as its HF instructions specify.

Exporters can include sample images in the archive and reference them as
`{"$asset":"samples/example.png"}` within `example_input`. Only integrity-listed
files inside the extracted package are resolved. Plain strings remain text.
No arbitrary shell commands or sample-generation code is executed from metadata.

A leading `./` is also accepted for catalog IDs when no local file exists:

```bash
fxbex run ./RedHatAI/Qwen3.5-9B-FP8-dynamic
```

An existing local file takes precedence. Missing `.fxbex` file paths remain file
errors. Downloads use the HF cache. Legacy packages without example metadata still
need `chat` or explicit inputs as described above.

## 0.4.0 release

Install with `python -m pip install --upgrade fxbex==0.4.0` on Python 3.10–3.14. This release includes catalog model-ID downloads (including `./org/model`), integrity checks, package I/O descriptions and sample execution. `huggingface_hub` is installed automatically. Legacy model packages still require their documented input or `chat` command when sample metadata is absent.

### 0.4.1 download location fix

Model IDs (with or without `./`) now create a directory under the current working directory and save `model.fxbex` there. HF transfer and local-copy progress are displayed. Existing `model.fxbex` files are reused. The HF cache and local copy may both occupy disk space. Upgrade using `python -m pip install --upgrade fxbex==0.4.1`.

### Qwen runtime compatibility finding

Qwen3.5-9B portable with SDK 2026.4.0 / PyTorch 2.10.0+cpu / Python 3.11 / transformers 5.22.1 produced intermittent native crashes during load validation. Limiting CPU threads and excluding CPU profiling did not resolve them. Keeping Python 3.11 and changing only transformers to 5.20.2 completed numerical/NPU acceptance and two chats (4 and Hello). Use `python -m pip install --no-deps transformers==5.20.2` for this observed combination. The native SDK fault itself has not been localized; one successful comparison is not a stability guarantee. Other same-major transformers differences retain the warning policy.

## Add FXBEX to the official Furiosa LLM API

```python
from furiosa_llm import LLM, SamplingParams
from fxbex import hug

LLM = hug(LLM)
# Official models retain the original constructor, options, methods and objects.
# llm = LLM("your-official-model", ...)
llm = LLM("./model.fxbex", max_model_len=512)
try:
    output = llm.generate("The numbers are one, two,", SamplingParams(temperature=0, max_tokens=16))
    print(output.outputs[0].text)
    answers = llm.chat([{"role": "user", "content": "Hello!"}],
                       SamplingParams(temperature=0, max_tokens=16))
    print(answers[0].outputs[0].text)
finally:
    llm.shutdown()
```

The wrapper does not patch the official class globally. Official construction
returns the original object; a wrapped class keeps its inherited class helpers and
instance checks. Local `.fxbex` paths, folders containing `model.fxbex`, or exact
FuryAwesome IDs select FXBEX. `fxbex=True` requires that route; `fxbex=False` forces
native dispatch (useful when a model ID exists in both places). Offline unknown
catalog IDs fall back to the official constructor with a warning. Catalog IDs use
the existing verified download/cache mechanism. The native engine still needs its
own weights/SDK and licenses; this wrapper does not enable unsupported models there.

### Live streaming

```python
import asyncio

async def show(llm):
    async for text in llm.stream_generate(
        "Once upon a time", SamplingParams(temperature=0, max_tokens=32)
    ):
        print(text, end="", flush=True)
    print()

# Call while llm is loaded, before shutdown:
# asyncio.run(show(llm))
```

FXBEX streams actual decoder callbacks through its isolated worker. It does not
split a completed answer to simulate streaming. Unicode byte fragments are buffered.

### Compatibility boundary in 0.5.0

|Path|Supported behavior|
|---|---|
|Official model / official FXB|Original constructor and engine, all original functionality forwarded|
|Qwen FXBEX `generate`|Raw string or list of strings, actual official RequestOutput/CompletionOutput, real token IDs|
|Qwen FXBEX `chat`|Single or sequential batches of text messages, list of official RequestOutput|
|Qwen FXBEX `stream_generate`|Async text deltas from greedy decoder, bounded generation|
|FXBEX sampling|Greedy (`temperature=0`), `max_tokens` 1–256; default call uses greedy16|
|FXBEX context|512 tokens; only max_model_len=512 is accepted|
|Unsupported FXBEX options|Nondefault sampling, logprobs, tools/templates, embeddings/scoring, custom tokenizer, pretokenized input, device/parallel/scheduler options raise errors|

Batches are sequential, not native-engine batching. Official model throughput and
stream scheduling are unaffected. This first hug adapter targets furiosa_llm 2026.4.0
and the validated Qwen replay runtime; diffusers, Ultralytics and ONNX Runtime are
not yet hug adapters. The separate package-first `from fxbex import LLM` API remains
available with its original dictionary-returning convenience methods.

For an environment affected by the open native-crash issue, keep the application
interpreter and select a separately installed, verified worker interpreter explicitly:

```bash
export FXBEX_WORKER_PYTHON=/usr/bin/python3.10
```

That interpreter must have FXBEX 0.5.0 and the compatible SDK/adapter dependencies.
The parent still returns its official Python output classes. There is no automatic
interpreter switching or suppression of a crash. Python3.11 + transformers5.20.2
also crashed in a subsequent cold load; the version pin is not a complete fix.
See FXBEX issue #2 for the continuing native-runtime investigation.

### Real model validation

`docs/HUG_NPU_VALIDATION.json` records a Python3.10 test and
`docs/HUG_CROSS_ENV_VALIDATION.json` records a conda Python3.11 parent with a
Python3.10 worker. Both loaded the existing Qwen portable2 and passed numerical
acceptance (cosine0.999929, top1 match, 1,562 official NPU tasks, zero CPU routes/
unknown graphs). Generate returned official output objects with real token IDs,
chat answered `4`, and streaming yielded successive fragments before completion.
Cross-environment first delta arrived at 1.18s and the final delta at 2.82s.

Raw `generate` is completion, matching the official API; use `chat` for instruct
questions. These tests do not establish general answer quality or benchmark a
separate officially supported model. Native constructor/functionality preservation
is covered by forwarding/class-semantics regression tests. The open native crash
in the Python3.11 worker was reproduced and is not fixed by hug; select the verified
worker environment explicitly. See `examples/hug_llm.py` for a runnable example.

## Versioned local models (0.5.1)

New downloads use readable, version-separated paths:

```text
RedHatAI/Qwen3.5-9B-FP8-dynamic/
  2026.4.0-portable2/
    Qwen3.5-9B-FP8-dynamic.fxbex
  opt3/
    Qwen3.5-9B-FP8-dynamic.fxbex
```

```bash
fxbex chat RedHatAI/Qwen3.5-9B-FP8-dynamic@2026.4.0-portable2
# A future locally installed or published optimization can be selected explicitly:
# fxbex chat RedHatAI/Qwen3.5-9B-FP8-dynamic@opt3
```

Release labels are explicit; opt3 above is illustrative, not a published model.
Local versions work offline. Multiple local versions require `@VERSION` or an
exact file path; no lexical guess of the newest/best optimization is made. Existing
legacy `model.fxbex` files remain usable and are not renamed or deleted. Explicit
versions never accidentally reuse an unversioned legacy package. The hug adapter
uses the same resolution rules. HF caches remain separate from local copies.
