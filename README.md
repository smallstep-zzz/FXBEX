# fxbX

Furiosa RNGD에서 배포된 모델을 내려받고 실행하는 Linux CLI와 Python 라이브러리입니다. [FuryAwesome](https://github.com/smallstep-zzz/furyawesome)에서 실행할 모델과 모델별 예제를 찾을 수 있습니다.

## 설치하고 실행하기

호환 Furiosa SDK와 RNGD 드라이버가 설치된 Linux에서 실행합니다. fxbX는 Python 3.10–3.14를 지원하며, 모델마다 검증된 SDK와 Python 의존성이 다를 수 있습니다.

```bash
python -m pip install fxbx --upgrade --no-cache-dir
fxbx run yolo11-onnx --sample
fxbx chat Qwen3.5-9B-FP8-dynamic
```

모델명으로 실행하면 카탈로그에서 릴리스를 찾아 파일을 내려받고 공유 캐시를 재사용합니다. `org/model@release`로 버전을 지정할 수 있습니다. 로컬 `.fxbx`와 기존 `.fxbex` 파일도 열 수 있습니다. 채팅은 `/reset`으로 초기화하고 `/quit`으로 종료합니다. 채팅 모델을 한 번만 실행하려면 `fxbx run MODEL --sample`을 사용합니다.

```bash
fxbx --help
fxbx doctor
fxbx fetch MODEL --refresh
fxbx cache list
fxbx cache clear --yes
```

캐시 정리는 fxbX가 관리하는 모델 파일에만 적용됩니다. 다운로드한 파일을 실행할 때는 신뢰할 수 있는 배포자의 모델만 사용하세요. 일부 패키지에는 로드할 때 실행하는 Python 어댑터가 포함됩니다.

## 로컬 모델 확인과 실행

```bash
fxbx inspect model.fxbx
fxbx validate model.fxbx --integrity-only
fxbx describe model.fxbx
fxbx sample model.fxbx
fxbx run model.fxbx --input inputs.json -o result.json
```

`inspect`에서 무결성, 런타임 설정, 원본 모델 정보와 논리 메모리 계획(`--memory`)을 확인할 수 있습니다. `validate`는 SDK, 어댑터와 가중치 등 실행 조건을 검사합니다. `--integrity-only`는 SDK나 NPU 없이 파일만 검사할 수 있으며, 어느 쪽도 추론을 실행하지 않습니다. 외부 가중치가 필요한 모델에는 `--weights-dir`로 체크포인트 경로를 지정합니다.

입력 및 결과 형식은 모델의 계약을 따릅니다. 샘플이 포함된 모델에는 `--sample`, 이미지 입력이 선언된 모델에는 `--image photo.jpg`를 사용할 수 있습니다. 이미지 생성 결과는 `fxbx run FLUX.2-klein-4B --sample -o image.png`처럼 저장합니다.

## Python과 HTTP

```python
from fxbx import LLM

with LLM("chat.fxbx") as model:
    print(model.generate("안녕하세요"))
```

이미지 감지에는 `YOLO(path).predict(image)`, 이미지 생성에는 `ImageGeneration(path).generate(prompt)`, 일반 모델에는 `Model(path).predict(inputs)`를 사용합니다. 텐서 계약이 선언된 Runnable DAG는 `as_torch(path)`로 추론용 PyTorch 모듈처럼 호출할 수 있습니다. 공식 Furiosa LLM을 쓰는 프로그램에는 [hug 어댑터 예제](examples/hug_llm.py)를 참고하세요.

```bash
fxbx serve Qwen3.5-9B-FP8-dynamic --port 8000
```

채팅은 `http://127.0.0.1:8000/v1/chat/completions`에서 OpenAI 호환 비스트리밍 요청을, 예측 모델은 `/predict`에서 JSON 입력을 받습니다. 지원하는 생성 옵션은 모델과 어댑터에 따라 달라집니다.

## 재사용 가능한 Gradio 모델 페이지

이미 준비된 모델 파일을 로드하고 실행하는 예제입니다. Gradio를 설치한 뒤 `fxbx ui --port 7861`로 로컬 UI를 실행합니다. 모델별 화면을 만들려면 [examples/model_page.py](examples/model_page.py)를 복사하고 `PACKAGE`에 기존 모델 파일 경로를 지정하세요. 페이지에서 모델을 적재·해제하고 실행 시간을 확인할 수 있습니다.

## SDK 환경과 파일 형식

NPU 실행에는 대상 모델에 맞는 Furiosa SDK와 PyTorch가 필요합니다. 먼저 `fxbx doctor`를 실행하세요. fxbX는 누락된 패키지의 설치를 제안할 수 있습니다. 시스템 드라이버는 호스트에서 설치해야 합니다. 다른 Python 환경에 호환 SDK와 fxbX를 설치했다면 `FXBEX_WORKER_PYTHON`에 해당 Python 실행 파일의 경로를 지정할 수 있습니다.

`.fxbx`와 구형 `.fxbex`는 fxbX 모델 패키지입니다. 공식 `.fxb`는 해당 Furiosa 런타임에서 사용합니다. fxbX는 지원되는 그래프 FXB를 위한 `GraphFXB`와 커스텀 실행 계획을 위한 `ExecutionFXB`도 제공합니다. 일부 파일은 [외부 체크포인트](docs/CHECKPOINT_WEIGHTS.md)가 필요합니다.
