# 웹 실행 샘플

- [chat.json](chat.json) · 채팅 메시지와 최대 생성 토큰
- [image-generation.json](image-generation.json) · SDXL Turbo의 프롬프트와 시드
- [decision.json](decision.json) · 상태와 질문, 선택지를 받는 의사결정 어댑터용 예시. Julia 스타일의 입력이며, 개별 패키지의 계약에 맞춰 사용합니다.
- [color-shapes.png](color-shapes.png), [color-grid.png](color-grid.png) · 직접 생성한 192×128 RGB 입력 테스트 이미지. 리사이즈, 경계 및 색상 처리를 확인하는 데 사용할 수 있습니다.

패키지에 맞는 JSON 파일을 내려받아 웹 화면의 **JSON 입력 파일**에서 엽니다. CLI에서는 채팅은 `fxbx chat MODEL --input chat.json`, 예측은 `fxbx predict MODEL --input FILE`로 실행합니다. 이미지 입력을 받는 호환 패키지에는 `fxbx run MODEL --sample --image color-shapes.png`처럼 로컬 파일을 줄 수 있습니다. 브라우저 예제에 이미지를 포함하려면 패키지의 `io_contract.example_input`에서 무결성 목록에 포함된 파일을 `{"$asset": "color-shapes.png"}`로 참조합니다.
