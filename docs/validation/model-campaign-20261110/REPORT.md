# 정적 FXB 실패 수집 결과

관측된 guard/예외와 근본 원인 추정을 구분합니다. 규칙 부재는 SDK/NPU 미지원 판정이 아닙니다.

|모델/구조|결과|실패 단계|코드|
|---|---|---|---|
|control (linear/relu; harness control)|smoke_passed|trace_gate||
|deit (patch projection, vision attention, position embedding)|failed|export_static|EXECUTION_RULE_GAP|
|lfm2 (short convolution and attention hybrid)|failed|export_static|EXECUTION_RULE_GAP|
|minilm (encoder attention, layer norm, embedding)|failed|export_static|EXECUTION_RULE_GAP|
|qwen35 (gated delta/recurrent and attention hybrid)|failed|export_static|EXECUTION_RULE_GAP|
|resnet (convolution, batch norm, residual, pooling)|failed|export_static|EXECUTION_RULE_GAP|
|segformer (multiscale segmentation, resize, depthwise convolution)|failed|export_static|LAYOUT_UNSUPPORTED|
|whisper (audio Conv1d, encoder-decoder cross attention)|failed|export_static|EXECUTION_RULE_GAP|

## control

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/control`
- 입력/모델: `{"model_id": "", "features": "linear/relu; harness control", "snapshot": "None", "revision": null, "input_scope": "synthetic deterministic smoke input; not task accuracy"}`
- 판정: NPU=confirmed_per_step; CPU fallback=no_trace_routes
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases control`

## deit

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/deit`
- 입력/모델: `{"model_id": "facebook/deit-tiny-patch16-224", "features": "patch projection, vision attention, position embedding", "snapshot": "/root/.cache/huggingface/hub/models--facebook--deit-tiny-patch16-224/snapshots/b3428f18dcc7b543470d07f14b4a4157815d1880", "revision": "b3428f18dcc7b543470d07f14b4a4157815d1880", "input_scope": "synthetic deterministic smoke input; not task accuracy", "config_sha256": "e8786a68c993c04b8721c4d379e4f9de6a9564fba5987704dbf8696d53b7972d", "scope": "single 224x224 image logits"}`
- 판정: NPU=not_run; CPU fallback=unknown
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases deit`
- 오류: `65 unresolved nodes; first={'node': 'conv2d', 'op': 'aten.conv2d.default', 'reason': 'no public EDF region rule', 'category': 'rule_or_contract_gap', 'detail': 'no public EDF region rule', 'module_stack': "), 'L__self__model.vit': ('model.vit', 'transformers.models.vit.modeling_vit.ViTModel'), 'L__self__model.vit.embeddings': ('model.vit.embeddings', 'transformers.models.vit.modeling_vit.ViTEmbeddings'), 'L__self__model.vit.embeddings.patch_embeddings': ('model.vit.embeddings.patch_embeddings', 'transformers.models.vit.modeling_vit.ViTPatchEmbeddings'), 'L__self__model.vit.embeddings.patch_embeddings.projection': ('model.vit.embeddings.patch_embeddings.projection', 'torch.nn.modules.conv.Conv2d')}", 'source': '/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 538, in forward\n    outputs: BaseModelOutputWithPooling = self.vit(\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 373, in forward\n    embedding_output = self.embeddings(\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 136, in forward\n    embeddings = self.patch_embeddings(pixel_values)\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 69, in forward\n    return self.projection(pixel_values).flatten(2).transpose(1, 2)\n  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 553, in forward\n    return self._conv_forward(i`
- 진단: {"observed_reason": "정적 실행 계획의 공유 EDF 규칙이 해당 연산/shape/dtype/layout을 수용하지 못함", "root_cause_status": "preflight guard confirmed; underlying SDK support unknown", "proposed_fix": "failure_groups별 최소 그래프를 분리하고 동일 shape/dtype의 공유 규칙을 추가·수치/trace 검증한다. 모델명 예외는 추가하지 않는다.", "next_evidence": ["graph.json의 해당 node와 preflight source를 대조", "최초 gap을 고친 뒤 재실행하여 뒤에 가려진 실패 수집", "CPU 기준값, 각 step NPU trace, 새 프로세스 재적재 검증"]}
- 지문: `94a19cd7e4b8e0cd6bba`

### `aten.transpose.int` × 49
조건: no public EDF region rule
노드: `transpose`
```text
campaign-20261110/collector.py", line 68, in forward
    def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 538, in forward
    outputs: BaseModelOutputWithPooling = self.vit(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 373, in forward
    embedding_output = self.embeddings(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 136, in forward
    embeddings = self.patch_embeddings(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 69, in forward
    return self.projection(pixel_values).flatten(2).transpose(1, 2)
'L__self__': ('', '__main__.fixture.<locals>.Vision'), 'L__self__model': ('model', 'transformers.models.vit.modeling_vit.ViTForImageClassification'), 'L__self__model.vit': ('model.vit', 'transformers.models.vit.modeling_vit.ViTModel'), 'L__self__model.vit.embeddings': ('model.vit.embeddings', 'transformers.models.vit.modeling_vit.ViTEmbeddings'), 'L__self__model.vit.embeddings.patch_embeddings': ('model.vit.embeddings.patch_embeddings', 'transformers.models.vit.modeling_vit.ViTPatchEmbeddings')}
```

### `aten.scaled_dot_product_attention.default` × 12
조건: no public EDF region rule
노드: `scaled_dot_product_attention`
```text
8, in forward
    def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 538, in forward
    outputs: BaseModelOutputWithPooling = self.vit(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 383, in forward
    hidden_states = layer(hidden_states, attention_mask, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 275, in forward
    hidden_states, _ = self.attention(hidden_states, attention_mask, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 224, in forward
    attn_output, attn_weights = attention_interface(
{'L__self__': ('', '__main__.fixture.<locals>.Vision'), 'L__self__model': ('model', 'transformers.models.vit.modeling_vit.ViTForImageClassification'), 'L__self__model.vit': ('model.vit', 'transformers.models.vit.modeling_vit.ViTModel'), 'L__self__model.vit.layers.0': ('model.vit.layers.0', 'transformers.models.vit.modeling_vit.ViTLayer'), 'L__self__model.vit.layers.0.attention': ('model.vit.layers.0.attention', 'transformers.models.vit.modeling_vit.ViTAttention')}
```

### `aten.conv2d.default` × 1
조건: no public EDF region rule
노드: `conv2d`
```text
/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 538, in forward
    outputs: BaseModelOutputWithPooling = self.vit(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 373, in forward
    embedding_output = self.embeddings(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 136, in forward
    embeddings = self.patch_embeddings(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 69, in forward
    return self.projection(pixel_values).flatten(2).transpose(1, 2)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 553, in forward
    return self._conv_forward(input, self.weight, self.bias)
), 'L__self__model.vit': ('model.vit', 'transformers.models.vit.modeling_vit.ViTModel'), 'L__self__model.vit.embeddings': ('model.vit.embeddings', 'transformers.models.vit.modeling_vit.ViTEmbeddings'), 'L__self__model.vit.embeddings.patch_embeddings': ('model.vit.embeddings.patch_embeddings', 'transformers.models.vit.modeling_vit.ViTPatchEmbeddings'), 'L__self__model.vit.embeddings.patch_embeddings.projection': ('model.vit.embeddings.patch_embeddings.projection', 'torch.nn.modules.conv.Conv2d')}
```

### `aten.cat.default` × 1
조건: no public EDF region rule
노드: `cat`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 68, in forward
    def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 538, in forward
    outputs: BaseModelOutputWithPooling = self.vit(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 373, in forward
    embedding_output = self.embeddings(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 147, in forward
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)
{'L__self__': ('', '__main__.fixture.<locals>.Vision'), 'L__self__model': ('model', 'transformers.models.vit.modeling_vit.ViTForImageClassification'), 'L__self__model.vit': ('model.vit', 'transformers.models.vit.modeling_vit.ViTModel'), 'L__self__model.vit.embeddings': ('model.vit.embeddings', 'transformers.models.vit.modeling_vit.ViTEmbeddings')}
```

### `aten.ge.Scalar` × 1
조건: no public EDF region rule
노드: `ge`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 68, in forward
    def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 538, in forward
    outputs: BaseModelOutputWithPooling = self.vit(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 376, in forward
    attention_mask = create_bidirectional_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Vision'), 'L__self__model': ('model', 'transformers.models.vit.modeling_vit.ViTForImageClassification'), 'L__self__model.vit': ('model.vit', 'transformers.models.vit.modeling_vit.ViTModel')}
```

### `aten.expand.default` × 1
조건: no public EDF region rule
노드: `expand_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 68, in forward
    def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 538, in forward
    outputs: BaseModelOutputWithPooling = self.vit(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/vit/modeling_vit.py", line 376, in forward
    attention_mask = create_bidirectional_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Vision'), 'L__self__model': ('model', 'transformers.models.vit.modeling_vit.ViTForImageClassification'), 'L__self__model.vit': ('model.vit', 'transformers.models.vit.modeling_vit.ViTModel')}
```

## lfm2

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/lfm2`
- 입력/모델: `{"model_id": "LiquidAI/LFM2-350M", "features": "short convolution and attention hybrid", "snapshot": "/root/.cache/huggingface/hub/models--LiquidAI--LFM2-350M/snapshots/f37d3f5c8c5484bc01dad379a595cf4c68c4e70e", "revision": "f37d3f5c8c5484bc01dad379a595cf4c68c4e70e", "input_scope": "synthetic deterministic smoke input; not task accuracy", "config_sha256": "fd3b3fba4e50e7b9a22bd41cbab59e9b28e319b2de19668d7fd9777c8d1a9ba1", "scope": "text-only fixed-length prefill; no cache/decode or full-context claim"}`
- 판정: NPU=not_run; CPU fallback=unknown
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases lfm2`
- 오류: `267 unresolved nodes; first={'node': '_to_copy', 'op': 'aten._to_copy.default', 'reason': 'cast requires explicit BF16/FP32 dtype only', 'category': 'rule_or_contract_gap', 'detail': 'cast requires explicit BF16/FP32 dtype only', 'module_stack': "{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}", 'source': 'File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward\n    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward\n    outputs: BaseModelOutputWithPast = self.model(\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward\n    "full_attention": create_causal_mask(**mask_kwargs),'}; details=/root/.cache/hf-librarian/model-campaign-20261110/attempts/lfm2/model.fxb.plan/preflight.json`
- 진단: {"observed_reason": "정적 실행 계획의 공유 EDF 규칙이 해당 연산/shape/dtype/layout을 수용하지 못함", "root_cause_status": "preflight guard confirmed; underlying SDK support unknown", "proposed_fix": "failure_groups별 최소 그래프를 분리하고 동일 shape/dtype의 공유 규칙을 추가·수치/trace 검증한다. 모델명 예외는 추가하지 않는다.", "next_evidence": ["graph.json의 해당 node와 preflight source를 대조", "최초 gap을 고친 뒤 재실행하여 뒤에 가려진 실패 수집", "CPU 기준값, 각 step NPU trace, 새 프로세스 재적재 검증"]}
- 지문: `6474e56979259264fba8`

### `aten.slice.Tensor` × 34
조건: no public EDF region rule
노드: `slice_2`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 370, in forward
    hidden_states = causal_conv1d_fn(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.transpose.int` × 32
조건: no public EDF region rule
노드: `transpose`
```text
.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 352, in forward
    BCx = self.in_proj(hidden_states).transpose(-1, -2)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `<built-in function getitem>` × 30
조건: no public EDF region rule
노드: `getitem`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 353, in forward
    B, C, x = BCx.chunk(3, dim=-2)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.cat.default` × 13
조건: no public EDF region rule
노드: `cat`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 507, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 104, in forward
    emb = torch.cat((freqs, freqs), dim=-1)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.lfm2.modeling_lfm2.Lfm2RotaryEmbedding')}
```

### `aten._unsafe_view.default` × 12
조건: no public EDF region rule
노드: `_unsafe_view`
```text
e/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 245, in forward
    attn_output, attn_weights = attention_interface(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

### `aten.mul.Tensor` × 10
조건: pointwise requires BF16/FP32; integer arithmetic needs a separate precision contract
노드: `mul_5`
```text
l-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 351, in forward
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.split.Tensor` × 10
조건: no public EDF region rule
노드: `split`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 353, in forward
    B, C, x = BCx.chunk(3, dim=-2)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.squeeze.dim` × 10
조건: no public EDF region rule
노드: `squeeze`
```text
l-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 371, in forward
    hidden_states, self.conv.weight.squeeze(1), self.conv.bias, seq_idx=seq_idx
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.expand.default` × 7
조건: no public EDF region rule
노드: `expand`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten.unsqueeze.default` × 6
조건: no public EDF region rule
노드: `unsqueeze_20`
```text
e/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 245, in forward
    attn_output, attn_weights = attention_interface(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

### `aten.clone.default` × 6
조건: no public EDF region rule
노드: `clone`
```text
e/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 245, in forward
    attn_output, attn_weights = attention_interface(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

### `aten.matmul.default` × 6
조건: explicit packing rule required for noncontiguous inputs
노드: `matmul`
```text
e/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 245, in forward
    attn_output, attn_weights = attention_interface(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

### `aten.softmax.int` × 6
조건: no public EDF region rule
노드: `softmax`
```text
e/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 245, in forward
    attn_output, attn_weights = attention_interface(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

### `aten.matmul.default` × 6
조건: only 2D/3D dense contractions have candidate rules
노드: `matmul_1`
```text
e/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 245, in forward
    attn_output, attn_weights = attention_interface(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

### `aten._to_copy.default` × 2
조건: cast requires explicit BF16/FP32 dtype only
노드: `_to_copy`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten.__and__.Tensor` × 2
조건: no public EDF region rule
노드: `and_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten.new_ones.default` × 1
조건: no public EDF region rule
노드: `new_ones`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten.le.Tensor` × 1
조건: no public EDF region rule
노드: `le`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten.index.Tensor` × 1
조건: no public EDF region rule
노드: `index`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten.zeros.default` × 1
조건: no public EDF region rule
노드: `zeros`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten.where.ScalarOther` × 1
조건: no public EDF region rule
노드: `where`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 502, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model')}
```

### `aten._to_copy.default` × 1
조건: pointwise requires BF16/FP32; integer arithmetic needs a separate precision contract
노드: `_to_copy_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 507, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 103, in forward
    freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.lfm2.modeling_lfm2.Lfm2RotaryEmbedding')}
```

### `aten.cos.default` × 1
조건: no public EDF region rule
노드: `cos`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 507, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 105, in forward
    cos = emb.cos() * self.attention_scaling
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.lfm2.modeling_lfm2.Lfm2RotaryEmbedding')}
```

### `aten.sin.default` × 1
조건: no public EDF region rule
노드: `sin`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 507, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 106, in forward
    sin = emb.sin() * self.attention_scaling
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.lfm2.modeling_lfm2.Lfm2RotaryEmbedding')}
```

### `aten.mul.Tensor` × 1
조건: getitem: explicit packing required for noncontiguous tensors
노드: `mul_6`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 354, in forward
    hidden_states = B * x
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.conv1d.default` × 1
조건: mul_6: explicit packing required for noncontiguous tensors
노드: `conv1d`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 370, in forward
    hidden_states = causal_conv1d_fn(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.mul.Tensor` × 1
조건: getitem_1: explicit packing required for noncontiguous tensors
노드: `mul_7`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 378, in forward
    y = C * hidden_states
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.0.conv': ('model.model.layers.0.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.mul.Tensor` × 1
조건: getitem_3: explicit packing required for noncontiguous tensors
노드: `mul_14`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 354, in forward
    hidden_states = B * x
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.1': ('model.model.layers.1', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.1.conv': ('model.model.layers.1.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.conv1d.default` × 1
조건: mul_14: explicit packing required for noncontiguous tensors
노드: `conv1d_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 370, in forward
    hidden_states = causal_conv1d_fn(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.1': ('model.model.layers.1', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.1.conv': ('model.model.layers.1.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.mul.Tensor` × 1
조건: getitem_4: explicit packing required for noncontiguous tensors
노드: `mul_15`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 417, in forward
    hidden_states = self.conv(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 378, in forward
    y = C * hidden_states
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.1': ('model.model.layers.1', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.1.conv': ('model.model.layers.1.conv', 'transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv')}
```

### `aten.mul.Tensor` × 1
조건: transpose_4: explicit packing required for noncontiguous tensors
노드: `mul_25`
```text
110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 236, in forward
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

### `aten.neg.default` × 1
조건: slice_5: explicit packing required for noncontiguous tensors
노드: `neg`
```text
110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 575, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 511, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 408, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/lfm2/modeling_lfm2.py", line 236, in forward
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.lfm2.modeling_lfm2.Lfm2ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.lfm2.modeling_lfm2.Lfm2Model'), 'L__self__model.model.layers.2': ('model.model.layers.2', 'transformers.models.lfm2.modeling_lfm2.Lfm2DecoderLayer'), 'L__self__model.model.layers.2.self_attn': ('model.model.layers.2.self_attn', 'transformers.models.lfm2.modeling_lfm2.Lfm2Attention')}
```

## minilm

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/minilm`
- 입력/모델: `{"model_id": "sentence-transformers/all-MiniLM-L6-v2", "features": "encoder attention, layer norm, embedding", "snapshot": "/root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "input_scope": "synthetic deterministic smoke input; not task accuracy", "config_sha256": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41", "scope": "text-only fixed-length prefill; no cache/decode or full-context claim"}`
- 판정: NPU=not_run; CPU fallback=unknown
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases minilm`
- 오류: `30 unresolved nodes; first={'node': 'gather', 'op': 'aten.gather.default', 'reason': 'no public EDF region rule', 'category': 'rule_or_contract_gap', 'detail': 'no public EDF region rule', 'module_stack': "{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel'), 'L__self__model.embeddings': ('model.embeddings', 'transformers.models.bert.modeling_bert.BertEmbeddings')}", 'source': 'File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward\n    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 653, in forward\n    embedding_output = self.embeddings(\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 93, in forward\n    buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)'}; details=/root/.cache/hf-librarian/model-campaign-20261110/attempts/minilm/model.fxb.plan/preflight.json`
- 진단: {"observed_reason": "정적 실행 계획의 공유 EDF 규칙이 해당 연산/shape/dtype/layout을 수용하지 못함", "root_cause_status": "preflight guard confirmed; underlying SDK support unknown", "proposed_fix": "failure_groups별 최소 그래프를 분리하고 동일 shape/dtype의 공유 규칙을 추가·수치/trace 검증한다. 모델명 예외는 추가하지 않는다.", "next_evidence": ["graph.json의 해당 node와 preflight source를 대조", "최초 gap을 고친 뒤 재실행하여 뒤에 가려진 실패 수집", "CPU 기준값, 각 step NPU trace, 새 프로세스 재적재 검증"]}
- 지문: `090aadadf33a9149f2ce`

### `aten.matmul.default` × 12
조건: only 2D/3D dense contractions have candidate rules
노드: `matmul`
```text
rn_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 669, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 436, in forward
    hidden_states = layer_module(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 383, in forward
    self_attention_output, _ = self.attention(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 314, in forward
    attention_output, attn_weights = self.self(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 192, in forward
    attn_output, attn_weights = attention_interface(
 'L__self__model.encoder': ('model.encoder', 'transformers.models.bert.modeling_bert.BertEncoder'), 'L__self__model.encoder.layer.0': ('model.encoder.layer.0', 'transformers.models.bert.modeling_bert.BertLayer'), 'L__self__model.encoder.layer.0.attention': ('model.encoder.layer.0.attention', 'transformers.models.bert.modeling_bert.BertAttention'), 'L__self__model.encoder.layer.0.attention.self': ('model.encoder.layer.0.attention.self', 'transformers.models.bert.modeling_bert.BertSelfAttention')}
```

### `aten.softmax.int` × 6
조건: no public EDF region rule
노드: `softmax`
```text
rn_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 669, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 436, in forward
    hidden_states = layer_module(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 383, in forward
    self_attention_output, _ = self.attention(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 314, in forward
    attention_output, attn_weights = self.self(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 192, in forward
    attn_output, attn_weights = attention_interface(
 'L__self__model.encoder': ('model.encoder', 'transformers.models.bert.modeling_bert.BertEncoder'), 'L__self__model.encoder.layer.0': ('model.encoder.layer.0', 'transformers.models.bert.modeling_bert.BertLayer'), 'L__self__model.encoder.layer.0.attention': ('model.encoder.layer.0.attention', 'transformers.models.bert.modeling_bert.BertAttention'), 'L__self__model.encoder.layer.0.attention.self': ('model.encoder.layer.0.attention.self', 'transformers.models.bert.modeling_bert.BertSelfAttention')}
```

### `aten.embedding.default` × 2
조건: embedding requires BF16 table and 1D/2D int32 indices
노드: `embedding_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 653, in forward
    embedding_output = self.embeddings(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 100, in forward
    token_type_embeddings = self.token_type_embeddings(token_type_ids)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/sparse.py", line 191, in forward
    return F.embedding(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel'), 'L__self__model.embeddings': ('model.embeddings', 'transformers.models.bert.modeling_bert.BertEmbeddings'), 'L__self__model.embeddings.token_type_embeddings': ('model.embeddings.token_type_embeddings', 'torch.nn.modules.sparse.Embedding')}
```

### `aten.__and__.Tensor` × 2
조건: no public EDF region rule
노드: `and_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

### `aten.gather.default` × 1
조건: no public EDF region rule
노드: `gather`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 653, in forward
    embedding_output = self.embeddings(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 93, in forward
    buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel'), 'L__self__model.embeddings': ('model.embeddings', 'transformers.models.bert.modeling_bert.BertEmbeddings')}
```

### `aten._to_copy.default` × 1
조건: cast requires explicit BF16/FP32 dtype only
노드: `_to_copy_6`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

### `aten.new_ones.default` × 1
조건: no public EDF region rule
노드: `new_ones`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

### `aten.ge.Scalar` × 1
조건: no public EDF region rule
노드: `ge`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

### `aten.index.Tensor` × 1
조건: no public EDF region rule
노드: `index`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

### `aten.expand.default` × 1
조건: no public EDF region rule
노드: `expand_2`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

### `aten.zeros.default` × 1
조건: no public EDF region rule
노드: `zeros`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

### `aten.where.ScalarOther` × 1
조건: no public EDF region rule
노드: `where`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 65, in forward
    return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py", line 661, in forward
    attention_mask, encoder_attention_mask = self._create_attention_masks(
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.bert.modeling_bert.BertModel')}
```

## qwen35

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/qwen35`
- 입력/모델: `{"model_id": "Qwen/Qwen3.5-0.8B", "features": "gated delta/recurrent and attention hybrid", "snapshot": "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17", "revision": "2fc06364715b967f1860aea9cf38778875588b17", "input_scope": "synthetic deterministic smoke input; not task accuracy", "config_sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204", "scope": "text-only fixed-length prefill; no cache/decode or full-context claim"}`
- 판정: NPU=not_run; CPU fallback=unknown
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases qwen35`
- 오류: `7907 unresolved nodes; first={'node': 'expand', 'op': 'aten.expand.default', 'reason': 'no public EDF region rule', 'category': 'rule_or_contract_gap', 'detail': 'no public EDF region rule', 'module_stack': "{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel')}", 'source': 'File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward\n    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward\n    outputs: BaseModelOutputWithPast = self.model(\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1249, in forward\n    position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)'}; details=/root/.cache/hf-librarian/model-campaign-20261110/attempts/qwen35/model.fxb.plan/preflight.json`
- 진단: {"observed_reason": "정적 실행 계획의 공유 EDF 규칙이 해당 연산/shape/dtype/layout을 수용하지 못함", "root_cause_status": "preflight guard confirmed; underlying SDK support unknown", "proposed_fix": "failure_groups별 최소 그래프를 분리하고 동일 shape/dtype의 공유 규칙을 추가·수치/trace 검증한다. 모델명 예외는 추가하지 않는다.", "next_evidence": ["graph.json의 해당 node와 preflight source를 대조", "최초 gap을 고친 뒤 재실행하여 뒤에 가려진 실패 수집", "CPU 기준값, 각 step NPU trace, 새 프로세스 재적재 검증"]}
- 지문: `c06b6485248b105e1921`

### `aten.select.int` × 2268
조건: no public EDF region rule
노드: `select_26`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.slice.Tensor` × 1183
조건: no public EDF region rule
노드: `slice_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1255, in forward
    position_ids = position_ids[1:]
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel')}
```

### `aten.copy.default` × 1156
조건: no public EDF region rule
노드: `copy`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1275, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 186, in forward
    sin = self.recomposition_frequencies(sin)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextRotaryEmbedding')}
```

### `aten.select_scatter.default` × 1156
조건: no public EDF region rule
노드: `select_scatter`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1275, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 186, in forward
    sin = self.recomposition_frequencies(sin)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextRotaryEmbedding')}
```

### `aten.slice_scatter.default` × 1138
조건: no public EDF region rule
노드: `slice_scatter`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1275, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 186, in forward
    sin = self.recomposition_frequencies(sin)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextRotaryEmbedding')}
```

### `aten.matmul.default` × 126
조건: BF16 required; implicit casts are forbidden
노드: `matmul`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.transpose.int` × 102
조건: no public EDF region rule
노드: `transpose_1`
```text
/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 588, in forward
    mixed_qkv = mixed_qkv.transpose(1, 2)
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.pad.default` × 90
조건: no public EDF region rule
노드: `pad`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.exp.default` × 72
조건: no public EDF region rule
노드: `exp`
```text
tor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 605, in forward
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `<built-in function getitem>` × 66
조건: no public EDF region rule
노드: `getitem`
```text
arian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 589, in forward
    query, key, value = torch.split(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.type_as.default` × 61
조건: no public EDF region rule
노드: `type_as`
```text
l-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 871, in forward
    hidden_states = self.input_layernorm(hidden_states)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 841, in forward
    return output.type_as(x)
.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.input_layernorm': ('model.model.layers.0.input_layernorm', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm')}
```

### `aten.view.default` × 60
조건: no public EDF region rule
노드: `view_2`
```text
ollector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 599, in forward
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.cat.default` × 26
조건: no public EDF region rule
노드: `cat`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1275, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 186, in forward
    sin = self.recomposition_frequencies(sin)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextRotaryEmbedding')}
```

### `aten.zeros.default` × 19
조건: no public EDF region rule
노드: `zeros`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1270, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel')}
```

### `aten.mul.Tensor` × 18
조건: pointwise requires BF16/FP32; integer arithmetic needs a separate precision contract
노드: `mul_5`
```text
, line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 543, in forward
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.squeeze.dim` × 18
조건: no public EDF region rule
노드: `squeeze`
```text
brarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 578, in forward
    self.conv1d.weight.squeeze(1),
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.conv1d.default` × 18
조건: qualified Conv1d is K3/pad1/dilation1/groups1/stride1-or-2
노드: `conv1d`
```text
ibrarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 576, in forward
    mixed_qkv = causal_conv1d_fn(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.split_with_sizes.default` × 18
조건: no public EDF region rule
노드: `split_with_sizes`
```text
arian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 589, in forward
    query, key, value = torch.split(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.softplus.default` × 18
조건: no public EDF region rule
노드: `softplus`
```text
tor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 605, in forward
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.ones.default` × 18
조건: no public EDF region rule
노드: `ones`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.triu.default` × 18
조건: no public EDF region rule
노드: `triu`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.cumsum.default` × 18
조건: no public EDF region rule
노드: `cumsum`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.masked_fill.Scalar` × 18
조건: no public EDF region rule
노드: `masked_fill`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.tril.default` × 18
조건: no public EDF region rule
노드: `tril`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.eye.default` × 18
조건: no public EDF region rule
노드: `eye`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten.zeros_like.default` × 18
조건: no public EDF region rule
노드: `zeros_like`
```text
ctor.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 875, in forward
    hidden_states = self.linear_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 625, in forward
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
re.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.0': ('model.model.layers.0', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.0.linear_attn': ('model.model.layers.0.linear_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet')}
```

### `aten._unsafe_view.default` × 18
조건: no public EDF region rule
노드: `_unsafe_view`
```text
model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 883, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 776, in forward
    gate = gate.reshape(*input_shape, -1)
__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.3': ('model.model.layers.3', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.3.self_attn': ('model.model.layers.3.self_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention')}
```

### `aten.matmul.default` × 12
조건: only 2D/3D dense contractions have candidate rules
노드: `matmul_24`
```text
ign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 883, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 792, in forward
    attn_output, attn_weights = attention_interface(
__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.3': ('model.model.layers.3', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.3.self_attn': ('model.model.layers.3.self_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention')}
```

### `aten.slice.Tensor` × 8
조건: layout slice supports positive unit step only
노드: `slice_3`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1275, in forward
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 186, in forward
    sin = self.recomposition_frequencies(sin)
{'L__self__': ('', '__main__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.rotary_emb': ('model.model.rotary_emb', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextRotaryEmbedding')}
```

### `aten.split.Tensor` × 6
조건: no public EDF region rule
노드: `split`
```text
ian/model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 883, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 773, in forward
    query_states, gate = torch.chunk(
__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.3': ('model.model.layers.3', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.3.self_attn': ('model.model.layers.3.self_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention')}
```

### `aten.clone.default` × 6
조건: no public EDF region rule
노드: `clone_378`
```text
model-campaign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 883, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 776, in forward
    gate = gate.reshape(*input_shape, -1)
__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.3': ('model.model.layers.3', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.3.self_attn': ('model.model.layers.3.self_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention')}
```

### `aten.softmax.int` × 6
조건: no public EDF region rule
노드: `softmax`
```text
ign-20261110/collector.py", line 64, in forward
    return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1709, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1278, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 883, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 792, in forward
    attn_output, attn_weights = attention_interface(
__.fixture.<locals>.Text'), 'L__self__model': ('model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM'), 'L__self__model.model': ('model.model', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel'), 'L__self__model.model.layers.3': ('model.model.layers.3', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer'), 'L__self__model.model.layers.3.self_attn': ('model.model.layers.3.self_attn', 'transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention')}
```

## resnet

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/resnet`
- 입력/모델: `{"model_id": "microsoft/resnet-18", "features": "convolution, batch norm, residual, pooling", "snapshot": "/root/.cache/huggingface/hub/models--microsoft--resnet-18/snapshots/65a5785d9156231087c481e0c7dd33a5ff6f7e3e", "revision": "65a5785d9156231087c481e0c7dd33a5ff6f7e3e", "input_scope": "synthetic deterministic smoke input; not task accuracy", "config_sha256": "11a01bcc873444bb433140e5f30f3687356641ed923266d738eef1c3438f6d22", "scope": "single 224x224 image logits"}`
- 판정: NPU=not_run; CPU fallback=unknown
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases resnet`
- 오류: `62 unresolved nodes; first={'node': 'conv2d', 'op': 'aten.conv2d.default', 'reason': 'no public EDF region rule', 'category': 'rule_or_contract_gap', 'detail': 'no public EDF region rule', 'module_stack': "'L__self__model.resnet': ('model.resnet', 'transformers.models.resnet.modeling_resnet.ResNetModel'), 'L__self__model.resnet.embedder': ('model.resnet.embedder', 'transformers.models.resnet.modeling_resnet.ResNetEmbeddings'), 'L__self__model.resnet.embedder.embedder': ('model.resnet.embedder.embedder', 'transformers.models.resnet.modeling_resnet.ResNetConvLayer'), 'L__self__model.resnet.embedder.embedder.convolution': ('model.resnet.embedder.embedder.convolution', 'torch.nn.modules.conv.Conv2d')}", 'source': 'deling_resnet.py", line 369, in forward\n    outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 314, in forward\n    embedding_output = self.embedder(pixel_values)\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 91, in forward\n    embedding = self.embedder(pixel_values)\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 66, in forward\n    hidden_states = self.convolution(hidden_states)\n  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 553, in forward\n    return self._conv_forward(i`
- 진단: {"observed_reason": "정적 실행 계획의 공유 EDF 규칙이 해당 연산/shape/dtype/layout을 수용하지 못함", "root_cause_status": "preflight guard confirmed; underlying SDK support unknown", "proposed_fix": "failure_groups별 최소 그래프를 분리하고 동일 shape/dtype의 공유 규칙을 추가·수치/trace 검증한다. 모델명 예외는 추가하지 않는다.", "next_evidence": ["graph.json의 해당 node와 preflight source를 대조", "최초 gap을 고친 뒤 재실행하여 뒤에 가려진 실패 수집", "CPU 기준값, 각 step NPU trace, 새 프로세스 재적재 검증"]}
- 지문: `23bbe1ec14732ebcf8c6`

### `aten.conv2d.default` × 20
조건: no public EDF region rule
노드: `conv2d`
```text
deling_resnet.py", line 369, in forward
    outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 314, in forward
    embedding_output = self.embedder(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 91, in forward
    embedding = self.embedder(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 66, in forward
    hidden_states = self.convolution(hidden_states)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 553, in forward
    return self._conv_forward(input, self.weight, self.bias)
'L__self__model.resnet': ('model.resnet', 'transformers.models.resnet.modeling_resnet.ResNetModel'), 'L__self__model.resnet.embedder': ('model.resnet.embedder', 'transformers.models.resnet.modeling_resnet.ResNetEmbeddings'), 'L__self__model.resnet.embedder.embedder': ('model.resnet.embedder.embedder', 'transformers.models.resnet.modeling_resnet.ResNetConvLayer'), 'L__self__model.resnet.embedder.embedder.convolution': ('model.resnet.embedder.embedder.convolution', 'torch.nn.modules.conv.Conv2d')}
```

### `aten._native_batch_norm_legit_no_training.default` × 20
조건: no public EDF region rule
노드: `_native_batch_norm_legit_no_training`
```text
transformers/models/resnet/modeling_resnet.py", line 369, in forward
    outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 314, in forward
    embedding_output = self.embedder(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 91, in forward
    embedding = self.embedder(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 67, in forward
    hidden_states = self.normalization(hidden_states)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/batchnorm.py", line 194, in forward
    return F.batch_norm(
l.resnet': ('model.resnet', 'transformers.models.resnet.modeling_resnet.ResNetModel'), 'L__self__model.resnet.embedder': ('model.resnet.embedder', 'transformers.models.resnet.modeling_resnet.ResNetEmbeddings'), 'L__self__model.resnet.embedder.embedder': ('model.resnet.embedder.embedder', 'transformers.models.resnet.modeling_resnet.ResNetConvLayer'), 'L__self__model.resnet.embedder.embedder.normalization': ('model.resnet.embedder.embedder.normalization', 'torch.nn.modules.batchnorm.BatchNorm2d')}
```

### `<built-in function getitem>` × 20
조건: no public EDF region rule
노드: `getitem`
```text
transformers/models/resnet/modeling_resnet.py", line 369, in forward
    outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 314, in forward
    embedding_output = self.embedder(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 91, in forward
    embedding = self.embedder(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 67, in forward
    hidden_states = self.normalization(hidden_states)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/batchnorm.py", line 194, in forward
    return F.batch_norm(
l.resnet': ('model.resnet', 'transformers.models.resnet.modeling_resnet.ResNetModel'), 'L__self__model.resnet.embedder': ('model.resnet.embedder', 'transformers.models.resnet.modeling_resnet.ResNetEmbeddings'), 'L__self__model.resnet.embedder.embedder': ('model.resnet.embedder.embedder', 'transformers.models.resnet.modeling_resnet.ResNetConvLayer'), 'L__self__model.resnet.embedder.embedder.normalization': ('model.resnet.embedder.embedder.normalization', 'torch.nn.modules.batchnorm.BatchNorm2d')}
```

### `aten.max_pool2d.default` × 1
조건: no public EDF region rule
노드: `max_pool2d`
```text
0/collector.py", line 68, in forward
    def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 369, in forward
    outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 314, in forward
    embedding_output = self.embedder(pixel_values)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 92, in forward
    embedding = self.pooler(embedding)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/pooling.py", line 224, in forward
    return F.max_pool2d(
{'L__self__': ('', '__main__.fixture.<locals>.Vision'), 'L__self__model': ('model', 'transformers.models.resnet.modeling_resnet.ResNetForImageClassification'), 'L__self__model.resnet': ('model.resnet', 'transformers.models.resnet.modeling_resnet.ResNetModel'), 'L__self__model.resnet.embedder': ('model.resnet.embedder', 'transformers.models.resnet.modeling_resnet.ResNetEmbeddings'), 'L__self__model.resnet.embedder.pooler': ('model.resnet.embedder.pooler', 'torch.nn.modules.pooling.MaxPool2d')}
```

### `aten.adaptive_avg_pool2d.default` × 1
조건: no public EDF region rule
노드: `adaptive_avg_pool2d`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 68, in forward
    def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 369, in forward
    outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/resnet/modeling_resnet.py", line 322, in forward
    pooled_output = self.pooler(last_hidden_state)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/pooling.py", line 1510, in forward
    return F.adaptive_avg_pool2d(input, self.output_size)
{'L__self__': ('', '__main__.fixture.<locals>.Vision'), 'L__self__model': ('model', 'transformers.models.resnet.modeling_resnet.ResNetForImageClassification'), 'L__self__model.resnet': ('model.resnet', 'transformers.models.resnet.modeling_resnet.ResNetModel'), 'L__self__model.resnet.pooler': ('model.resnet.pooler', 'torch.nn.modules.pooling.AdaptiveAvgPool2d')}
```

## segformer

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/segformer`
- 입력/모델: `{"model_id": "nvidia/segformer-b0-finetuned-ade-512-512", "features": "multiscale segmentation, resize, depthwise convolution", "snapshot": "/root/.cache/huggingface/hub/models--nvidia--segformer-b0-finetuned-ade-512-512/snapshots/489d5cd81a0b59fab9b7ea758d3548ebe99677da", "revision": "489d5cd81a0b59fab9b7ea758d3548ebe99677da", "input_scope": "synthetic deterministic smoke input; not task accuracy", "config_sha256": "209caa9091e4632f7c8883c11170cd08ad29af68b23c09590aa4a5befb1a2a7f", "scope": "single fixed resolution logits; segmentation deliberately 64x64 smoke geometry"}`
- 판정: NPU=not_run; CPU fallback=unknown
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases segformer`
- 오류: `conv2d_19: explicit packing required for noncontiguous tensors`
- 진단: {"observed_reason": "conv2d_19: explicit packing required for noncontiguous tensors", "root_cause_status": "tensor binding layout guard confirmed; required packing implementation unconfirmed", "proposed_fix": "graph.json에서 해당 노드의 shape/stride/consumer를 확인하고 명시적인 NPU packing 또는 검증된 layout 전달 규칙을 추가한다. 호스트 contiguous로 모델 연산을 숨기지 않는다.", "next_evidence": ["실패 텐서 shape/dtype/stride와 producer/consumer 연결", "최소 비연속 텐서 재현", "수정된 packing의 CPU 비교와 NPU trace"]}
- 지문: `1419b91e64c5d5182ade`

## whisper

- 증거: `/root/.cache/hf-librarian/model-campaign-20261110/attempts/whisper`
- 입력/모델: `{"model_id": "openai/whisper-tiny", "features": "audio Conv1d, encoder-decoder cross attention", "snapshot": "/root/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "input_scope": "synthetic deterministic smoke input; not task accuracy", "config_sha256": "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98", "scope": "full encoder + one decoder token; zero mel fixture, no transcription quality claim"}`
- 판정: NPU=not_run; CPU fallback=unknown
- 재현: `LIBRARIAN_PROVIDED_MODELS=/root/.cache/hf-librarian/model-campaign-20261110/provided-models.json /root/TCLMachine_test/.venv/bin/python /root/.cache/hf-librarian/model-campaign-20261110/collector.py run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases whisper`
- 오류: `74 unresolved nodes; first={'node': 'permute', 'op': 'aten.permute.default', 'reason': 'no public EDF region rule', 'category': 'rule_or_contract_gap', 'detail': 'no public EDF region rule', 'module_stack': "{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder')}", 'source': 'File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward\n    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward\n    outputs: Seq2SeqModelOutput = self.model(\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward\n    encoder_outputs = self.encoder(\n  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 618, in forward\n    inputs_embeds = inputs_embeds.permute(0, 2, 1)'}; details=/root/.cache/hf-librarian/model-campaign-20261110/attempts/whisper/model.fxb.plan/preflight.json`
- 진단: {"observed_reason": "정적 실행 계획의 공유 EDF 규칙이 해당 연산/shape/dtype/layout을 수용하지 못함", "root_cause_status": "preflight guard confirmed; underlying SDK support unknown", "proposed_fix": "failure_groups별 최소 그래프를 분리하고 동일 shape/dtype의 공유 규칙을 추가·수치/trace 검증한다. 모델명 예외는 추가하지 않는다.", "next_evidence": ["graph.json의 해당 node와 preflight source를 대조", "최초 gap을 고친 뒤 재실행하여 뒤에 가려진 실패 수집", "CPU 기준값, 각 step NPU trace, 새 프로세스 재적재 검증"]}
- 지문: `27b0d1fd76c54a4f7621`

### `aten.matmul.default` × 24
조건: only 2D/3D dense contractions have candidate rules
노드: `matmul`
```text
le "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 395, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 343, in forward
    attn_output, attn_weights = attention_interface(
('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.0': ('model.model.encoder.layers.0', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer'), 'L__self__model.model.encoder.layers.0.self_attn': ('model.model.encoder.layers.0.self_attn', 'transformers.models.whisper.modeling_whisper.WhisperAttention')}
```

### `aten.softmax.int` × 12
조건: no public EDF region rule
노드: `softmax`
```text
le "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 395, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 343, in forward
    attn_output, attn_weights = attention_interface(
('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.0': ('model.model.encoder.layers.0', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer'), 'L__self__model.model.encoder.layers.0.self_attn': ('model.model.encoder.layers.0.self_attn', 'transformers.models.whisper.modeling_whisper.WhisperAttention')}
```

### `aten.clone.default` × 10
조건: no public EDF region rule
노드: `clone`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 622, in forward
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder')}
```

### `aten.index.Tensor` × 3
조건: no public EDF region rule
노드: `index`
```text
paign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 781, in forward
    positions = self.embed_positions(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 213, in forward
    return self.weight[position_ids]
io'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder'), 'L__self__model.model.decoder.embed_positions': ('model.model.decoder.embed_positions', 'transformers.models.whisper.modeling_whisper.WhisperPositionalEmbedding')}
```

### `aten.__and__.Tensor` × 2
조건: no public EDF region rule
노드: `and_1`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.permute.default` × 1
조건: no public EDF region rule
노드: `permute`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 618, in forward
    inputs_embeds = inputs_embeds.permute(0, 2, 1)
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder')}
```

### `aten.embedding.default` × 1
조건: embedding requires BF16 table and 1D/2D int32 indices
노드: `embedding`
```text
paign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 621, in forward
    hidden_states = inputs_embeds + self.embed_positions(all_positions)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/sparse.py", line 191, in forward
    return F.embedding(
': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.embed_positions': ('model.model.encoder.embed_positions', 'torch.nn.modules.sparse.Embedding')}
```

### `aten.add.Tensor` × 1
조건: permute: explicit packing required for noncontiguous tensors
노드: `add`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 621, in forward
    hidden_states = inputs_embeds + self.embed_positions(all_positions)
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder')}
```

### `aten.add.Tensor` × 1
조건: clone: explicit packing required for noncontiguous tensors
노드: `add_3`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 401, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.0': ('model.model.encoder.layers.0', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.add.Tensor` × 1
조건: add_3: explicit packing required for noncontiguous tensors
노드: `add_6`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 409, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.0': ('model.model.encoder.layers.0', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.add.Tensor` × 1
조건: add_6: explicit packing required for noncontiguous tensors
노드: `add_9`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 401, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.1': ('model.model.encoder.layers.1', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.add.Tensor` × 1
조건: add_9: explicit packing required for noncontiguous tensors
노드: `add_12`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 409, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.1': ('model.model.encoder.layers.1', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.add.Tensor` × 1
조건: add_12: explicit packing required for noncontiguous tensors
노드: `add_15`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 401, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.2': ('model.model.encoder.layers.2', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.add.Tensor` × 1
조건: add_15: explicit packing required for noncontiguous tensors
노드: `add_18`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 409, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.2': ('model.model.encoder.layers.2', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.add.Tensor` × 1
조건: add_18: explicit packing required for noncontiguous tensors
노드: `add_21`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 401, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.3': ('model.model.encoder.layers.3', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.add.Tensor` × 1
조건: add_21: explicit packing required for noncontiguous tensors
노드: `add_24`
```text
-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 950, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 633, in forward
    hidden_states = encoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 409, in forward
    hidden_states = residual + hidden_states
.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.encoder': ('model.model.encoder', 'transformers.models.whisper.modeling_whisper.WhisperEncoder'), 'L__self__model.model.encoder.layers.3': ('model.model.encoder.layers.3', 'transformers.models.whisper.modeling_whisper.WhisperEncoderLayer')}
```

### `aten.repeat.default` × 1
조건: no public EDF region rule
노드: `repeat`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 777, in forward
    position_ids = position_ids.unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.alias.default` × 1
조건: no public EDF region rule
노드: `alias`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.sub.Tensor` × 1
조건: pointwise requires BF16/FP32; integer arithmetic needs a separate precision contract
노드: `sub_9`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.diff.default` × 1
조건: no public EDF region rule
노드: `diff`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.ne.Scalar` × 1
조건: no public EDF region rule
노드: `ne`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.cumsum.default` × 1
조건: no public EDF region rule
노드: `cumsum`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.new_ones.default` × 1
조건: no public EDF region rule
노드: `new_ones`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.le.Tensor` × 1
조건: no public EDF region rule
노드: `le`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.eq.Tensor` × 1
조건: no public EDF region rule
노드: `eq`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.expand.default` × 1
조건: no public EDF region rule
노드: `expand`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.zeros.default` × 1
조건: no public EDF region rule
노드: `zeros`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```

### `aten.where.ScalarOther` × 1
조건: no public EDF region rule
노드: `where`
```text
File "/root/.cache/hf-librarian/model-campaign-20261110/collector.py", line 72, in forward
    return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1110, in forward
    outputs: Seq2SeqModelOutput = self.model(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 962, in forward
    decoder_outputs = self.decoder(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 792, in forward
    causal_mask = create_causal_mask(
{'L__self__': ('', '__main__.fixture.<locals>.Audio'), 'L__self__model': ('model', 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'), 'L__self__model.model': ('model.model', 'transformers.models.whisper.modeling_whisper.WhisperModel'), 'L__self__model.model.decoder': ('model.model.decoder', 'transformers.models.whisper.modeling_whisper.WhisperDecoder')}
```
