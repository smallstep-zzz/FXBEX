"""Bounded, greedy generation against the bundled Qwen runtime."""


def generate(adapter, prompt, max_tokens, on_token=None):
    """Generate from raw text or chat messages; callback receives (text, token_ids).

    Callback values are cumulative snapshots. ``num_cached_tokens`` counts prompt
    tokens reused from the preceding call. EOS is not included in output IDs.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError('max_tokens must be a nonnegative integer')
    runtime, tokenizer = adapter.runtime, adapter.tokenizer
    if isinstance(prompt, str):
        ids = list(tokenizer.encode(prompt))
    elif isinstance(prompt, list):
        ids = list(tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False))
    else:
        raise ValueError('prompt must be text or a list of chat messages')
    if not ids:
        raise ValueError('prompt must contain at least one token')
    if len(ids) + max_tokens > runtime.capacity:
        raise ValueError(f'input {len(ids)} + output {max_tokens} exceeds context limit {runtime.capacity}')

    reused = len(runtime.consumed_ids)
    # Recompute the final prompt token when no new input supplies its logits.
    if not reused or reused >= len(ids) or ids[:reused] != runtime.consumed_ids:
        runtime.reset()
        reused = 0
    for index in range(reused, len(ids)):
        logits = runtime.step(ids[index], return_logits=(index == len(ids) - 1))

    eos = tokenizer.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    output = []
    text = ''
    finish_reason = 'length'
    for index in range(max_tokens):
        token = int(logits.argmax())
        if token in eos:
            finish_reason = 'stop'
            break
        output.append(token)
        # Consume even the final output token so prefix reuse describes real state.
        logits = runtime.step(token, return_logits=(index + 1 < max_tokens))
        decoded = tokenizer.decode(output, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        # Byte-piece tokenizers may temporarily end in an incomplete UTF-8 codepoint.
        safe_text = decoded.split('\ufffd', 1)[0]
        if safe_text != text:
            text = safe_text
            if on_token is not None:
                on_token(text, list(output))
    final=tokenizer.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    if final!=text and on_token is not None:on_token(final,list(output))
    return {'text': final, 'prompt':prompt if isinstance(prompt,str) else tokenizer.decode(ids,skip_special_tokens=False), 'prompt_token_ids': ids, 'token_ids': output,
            'finish_reason': finish_reason, 'num_cached_tokens': reused}
