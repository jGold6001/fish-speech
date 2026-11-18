"""Fish Speech 1.5 prompts for the shared DualAR transformer.

The 1.5 codec, tokenizer and text cleaning are restored from repository revision
58046eaa1a4cefb0c8cc3a3a667b34186ea02dde (before the S1 migration).
"""

from pathlib import Path

import torch

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.legacy_text import clean_text, split_text

FIREFLY_CHECKPOINT = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"


def is_legacy_checkpoint(path):
    return (Path(path) / "tokenizer.tiktoken").is_file()


def encode_legacy_text(tokenizer, text, num_codebooks):
    tokens = tokenizer.encode(text)
    values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
    values[0] = torch.tensor(tokens, dtype=torch.int)
    return values


def encode_legacy_turn(tokenizer, text, num_codebooks, codes=None):
    """1.5 has no newline after im_end and keeps the first codebook in row 1."""
    values = encode_legacy_text(
        tokenizer,
        f"<|im_start|>user\n{clean_text(text)}<|im_end|>"
        "<|im_start|>assistant\n<|voice|>",
        num_codebooks,
    )
    if codes is None:
        return values
    codes = codes.detach().cpu().int()
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]
    if codes.ndim != 2 or codes.shape[0] != num_codebooks:
        raise InvalidInputError(f"Expected {num_codebooks} reference codebooks")
    audio = torch.cat((codes[:1] + tokenizer.semantic_begin_id, codes), dim=0)
    end = encode_legacy_text(tokenizer, "<|im_end|>", num_codebooks)
    return torch.cat((values, audio, end), dim=1)


def generate_long_legacy(
    *,
    model,
    device,
    decode_one_token,
    text,
    num_samples,
    max_new_tokens,
    top_p,
    top_k,
    temperature,
    iterative_prompt,
    chunk_length,
    prompt_text,
    prompt_tokens,
    single_chunk,
):
    # Imported here because the shared generation module dispatches to us.
    from fish_speech.models.text2semantic.inference import (
        GenerateResponse,
        generate,
        generated_codes,
    )

    if isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]
    prompt_text = list(prompt_text or [])
    prompt_tokens = list(prompt_tokens or [])
    if len(prompt_text) != len(prompt_tokens):
        raise InvalidInputError("Prompt text and tokens must have the same length")

    tokenizer = model.tokenizer
    num_codebooks = model.config.num_codebooks
    im_end_id = tokenizer.get_token_id("<|im_end|>")
    references = []
    if prompt_text:
        references.append(
            encode_legacy_text(
                tokenizer,
                "<|im_start|>system\nSpeak out the provided text.<|im_end|>",
                num_codebooks,
            )
        )
        references.extend(
            encode_legacy_turn(tokenizer, transcript, num_codebooks, codes)
            for transcript, codes in zip(prompt_text, prompt_tokens)
        )
    chunks = (
        split_text(text, chunk_length)
        if iterative_prompt and chunk_length > 0 and not single_chunk
        else [clean_text(text)]
    )
    if not chunks:
        raise InvalidInputError("No text to synthesize")

    for _ in range(num_samples):
        history = []
        for chunk in chunks:
            current = encode_legacy_turn(tokenizer, chunk, num_codebooks)
            # Keep complete turns, preferentially retaining the first voice sample.
            available = model.config.max_seq_len - max(1, max_new_tokens)
            while history and sum(
                p.shape[1] for p in references + history + [current]
            ) > available:
                history.pop(1 if len(history) > 1 else 0)
            prompt = torch.cat(references + history + [current], dim=1).to(device)
            y = generate(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                audio_masks=None,
                audio_parts=None,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            codes = generated_codes(y, prompt.shape[1], im_end_id)
            if codes.shape[1]:
                history.append(encode_legacy_turn(tokenizer, chunk, num_codebooks, codes))
                yield GenerateResponse(action="sample", codes=codes, text=chunk)
        yield GenerateResponse(action="next")
