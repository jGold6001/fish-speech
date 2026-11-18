import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.inference import generate_long, resolve_codec_path
from fish_speech.models.text2semantic.legacy import (
    FIREFLY_CHECKPOINT,
    encode_legacy_turn,
)
from fish_speech.tokenizer import FishTokenizer


class LegacyCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        # A byte vocabulary keeps these tests independent of downloaded weights.
        (self.path / "tokenizer.tiktoken").write_text("".join(
            f"{base64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256)
        ))
        self.tokenizer = FishTokenizer.from_pretrained(self.path)
        self.model = SimpleNamespace(
            tokenizer=self.tokenizer,
            config=SimpleNamespace(num_codebooks=2, max_seq_len=512),
        )

    def test_local_tiktoken_and_special_tokens(self):
        text = "Привіт!<|im_end|>"
        self.assertTrue(self.tokenizer.is_legacy)
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)
        self.assertEqual(
            self.tokenizer.encode("<|semantic:3|>"),
            [self.tokenizer.semantic_begin_id + 3],
        )

    def test_codec_resolution_uses_firefly(self):
        firefly = self.path / FIREFLY_CHECKPOINT
        firefly.touch()
        (self.path / "codec.pth").touch()
        self.assertEqual(resolve_codec_path(self.path), firefly)

    def test_missing_legacy_codec_does_not_use_dac(self):
        (self.path / "codec.pth").touch()
        with self.assertRaisesRegex(FileNotFoundError, "Firefly"):
            resolve_codec_path(self.path)

    def test_reference_turn_preserves_all_codebooks(self):
        codes = torch.tensor([[3, 4], [5, 6]])
        turn = encode_legacy_turn(self.tokenizer, "Reference.", 2, codes)
        self.assertTrue(torch.equal(turn[1:, -3:-1], codes))
        self.assertTrue(torch.equal(
            turn[0, -3:-1], codes[0] + self.tokenizer.semantic_begin_id,
        ))
        self.assertEqual(turn[0, -1], self.tokenizer.get_token_id("<|im_end|>"))
        text = self.tokenizer.decode(turn[0, :-3].tolist())
        self.assertEqual(
            text, "<|im_start|>user\nReference.<|im_end|>"
            "<|im_start|>assistant\n<|voice|>",
        )

    def test_reference_rejects_incompatible_codebooks(self):
        with self.assertRaises(InvalidInputError):
            encode_legacy_turn(self.tokenizer, "Reference.", 2, torch.zeros(3, 2))

    def test_legacy_dispatch_and_eos_trimming(self):
        for eos in (False, True):
            with self.subTest(eos=eos):
                seen = []

                def generate(**kwargs):
                    prompt = kwargs["prompt"]
                    seen.append(prompt)
                    audio = torch.tensor([
                        [self.tokenizer.semantic_begin_id + 3], [3], [5],
                    ])
                    parts = [prompt, audio]
                    if eos:
                        parts.append(torch.tensor([
                            [self.tokenizer.get_token_id("<|im_end|>")], [0], [0],
                        ]))
                    return torch.cat(parts, dim=1)

                with patch(
                    "fish_speech.models.text2semantic.inference.generate", generate,
                ):
                    responses = list(generate_long(
                        model=self.model, device="cpu", decode_one_token=None,
                        text="Привіт!", single_chunk=True,
                    ))
                self.assertEqual([r.action for r in responses], ["sample", "next"])
                self.assertTrue(torch.equal(responses[0].codes, torch.tensor([[3], [5]])))
                self.assertEqual(
                    self.tokenizer.decode(seen[0][0].tolist()),
                    "<|im_start|>user\nПривіт!<|im_end|>"
                    "<|im_start|>assistant\n<|voice|>",
                )

    def test_reference_prompt_has_legacy_system_instruction(self):
        seen = []

        def generate(**kwargs):
            seen.append(kwargs["prompt"])
            return kwargs["prompt"]

        with patch("fish_speech.models.text2semantic.inference.generate", generate):
            list(generate_long(
                model=self.model, device="cpu", decode_one_token=None,
                text="Привіт!", single_chunk=True,
                prompt_text=["Reference."], prompt_tokens=[torch.tensor([[1], [2]])],
            ))
        text = self.tokenizer.decode(seen[0][0].tolist())
        self.assertTrue(text.startswith(
            "<|im_start|>system\nSpeak out the provided text.<|im_end|>",
        ))
        self.assertIn("Reference.", text)
        self.assertNotIn("<|speaker:", text)


if __name__ == "__main__":
    unittest.main()
