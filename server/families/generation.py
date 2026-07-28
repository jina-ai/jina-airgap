"""Generative families: everything served at /v1/chat/completions."""

import logging
from abc import abstractmethod

import torch

from config import settings

from . import Family

logger = logging.getLogger(__name__)


class ChatFamily(Family):
    verb, kind, endpoint = "generate", "chat", "/v1/chat/completions"

    accepts_images = False
    # use_model_defaults was added in transformers >=4.51; reader-lm pins 4.48.3.
    generate_kwargs: dict = {}

    @abstractmethod
    def encode_prompt(self, conversation: list, images: list, max_length: int):
        """Template and tokenize the conversation into model inputs."""

    @property
    @abstractmethod
    def decoder_tokenizer(self):
        """Tokenizer used to decode generated ids back to text."""

    def _load_causal_lm(self) -> None:
        from transformers import AutoModelForCausalLM

        # Target dtype: fp16 on cuda (no bf16 native on L4), fp32 on cpu/mps.
        if settings.device == "cuda":
            dtype = (
                torch.bfloat16
                if settings.dtype in ("bfloat16", "bf16")
                else torch.float16
            )
        else:
            dtype = torch.float32
        # Flash-attn is intentionally not installed (no nvcc/git in the runtime
        # image); sdpa is the supported fallback and matches HF's documented
        # CPU/non-fa path.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_repo,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=settings.device if settings.device != "mps" else None,
            attn_implementation="sdpa",
        )
        if settings.device == "mps":
            self.model = self.model.to(settings.device)
        self.model.eval()
        logger.info(
            f"Loaded {self.spec.hf_repo} as {type(self).__name__} dtype={dtype}"
        )


class VlmFamily(ChatFamily):
    accepts_images = True
    generate_kwargs = {"use_model_defaults": True}

    def _load(self) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.spec.hf_repo, trust_remote_code=True, use_fast=False
        )
        self._load_causal_lm()

    def encode_prompt(self, conversation: list, images: list, max_length: int):
        # apply_chat_template returns the templated string for ONE conversation
        # when given a single list.
        texts = self.processor.apply_chat_template(
            [conversation], add_generation_prompt=True
        )
        return self.processor(
            text=texts,
            images=[images] if images else None,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        )

    @property
    def decoder_tokenizer(self):
        return self.processor.tokenizer


class TextChatFamily(ChatFamily):
    """Text-only chat / reader models: no vision tower, no AutoProcessor."""

    def _load(self) -> None:
        self._load_causal_lm()

    def encode_prompt(self, conversation: list, images: list, max_length: int):
        # AutoTokenizer's apply_chat_template only needs role + text.
        messages = [
            {
                "role": message["role"],
                "content": "".join(
                    part.get("text", "")
                    for part in message["content"]
                    if part.get("type") == "text"
                ),
            }
            for message in conversation
        ]
        templated = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return self.tokenizer(
            templated, return_tensors="pt", truncation=True, max_length=max_length
        )

    @property
    def decoder_tokenizer(self):
        return self.tokenizer
