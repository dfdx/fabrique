from typing import List, Dict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jinja2 import Environment
from multimethod import multimethod

from fabrique.generation import sample
from fabrique.loading import from_pretrained

import logging
logger = logging.getLogger("fabrique")


@dataclass
class ChatMessage:
    role: str
    content: str

    @staticmethod
    def from_dict(d: Dict):
        return ChatMessage(**d)

    def dict(self):
        return {"role": self.role, "content": self.content}


class LLM:

    def __init__(self, tokenizer, model, hf_config: dict):
        self.tokenizer = tokenizer
        self.model = model
        self.hf_config = hf_config
        self.rngs = nnx.Rngs(0)
        self.chat_template = None
        if tmpl := self.hf_config.get("chat_template"):
            self.chat_template = Environment().from_string(tmpl)

    @staticmethod
    def from_pretrained(repo_id: str, revision: str | None = None, **model_args):
        tokenizer, model, hf_config = from_pretrained(
            repo_id, revision=revision, **model_args
        )
        return LLM(tokenizer, model, hf_config)

    def apply_chat_template(self, messages: List[ChatMessage]):
        msg_dicts = [msg.dict() for msg in messages]
        return self.chat_template.render(messages=msg_dicts).strip()


    def collate_with_padding(self, prompt_token_list: List[List[int]], pad_token_id: int):
        if len(prompt_token_list) > 1:
            logger.warning(
                "Padding sequences from the beginning. Results for shorter " +
                "sequences may or may not be meaningful depending on the model."
            )
        max_tokens = max(len(tokens) for tokens in prompt_token_list)
        return jnp.array([
            [pad_token_id] * (max_tokens - len(tokens)) + tokens
            for tokens in prompt_token_list
        ])

    @multimethod
    def generate(
        self,
        prompts: List[str],
        new_only: bool = True,
        max_length: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        prng_key: jax.Array | None = None,
        skip_special_tokens: bool = True
    ):
        eos_token_id = self.hf_config["eos_token_id"]
        pad_token_id = self.hf_config.get("pad_token_id", eos_token_id)
        prompt_token_list = [enc.ids for enc in self.tokenizer.encode_batch(prompts)]
        prompt_tokens = self.collate_with_padding(prompt_token_list, pad_token_id)
        sequences = sample(
            self.model,
            prompt_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prng_key=prng_key if prng_key is not None else self.rngs(),
        )
        out = []
        for prompt, seq in zip(prompt_token_list, sequences):
            if new_only:
                seq = seq[len(prompt):]
            generated = self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
            out.append(generated)
        return out

    @multimethod
    def generate(self, prompt: str, **kwargs):
        return self.generate([prompt], **kwargs)

    @multimethod
    def generate(
        self,
        messages: List[ChatMessage],
        **kwargs
    ):
        prompt = self.apply_chat_template(messages)
        result = self.generate(prompt, **kwargs)


####################################################


def test_inference():
    model_id = "meta-llama/Meta-Llama-3-8B"
    model_args = {"max_seq_len": 512, "max_batch_size": 1, "dtype": jnp.bfloat16}  # limit cache size
    llm = LLM.from_pretrained(model_id,  **model_args)
    kwargs = {"max_length": 512}
    self = llm
    prng_key = key = jax.random.key(94)
    new_only: bool = True
    max_length: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    prompts = ["What is the meaning of life?", "Write a tanku"]

    prompts = [self.apply_chat_template(messages)]
    result = self.generate(prompts, skip_special_tokens=False)


    result = llm.generate("What is the meaning of life?", new_only=False, max_length=32, prng_key=key)
    assert isinstance(result, str)   # smoke test

    messages = [ChatMessage(role="user", content="What is the meaning of life?")]

    result = llm.generate(messages, new_only=False, max_length=32, prng_key=key)
    assert isinstance(result, ChatMessage)