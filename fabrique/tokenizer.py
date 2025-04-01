import os
import json
import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer as HFTokenizer



    # def collate_with_padding(
    #     self, prompt_token_list: List[List[int]], pad_token_id: int
    # ):
    #     if len(prompt_token_list) > 1:
    #         logger.warning(
    #             "Padding sequences from the beginning. Results for shorter "
    #             + "sequences may or may not be meaningful depending on the model."
    #         )
    #     max_tokens = max(len(tokens) for tokens in prompt_token_list)
    #     return jnp.array(
    #         [
    #             [pad_token_id] * (max_tokens - len(tokens)) + tokens
    #             for tokens in prompt_token_list
    #         ]
    #     )



class SpecialTokenMap:

    def __init__(self, hf_config: dict):
        # BOS
        self.bos_id = hf_config["bos_token_id"]
        # PAD, may not present
        self.pad_id = hf_config.get("pad_token_id")
        # EOS, may be a single int or a tuple
        self.eos_ids = hf_config["eos_token_id"]
        if isinstance(self.eos_ids, int):
            self.eos_ids = (self.eos_ids,)
        elif isinstance(self.eos_ids, list):
            self.eos_ids = tuple(self.eos_ids)
        self.eos_id = self.eos_ids[0]


# token_lists = [enc.ids for enc in llm.tokenizer.encode_batch(texts)]
#     max_length = max(len(token_list) for token_list in token_lists)
#     # align to multiple of certain value to minimize re-compilation for every length
#     max_length = math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of
#     pad_token_id = llm.special_tokens.pad_id or llm.special_tokens.eos_id
#     token_lists = [
#         token_list + [pad_token_id] * (max_length - len(token_list))
#         for token_list in token_lists
#     ]
#     token_lists = [token_list[:truncate] for token_list in token_lists]
#     tokens = jnp.array(token_lists)


class Tokenizer:
    """
    Convenient wrapper around HuggingFace's tokenizers.Tokenizer.

    Generally, this class follows the logic of a more high-level
    transformers.PreTrainedTokenizer without depending on the whole
    transformers library.
    """

    def __init__(self, hf_tokenizer: HFTokenizer, hf_config: dict):
        self.hf = hf_tokenizer
        self.hf_config = hf_config
        self.special_tokens = SpecialTokenMap(hf_config)

    @staticmethod
    def from_model_dir(model_dir: str):
        config_file = os.path.join(model_dir, "config.json")
        with open(config_file) as fp:
            hf_config = json.load(fp)
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        hf_tokenizer = HFTokenizer.from_file(tokenizer_file)
        return Tokenizer(hf_tokenizer, hf_config)

    @staticmethod
    def from_pretrained(model_id: str, revision=None):
        model_dir = snapshot_download(model_id, revision=revision, repo_type="model")
        return Tokenizer.from_model_dir(model_dir)

    def __repr__(self):
        return f"fabrique.Tokenizer(vocab_size={self.hf.get_vocab_size()})"

    def __call__(
        self,
        texts: str | list[str],
        padding_side: str = "right",
        pad_to_multiple_of: int | None = None,
        padding_length: int | None = None,
        max_length: int | None = None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        # (re-)configure HF tokenizer
        if padding_length or pad_to_multiple_of:
            self.hf.enable_padding(
                direction=padding_side,
                pad_id=self.special_tokens.pad_id,
                pad_to_multiple_of=pad_to_multiple_of,
                length=padding_length,
            )
        else:
            self.hf.no_padding()
        if max_length:
            self.hf.enable_truncation(
                direction=padding_side,
                max_length=max_length,
            )
        else:
            self.hf.no_truncation()
        # encode batch
        encodings = self.hf.encode_batch(texts)
        lengths = [len(e.ids) for e in encodings]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                "Tokenization produced sequences of different lengths, " +
                "which cannot be combined into a single array. Please, " +
                "use padding and/or truncation to align the sequences. " +
                f"Lengths of sequences were: {lengths}"

            )
        tokens = jnp.asarray([e.ids for e in encodings])
        padding_mask = jnp.array([e.attention_mask for e in encodings])
        return tokens, padding_mask

    def decode(self, tokens: jax.Array, skip_special_tokens=True):
        return self.hf.decode_batch(tokens, skip_special_tokens=skip_special_tokens)