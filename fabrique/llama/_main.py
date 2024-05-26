import jax
import jax.numpy as jnp


def top_level(x: jax.Array):
    print("compiling")
    return x * 2


def outer(a: jax.Array):
    def inner(x: jax.Array):
        print("compiling")
        return x * 2

    print(f"id(inner) = {id(inner)}")
    print(f"hash(inner) = {hash(inner)}")
    jitted_inner = jax.jit(inner)
    return jitted_inner(a)


def main():
    a = jax.random.normal(jax.random.key(0), (3, 4))
    outer(a)


def main():
    import os
    import sys

    print(os.getcwd())
    print(sys.path)
    sys.path.append("/workspaces/fabrique")

    from fabrique.generation import greedy
    from fabrique.llama import Llama

    model_id = "meta-llama/Meta-Llama-3-8B"
    kwargs = {"max_seq_len": 512, "max_batch_size": 1}
    llama = Llama.from_pretrained(model_id, **kwargs)
    model, variables = llama.model, llama.variables

    prompt = "Albert Einstein was born in"
    prompt_tokens = llama.tokenizer.encode(prompt).ids
    prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)
    sequences = greedy(
        model,
        variables,
        prompt_tokens,
        pad_token_id=llama.hf_config["eos_token_id"],
        eos_token_id=llama.hf_config["eos_token_id"],
        max_length=llama.model.args.max_seq_len,
    )
    out = llama.tokenizer.decode(sequences[0])
    print(out)


main()
