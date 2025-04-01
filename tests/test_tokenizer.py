import pytest
from fabrique import Tokenizer


def test_tokenizer():
    model_id = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = Tokenizer.from_pretrained(model_id)
    texts = [
        "Once upon a midnight dreary, while I pondered, weak and weary,",
        "Over many a quaint and curious volume of forgotten lore—",
        "While I nodded, nearly napping, suddenly there came a tapping,",
        "As of some one gently rapping, rapping at my chamber door.",
    ]
    # should error on unequal lengths
    with pytest.raises(ValueError):
        tokenizer(texts)

    # but not on a single text
    tokenizer(texts[0])

    # truncation
    tokens, mask = tokenizer(texts, max_length=10)
    assert tokens.shape == (4, 10)
    assert (mask == 1).all()

    # padding to a specific length
    tokens, mask = tokenizer(texts, padding_length=32)
    assert tokens.shape == (4, 32)
    assert (mask[:, 0] == 1).all()       # first token is always unmasked
    assert not (mask[:, -1] == 1).all()   # last token is sometimes masked

    # padding from the left
    tokens, mask = tokenizer(texts, padding_length=32, padding_side="left")
    assert tokens.shape == (4, 32)
    assert not (mask[:, 0] == 1).all()       # first token is always unmasked
    assert (mask[:, -1] == 1).all()   # last token is sometimes masked

    # padding to multiple of 16
    tokens, mask = tokenizer(texts, pad_to_multiple_of=16)
    assert tokens.shape == (4, 32)
    assert (mask[:, 0] == 1).all()       # first token is always unmasked
    assert not (mask[:, -1] == 1).all()   # last token is sometimes masked

    # check padding token
    tokens, mask = tokenizer(texts, padding_length=128)
    assert tokens[0, -1] == tokenizer.special_tokens.pad_id
