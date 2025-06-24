import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.tokenizer import WordLevelTokenizer


def test_encode_decode():
    data = "Hello world! How are you?"
    vocab = sorted(list(set(data)))
    tok = WordLevelTokenizer(vocab)
    x = data[:10]
    x_enc = tok.encode(x)
    x_dec = tok.decode(x_enc)
    assert x_dec == x


if __name__ == "__main__":
    print("All test passed!")
