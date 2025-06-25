import torch


class WordLevelTokenizer:
    def __init__(self, vocab):
        self.vocab = sorted(vocab)
        self.vocab_size = len(vocab)
        self.encoder_mapping = {v: i for i, v in enumerate(vocab)}
        self.decoder_mapping = {v: k for k, v in self.encoder_mapping.items()}

    def encode(self, x):
        assert len(x) > 0
        x_encoded = []
        for x_ in x:
            x_encoded.append(self.encoder_mapping[x_])
        return x_encoded

    def decode(self, x):
        assert len(x) > 0
        x_decode = []
        for x_ in x:
            x_ = self.from_tensor(x_)
            x_decode.append(self.decoder_mapping[x_])
        return "".join(x_decode)

    def from_tensor(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        return x
