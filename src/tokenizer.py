class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

        self.token_to_id = {
            token: i
            for i, token in enumerate(vocab)
        }

        self.id_to_token = {
            i: token
            for i, token in enumerate(vocab)
        }

    def encode(self, text):
        return [self.token_to_id[character] for character in text]

    def decode(self, ids):
        return "".join(self.id_to_token[i] for i in ids)