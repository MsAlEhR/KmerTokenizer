import itertools
import json
import os
import numpy as np
from transformers import PreTrainedTokenizer
import pickle

class KmerTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file=None, kmerlen=6, overlapping=True, maxlen=400, **kwargs):
        self.kmerlen = kmerlen
        self.overlapping = overlapping
        self.maxlen = maxlen
        
        # Lazy initialization for VOCAB and tokendict
        self._vocab = None
        self._tokendict = None
        
        super().__init__(**kwargs)

    @property
    def VOCAB(self):
        if self._vocab is None:
            self._vocab = [''.join(i) for i in itertools.product(*(['ACTG'] * int(self.kmerlen)))]
        return self._vocab

    @property
    def tokendict(self):
        if self._tokendict is None:
            self._tokendict = dict(zip(self.VOCAB, range(5, len(self.VOCAB) + 5)))
            self._tokendict.update({'[UNK]': 0, '[SEP]': 1, '[PAD]': 2,'[CLS]': 3, '[MASK]': 4})
        return self._tokendict

    def _tokenize(self, text):
        stoprange = len(text) - (self.kmerlen - 1)
        tokens = [text[k:k + self.kmerlen] for k in range(0, stoprange, 1 if self.overlapping else self.kmerlen)]
        return [token for token in tokens if set(token).issubset('ACTG')]

    def _convert_token_to_id(self, token):
        return np.int16(self.tokendict.get(token, self.tokendict['[UNK]']))

    def _convert_id_to_token(self, index):
        return next((k for k, v in self.tokendict.items() if v == index), '[UNK]')

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        sep = [self.tokendict['[SEP]']]
        cls = [self.tokendict['[CLS]']]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def kmer_tokenize(self, seq):
        tokens = self._tokenize(seq)
        # Convert tokens to int16 token IDs
        token_ids = np.array([self._convert_token_to_id(token) for token in tokens], dtype=np.int16)
        if len(token_ids) < self.maxlen:
            # Padding to ensure the output is of maxlen size
            token_ids = np.pad(token_ids, (0, self.maxlen - len(token_ids)), 'constant', constant_values=self.tokendict['[PAD]'])
        else:
            # Truncate to maxlen
            token_ids = token_ids[:self.maxlen]
        return token_ids  # Return a single 1D array

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + 'vocab.pkl')
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.tokendict, f)
        
        return (vocab_file,)

    def save_pretrained(self, save_directory, **kwargs):
        special_tokens_map_file = os.path.join(save_directory, "special_tokens_map.pkl")
        with open(special_tokens_map_file, "wb") as f:
            pickle.dump({
                "kmerlen": self.kmerlen,
                "overlapping": self.overlapping,
                "maxlen": self.maxlen
            }, f)
        vocab_files = self.save_vocabulary(save_directory)
        return (special_tokens_map_file,) + vocab_files

    def get_vocab(self):
        return self.tokendict

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        special_tokens_map_file = os.path.join(pretrained_model_name_or_path, "special_tokens_map.pkl")
        if os.path.isfile(special_tokens_map_file):
            with open(special_tokens_map_file, "rb") as f:
                special_tokens_map = pickle.load(f)
            tokenizer.kmerlen = special_tokens_map.get("kmerlen", 6)
            tokenizer.overlapping = special_tokens_map.get("overlapping", True)
            tokenizer.maxlen = special_tokens_map.get("maxlen", 400)
        
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.pkl")
        if os.path.isfile(vocab_file):
            with open(vocab_file, "rb") as f:
                tokendict = pickle.load(f)
            tokenizer._tokendict = tokendict
        
        return tokenizer
