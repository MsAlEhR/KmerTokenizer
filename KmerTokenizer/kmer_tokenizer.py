import time
import itertools
from transformers import PreTrainedTokenizer, AutoTokenizer
import json
import os

class KmerTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file=None, kmerlen=6, overlapping=True, maxlen=400, **kwargs):
        self.kmerlen = kmerlen
        self.overlapping = overlapping
        self.maxlen = maxlen
        
        # Initialize vocabulary
        self.VOCAB = [''.join(i) for i in itertools.product(*(['ATCG'] * int(self.kmerlen)))]
        self.VOCAB_SIZE = len(self.VOCAB) + 5
        
        self.tokendict = dict(zip(self.VOCAB, range(5, self.VOCAB_SIZE)))
        self.tokendict['[UNK]'] = 0
        self.tokendict['[SEP]'] = 1
        self.tokendict['[CLS]'] = 2
        self.tokendict['[MASK]'] = 3
        self.tokendict['[PAD]'] = 4
        
        super().__init__(**kwargs)

    def _tokenize(self, text):
        tokens = []
        stoprange = len(text) - (self.kmerlen - 1)
        if self.overlapping:
            for k in range(0, stoprange):
                kmer = text[k:k + self.kmerlen]
                if set(kmer).issubset('ATCG'):
                    tokens.append(kmer)
        else:
            for k in range(0, stoprange, self.kmerlen):
                kmer = text[k:k + self.kmerlen]
                if set(kmer).issubset('ATCG'):
                    tokens.append(kmer)
        return tokens

    def _convert_token_to_id(self, token):
        return self.tokendict.get(token, self.tokendict['[UNK]'])

    def _convert_id_to_token(self, index):
        inv_tokendict = {v: k for k, v in self.tokendict.items()}
        return inv_tokendict.get(index, '[UNK]')

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.tokendict['[CLS]']] + token_ids_0 + [self.tokendict['[SEP]']]
        return [self.tokendict['[CLS]']] + token_ids_0 + [self.tokendict['[SEP]']] + token_ids_1 + [self.tokendict['[SEP]']]

    def get_vocab(self):
        return self.tokendict

    def kmer_tokenize(self, seq_list):
        seq_ind_list = []
        for seq in seq_list:
            tokens = self._tokenize(seq)
            token_ids = [self._convert_token_to_id(token) for token in tokens]
            if len(token_ids) < self.maxlen:
                token_ids.extend([self.tokendict['[PAD]']] * (self.maxlen - len(token_ids)))
            else:
                token_ids = token_ids[:self.maxlen]
            seq_ind_list.append(token_ids)
        return seq_ind_list

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + 'vocab.json')
        
        with open(vocab_file, 'w') as f:
            json.dump(self.tokendict, f)
        
        return (vocab_file,)

    def save_pretrained(self, save_directory, **kwargs):
        special_tokens_map_file = os.path.join(save_directory, "special_tokens_map.json")
        with open(special_tokens_map_file, "w") as f:
            json.dump({
                "kmerlen": self.kmerlen,
                "overlapping": self.overlapping,
                "maxlen": self.maxlen
            }, f)
        vocab_files = self.save_vocabulary(save_directory)
        return (special_tokens_map_file,) + vocab_files

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load tokenizer using the parent class method
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Load special tokens map
        special_tokens_map_file = os.path.join(pretrained_model_name_or_path, "special_tokens_map.json")
        if os.path.isfile(special_tokens_map_file):
            with open(special_tokens_map_file, "r") as f:
                special_tokens_map = json.load(f)
            tokenizer.kmerlen = special_tokens_map.get("kmerlen", 6)
            tokenizer.overlapping = special_tokens_map.get("overlapping", True)
            tokenizer.maxlen = special_tokens_map.get("maxlen", 400)
        
        # Load vocabulary
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        if os.path.isfile(vocab_file):
            with open(vocab_file, "r") as f:
                tokendict = json.load(f)
            tokenizer.tokendict = tokendict
        
        return tokenizer