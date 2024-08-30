# KmerTokenizer

KmerTokenizer is a Python package for k-mer tokenization.

## Installation

You can install the package via pip:

```sh
pip install git+https://github.com/MsAlEhR/KmerTokenizer.git
```



## Usage
```py
from KmerTokenizer import KmerTokenizer
import torch

seq_list = ["ATTTTTTTTTTTCCCCCCCCCCCGGGGGGGGATCGATGC"]

# Test loading the tokenizer
tokenizer = KmerTokenizer(kmerlen=6, overlapping=True, maxlen=4096)

# Tokenize the sequence
tokenized_output = tokenizer.kmer_tokenize(seq_list)

# Convert the tokenized output to a tensor
inputs = torch.tensor(tokenized_output)
print(inputs)
```

