import os
import random
from collections import Counter
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

class Tokenizer:
    def __init__(self, vocab_path):
        self.tokenizer = BertTokenizer.from_pretrained("facebook/bart-base")
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_path = vocab_path

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
class Corpus:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = []

    def gen_vocab(self):
        if not os.path.exists(self.tokenizer.vocab_path):
            with open(self.config["corpus_path"], "r", encoding="UTF-8") as f:
                corpus = f.read()
            vocab_with_freq = Counter(corpus).most_common()
            vocab = [char for (char, freq) in vocab_with_freq if freq >= self.config["charactor_freq_threshold"]]

            with open(self.tokenizer.vocab_path, "w", encoding="UTF-8") as f:
                f.write(str(vocab))

    # make and parse
    def passages(self):
        with open(self.config["corpus_path"], "r", encoding="UTF-8") as f:
            corpus = f.read()
        
        for line in corpus:
            yield line.replace('"', "")
        
    def color(self):
        passages = self.passages()

        for passage in passages:
            sentence = passage.strip('\n').split(' ; ')
            sample = []
            for i in range(len(sentence)):
                sample.extend(self.tokenizer.encode(sentence[i]))
            
                if i < self.config["max_palette_length"]:
                    sample.extend([self.tokenizer.vocab["[PAD]"]] * (self.config["max_palette_length"] - i))

            self.data.append(sample)

    def token_id_to_list(self):
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in self.data]