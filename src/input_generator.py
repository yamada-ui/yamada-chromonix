import os
import random
from collections import Counter
import numpy as np
import tensorflow as tf
import keras
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
    
class TextEmbedding:
    def text_embeddings(self, file_path):
        with open(file_path, 'r', encoding='UTF-8') as f:
            text = f.readlines()
        for line in text:
            yield line
            
    def text(self, config):
        text_embeddings = []
        text = self.text_embeddings(config["text_embeddings_path"])
        for line in text:
            text_embeddings.append(float(conf) for conf in line.split(' '))
            
        return text_embeddings
            
                
class InputGenerator(keras.utils.Sequence):
    def __init__(self, config, tokenizer):
        self.config = config
        self.corpus = Corpus(config, tokenizer)
        self.corpus.gen_vocab()
        self.corpus.color()
        self.data = self.corpus.data
        self.text_embeddings = TextEmbedding()
        self.text_input_embeddings = self.text_embeddings.text(config)
        self.batch_size = self.config["batch_size"]
        self.mask_token_id = tokenizer.tokenizer.mask_token_id
        
    def __len__(self):
        return len(self.data) // self.batch_size
    
    def mask_tokens(self, batch_token_id):
        batch_size = len(batch_token_id)
        ignroe_pad = (np.array(batch_token_id) != self.corpus.tokenizer.pad_token_id).astype(int)
        ignore_sep = (np.array(batch_token_id) != self.corpus.tokenizer.sep_token_id).astype(int)
        batch_ignore = (ignroe_pad * ignore_sep).astype(int)
        
        
        seq_len = np.sum(batch_ignore, axis=1)
        mask_word = np.ceil(seq_len * self.config["mask_rate"]).astype(int)
        mask_position = []
        
        for i in range(batch_size):
            seq = [index for index, value in enumerate(batch_ignore[i]) if value >= 1]
            
            if len(self.config["mask_position"]) == 0:
                if random.random() < self.config["mask_position"]:
                    position = np.random.choice(seq, mask_word[i], replace=False)
                
                else:
                    position = []
            elif self.config["mask_position"] == "random":
                position = np.random.choice(seq, self.config["mask_num"] if self.config["mask_num"] < len(seq) else len(seq), replace=False)
            else:
                position = self.config["mask_position"]
            mask_position.append(np.sum(np.eye(self.config["max_seq_length"])[position], axis=0))
            
        mask_position = np.array(mask_position)
        
        mask_value = mask_position * self.mask_token_id
        mask = (mask_position == 0).astype(int)
        token_id_masked = (batch_token_id * mask + mask_value).astype(int)
        
        unmask = (mask_position == 1).astype(int)
        mask_class = (batch_token_id * unmask).astype(int)
        
        return token_id_masked, mask_position, mask_class
    
    def segment(self, token_id):
        segment = []
        
        for i in range(len(token_id)):
            sample_segment = [0, 0, 0, 0, 0, 0]
            segment.append(sample_segment)
        segment = np.array(segment)
        
        return segment