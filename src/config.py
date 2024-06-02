import os

PROJECT_PATH = "../data/"
bin_range = 16
text_model = "_text"
embedding_path = "embedding_text"
representation = "CIELAB"

Config = {
    "project_path": PROJECT_PATH,
    "bin_range": bin_range,
    "representation": representation,
    "text_model": text_model,
    "embedding_file": embedding_path,
    "corpus_path": os.path.join(PROJECT_PATH, "corpus.txt"),
    "vocabularies": os.path.join(PROJECT_PATH, "vocab.txt"),
    "text_files": os.path.join(PROJECT_PATH, "text_input.txt"),
    "text_embedding": os.path.join(PROJECT_PATH, "text_embedding.txt"),
    "log_path": os.path.join(PROJECT_PATH, "log.txt"),
    "saved_weight": os.path.join(PROJECT_PATH, "saved_weight.h5"),
    "character_freq_threshold": 1,
    "segment_size": 1,
    "batch_size": 32,
    "max_palette_length": 5,
    "max_sequence_length": 6,
    "max_text_length": 1,
    "mask_rate": 0.8,
    "mask_token_rate": 0.5,
    "mask_position": [],
    "vocab_size": 817,
    "embedding_dim": 256,
    "transformer_layers": 2,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "initializer_range": 0.02,
    "bias_regularizer": 1e-4,
    "learning_rate": 1e-4,
}