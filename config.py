import math

config = {
    "num_merges": 4000,
    "max_seq_len": 256,
    "embedding_dim": 256,
    "num_att_heads": 4,
    "num_decoder_blocks": 6,
    
    "epochs": 20,
    "warm_up_epochs": 3,
    
    "batch_size": 16,
    "learning_rate": 3e-4,
    "dropout_rate": 0.1,
    
    "patience": 5,
    "train_val_diff": 0.1,
    "val_improvement_threshold": 0.001,
    "skip_train_val_overfit_check": True,
    
    "gelu_const": math.sqrt(2.0 / math.pi),

    "train_vocab": True,
    "vocab_save_path": "data/vocab/tiny_stories_vocab.pkl",
    
    "train_dataset_path": "data/tiny_stories/train.txt",
    "test_dataset_path": "data/tiny_stories/test.txt",

    "model_save_path": "models/tiny_stories_mini_llm_checkpoint.pt"
}