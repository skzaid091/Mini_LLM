# Mini LLM — Decoder-Only Transformer Built from Scratch

A GPT-style language model built entirely from scratch in PyTorch.
No built-in transformer modules — every component implemented manually.

Pre-trained on TinyStories (~16M tokens) achieving **val perplexity of 2.64**,
then instruction fine-tuned to generate coherent stories from natural language prompts.

---

## 🏗️ Architecture

| Component | Details |
|---|---|
| Model Type | Decoder-only Transformer |
| Parameters | ~20M |
| Embedding Dim | 512 |
| Attention Heads | 4 |
| Decoder Blocks | 6 |
| Max Seq Length | 512 tokens |
| Vocabulary Size | ~4,260 (BPE) |
| FFN Hidden Dim | 2,048 |

---

## 📁 Project Structure

```
mini_llm/
│
├── data/                          # Datasets and tokenizer vocab
├── models/                        # Saved checkpoints
├── Notebooks/                     # Experiments and analysis
│
├── scripts/
│   ├── Tokenization.py            # Custom BPE tokenizer
│   ├── Embedding.py               # Token embedding layer
│   ├── Positional_Encoding.py     # Sinusoidal positional encoding
│   ├── Multi_Head_Causal_Attention.py  # Multi-head causal self-attention
│   ├── Feed_Forward.py            # Position-wise feed-forward network
│   ├── Layer_Normalization.py     # Custom layer normalization
│   ├── Decoder_Block.py           # Single transformer decoder block
│   ├── Linear.py                  # Output projection layer
│   ├── Early_Stopping.py          # Early stopping with warmup
│   └── Mini_LLM.py                # Main model class
│
├── config.py                      # Hyperparameters and settings
├── train.py                       # Pre-training loop
├── instruction_training.py        # Fine-tuning loop
├── inference.py                   # Text generation
└── utils.py                       # Data loading, save/load helpers
```

---

## 🚀 Training Results

### Phase 1 — WikiText-2 (Bug Fixing & Validation)
| Epoch | Train Loss | Val Loss | Perplexity |
|---|---|---|---|
| 1 | 3.55 | 2.96 | 19.3 |
| 10 | 2.39 | 2.25 | 9.5 |
| 21 | 2.07 | 2.10 | 8.2 |

### Phase 2 — TinyStories Pre-training
| Epoch | Train Loss | Val Loss | Perplexity |
|---|---|---|---|
| 1 | 2.06 | 1.58 | 4.87 |
| 10 | 1.00 | 1.01 | 2.75 |
| 26 | 0.80 | 0.97 | **2.64** ← early stop |

### Phase 3 — Instruction Fine-tuning
| Epoch | Train Loss |
|---|---|
| 1 | 1.04 |
| 10 | 0.57 |
| 20 | 0.34 |

---

## 💬 Inference Examples

```
Prompt: Instruction: Tell me a story about a dragon.

Output: The dragon was a mild dragon, and everything around the world.
One day, the dragon was feeling brave and decided to explore a pitch.
He grabbed the pitch and started to dig in. He dug and dug until he
found himself in a big surprise!...
```

```
Prompt: Instruction: Write a bedtime story.

Output: One day, a young boy named Tim found a big box. He was very
happy and wanted to know what was inside. He tried to open the box,
but it was stuck. He asked his friend, Sam, for help.
"Sam, can you help me open...
```

---

## ⚙️ Setup & Usage

### Install dependencies
```bash
pip install torch numpy
```

### Prepare data
Download the TinyStories dataset and place it in the `/data` directory:
```
data/
├── tiny_stories_train.txt
├── tiny_stories_val.txt
```
Then train the BPE tokenizer vocab by setting `train_vocab: True` in `config.py` on the first run.

### Train the model
```bash
python train.py
```
Checkpoints will be saved to `/models` (gitignored — not included in repo).

### Instruction fine-tuning
```bash
python instruction_training.py
```

### Run inference
```bash
python inference.py --prompt "Instruction: Tell me a story about a dragon."
```
> Note: Requires a trained checkpoint in `/models`. Train the model first or provide your own checkpoint.

---

## 🔧 Configuration

Edit `config.py` to change hyperparameters:

```python
config = {
    "embedding_dim": 512,
    "num_att_heads": 4,
    "num_decoder_blocks": 6,
    "max_seq_len": 512,
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 100,
    "dropout_rate": 0.1,
}
```

---

## 🐛 Key Training Challenges Solved

8 critical bugs were diagnosed and resolved during training:

1. **Device mismatch** — embedding weights pinned to CPU while data on GPU
2. **Data pipeline truncation** — `zip()` silently discarding 90% of training data
3. **Vanishing gradients** — zero gradient signal in blocks 1–5 of 6
4. **Exploding activations** — residual stream growing from mean=0.6 to mean=463
5. **PE overwhelming embeddings** — positional encoding 20x larger than token embeddings
6. **Wrong attention initialization** — causing near-uniform softmax distributions
7. **Attention score overflow** — raw scores reaching mean=362 instead of ~0
8. **Learning rate too high** — 0.01 causing loss divergence after step 250

Full debugging log available in `Mini_LLM_Training_Log.docx`

---

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (trained on RTX 4080 8GB VRAM)

---
