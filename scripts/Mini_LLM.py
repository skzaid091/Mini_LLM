import math
import torch
import torch.nn as nn

from .Tokenization import Tokenizer
from .Embedding import Embedding
from .Positional_Encoding import PositionalEncoding

from .Linear import Linear
from .Layer_Normalization import LayerNorm

from .Decoder_Block import DecoderBlock


class Mini_LLM(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        num_merges = config["num_merges"]
        self.max_seq_len = config["max_seq_len"]
        self.embedding_dim = config["embedding_dim"]
        self.num_att_heads = config["num_att_heads"]

        self.tokenizer = Tokenizer(num_merges, self.max_seq_len)

        if config["train_vocab"]:
            self.tokenizer.upload_dataset(config["train_dataset_path"])
            self.tokenizer.train(config["vocab_save_path"])

        else:
            vocab_path = config["vocab_path"]
            self.tokenizer = self.tokenizer.load(vocab_path)
        
        self.vocab_size = self.tokenizer.vocab_size

        self.dropout_rate = config["dropout_rate"]
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.embedding = Embedding(self.vocab_size, self.embedding_dim)
        self.PE = PositionalEncoding(self.max_seq_len, self.embedding_dim)
        self.embedding_layer_norm = LayerNorm(self.embedding_dim)

        self.gelu_const = config["gelu_const"]
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embedding_dim=self.embedding_dim,
                max_seq_len=self.max_seq_len, 
                num_att_heads=self.num_att_heads,
                gelu_const=self.gelu_const,
                dropout_rate=self.dropout_rate,
                num_layers=config["num_decoder_blocks"]
            )
            for _ in range(config["num_decoder_blocks"])
        ])

        self.output_projection = Linear(
            self.embedding_dim,
            self.vocab_size
        )

        self.final_layer_norm = LayerNorm(
            self.embedding_dim
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.get_pad_token_id()
        )

        self.learning_rate = config["learning_rate"]
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["epochs"],
            eta_min=1e-6
        )

        self.to(self.device)


    def preprocess(self, data):
        encoded = self.tokenizer.encode_batch(
            data
        )
    
        input_ids = [item["input_ids"] for item in encoded]
    
        attention_masks = [item["attention_mask"] for item in encoded]
    
        tokens = torch.tensor(
            input_ids,
            dtype=torch.long
        ).to(self.device)
    
        att_mask = torch.tensor(
            attention_masks,
            dtype=torch.long
        ).to(self.device)
    
        # ============================================
        # Teacher Forcing Shift
        # ============================================

        input_tokens = tokens[:, :-1]    
        target_tokens = tokens[:, 1:]
        input_attention_mask = att_mask[:, :-1]

        return input_tokens, target_tokens, input_attention_mask
        

    def forward(self, input_ids, attention_mask):

        token_embeddings = self.embedding.forward(input_ids)
        token_embeddings = token_embeddings * math.sqrt(self.embedding_dim)

        x = self.PE.forward(token_embeddings)
        x = self.embedding_layer_norm(x)

        x = self.dropout(x)

        for block in self.decoder_blocks:
            x = block(x, attention_mask)

        x = self.final_layer_norm.forward(x)

        x = self.dropout(x)

        logits = self.output_projection(x)
        
        return logits


    def train_step(self, train_data):

        self.train()

        input_tokens, target_tokens, attention_mask = self.preprocess(train_data)
        
        logits = self.forward(input_tokens, attention_mask)

        B, T, V = logits.shape
        loss = self.loss_fn(
            logits.view(B * T, V),
            target_tokens.reshape(B * T)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


    def val_step(self, val_data):

        self.eval()

        input_tokens, target_tokens, attention_mask = self.preprocess(val_data)

        logits = self.forward(input_tokens, attention_mask)

        B, T, V = logits.shape
        loss = self.loss_fn(
            logits.view(B * T, V),
            target_tokens.reshape(B * T)
        )

        return loss.item()


    @torch.no_grad()
    def predict(self, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()

        encoded = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded["input_ids"]],
            dtype=torch.long,
            device=self.device
        )

        attention_mask = torch.tensor(
            [encoded["attention_mask"]],
            dtype=torch.long,
            device=self.device
        )

        pad_token_id = self.tokenizer.get_pad_token_id()

        # Remove trailing pads
        valid_length = attention_mask.sum().item()

        generated = input_ids[:, :valid_length]

        for _ in range(max_new_tokens):

            current_len = generated.size(1)

            if current_len > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]

            current_mask = torch.ones_like(generated)

            logits = self.forward(
                generated,
                current_mask
            )

            next_token_logits = logits[:, -1, :]

            # Temperature
            next_token_logits = next_token_logits / temperature

            # Top-K sampling
            if top_k is not None:
                values, _ = torch.topk(
                    next_token_logits,
                    top_k
                )

                min_value = values[:, -1].unsqueeze(-1)

                next_token_logits = torch.where(
                    next_token_logits < min_value,
                    torch.full_like(
                        next_token_logits,
                        float("-inf")
                    ),
                    next_token_logits
                )

            probs = torch.softmax(
                next_token_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

        generated_ids = generated[0].cpu().tolist()

        return self.tokenizer.decode(
            generated_ids
        )