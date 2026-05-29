import re
import pickle
from collections import defaultdict

class Tokenizer:

    def __init__(self, num_merges, max_seq_len):

        self.dataset = None

        self.num_merges = num_merges
        self.max_seq_len = max_seq_len

        # =====================================================
        # Core Storage
        # =====================================================

        self.vocab = {}
        self.bytes_to_ids = {}

        # =====================================================
        # Special Tokens
        # =====================================================

        self.special_tokens = {
            "[BOS]": None,
            "[EOS]": None,
            "[PAD]": None,
            "[UNK]": None
        }

        self.special_token_ids = {}

        self.is_trained = False

    # =========================================================
    # Dataset Upload
    # =========================================================

    def upload_dataset(self, path):
        with open(path, "r") as f:
            self.dataset = f.read()
        
    
    # =========================================================
    # Utilities
    # =========================================================

    @staticmethod
    def get_byte_representation(text):
        return list(text.encode("utf-8"))

    @staticmethod
    def decode_byte_representation(token):
        return bytes(token).decode(
            "utf-8",
            errors="replace"
        )

    @staticmethod
    def pretokenize(text):

        return re.findall(
            r"\s+|\w+|[^\w\s]",
            text
        )

    # =========================================================
    # Pair Frequency
    # =========================================================

    def get_pair_frequencies(self, tokens):

        pair_freq = defaultdict(int)

        for word in tokens:

            for i in range(len(word) - 1):

                pair = (
                    word[i],
                    word[i + 1]
                )

                pair_freq[pair] += 1

        return pair_freq

    # =========================================================
    # Merge Pair
    # =========================================================

    def merge_pair(self, tokens, pair, new_id):

        merged_tokens = []

        for word in tokens:

            i = 0
            merged_word = []

            while i < len(word):

                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):

                    merged_word.append(new_id)
                    i += 2

                else:
                    merged_word.append(word[i])
                    i += 1

            merged_tokens.append(merged_word)

        return merged_tokens

    # =========================================================
    # Training
    # =========================================================

    def train(self, save_path):

        if not self.dataset:
            raise ValueError("Dataset not uploaded. Using upload_dataset upload the dataset path.")

        # =====================================================
        # Pretokenize
        # =====================================================

        tokens = self.pretokenize(self.dataset)

        tokens = [
            self.get_byte_representation(token)
            for token in tokens
        ]

        # =====================================================
        # Initialize Base Vocabulary
        # =====================================================

        self.vocab = {}

        for idx in range(256):
            self.vocab[idx] = [idx]

        next_token_id = 256

        # =====================================================
        # Train BPE
        # =====================================================

        for i in range(self.num_merges):

            pair_freq = self.get_pair_frequencies(tokens)

            if not pair_freq:
                break

            best_pair = max(
                pair_freq,
                key=pair_freq.get
            )

            # Build merged token bytes
            self.vocab[next_token_id] = (
                self.vocab[best_pair[0]]
                + self.vocab[best_pair[1]]
            )

            # Apply merge
            tokens = self.merge_pair(
                tokens,
                best_pair,
                next_token_id
            )

            print(
                f"Merge {i+1}: "
                f"{best_pair} -> {next_token_id}"
            )

            next_token_id += 1

        # =====================================================
        # Build Reverse Lookup
        # =====================================================

        self.bytes_to_ids = {
            tuple(value): key
            for key, value in self.vocab.items()
        }

        # =====================================================
        # Add Special Tokens
        # =====================================================

        current_vocab_size = len(self.vocab)

        self.special_tokens = {
            "[BOS]": current_vocab_size,
            "[EOS]": current_vocab_size + 1,
            "[PAD]": current_vocab_size + 2,
            "[UNK]": current_vocab_size + 3
        }

        self.special_token_ids = {
            v: k
            for k, v in self.special_tokens.items()
        }

        self.vocab.update(self.special_token_ids)
        self.bytes_to_ids.update(self.special_tokens)

        self.is_trained = True

        print(
            f"\nTraining Complete "
            f"| Vocab Size: {len(self.vocab)}"
        )

        self.save(save_path)
        print(f"Model saved at path: {save_path}")

    # =========================================================
    # Encode Single Word
    # =========================================================

    def encode_word(self, word: str):

        word_bytes = self.get_byte_representation(word)

        tokens = []

        start = 0

        while start < len(word_bytes):

            end = len(word_bytes)

            found = None

            while end > start:

                piece = tuple(
                    word_bytes[start:end]
                )

                if piece in self.bytes_to_ids:

                    found = piece
                    break

                end -= 1

            # Unknown fallback
            if found is None:

                tokens.append(
                    self.special_tokens["[UNK]"]
                )

                break

            tokens.append(
                self.bytes_to_ids[found]
            )

            start = end

        return tokens

    # =========================================================
    # Encode Sequence
    # =========================================================

    def encode(self, sequence: str, max_seq_len: int = None):

        if not self.is_trained:
            raise ValueError("Tokenizer must be trained first.")

        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        tokens = self.pretokenize(sequence)

        encoded_sequence = [
            self.special_tokens["[BOS]"]
        ]

        for token in tokens:

            encoded = self.encode_word(token)

            encoded_sequence.extend(encoded)

        encoded_sequence.append(
            self.special_tokens["[EOS]"]
        )

        # =====================================================
        # Attention Mask
        # =====================================================

        attention_mask = [1] * len(encoded_sequence)

        # =====================================================
        # Padding
        # =====================================================

        if len(encoded_sequence) < max_seq_len:

            pad_count = (
                max_seq_len - len(encoded_sequence)
            )

            encoded_sequence.extend(
                [self.special_tokens["[PAD]"]]
                * pad_count
            )

            attention_mask.extend(
                [0] * pad_count
            )

        # =====================================================
        # Truncation
        # =====================================================

        elif len(encoded_sequence) > max_seq_len:

            encoded_sequence = encoded_sequence[
                :max_seq_len - 1
            ]

            encoded_sequence.append(
                self.special_tokens["[EOS]"]
            )

            attention_mask = attention_mask[
                :max_seq_len
            ]

        return {
            "input_ids": encoded_sequence,
            "attention_mask": attention_mask
        }

    # =========================================================
    # Batch Encode
    # =========================================================

    def encode_batch(self, batch):

        return [
            self.encode(sequence)
            for sequence in batch
        ]

    # =========================================================
    # Decode
    # =========================================================

    def decode(self, sequence):

        decoded = ""

        for token in sequence:

            # Skip special tokens
            if token in self.special_token_ids:
                continue

            decoded += self.decode_byte_representation(
                self.vocab[token]
            )

        return decoded

    # =========================================================
    # Batch Decode
    # =========================================================

    def decode_batch(self, batch):

        decoded_batch = []

        for encoded_sequence in batch:

            decoded_batch.append(
                self.decode(
                    encoded_sequence["input_ids"]
                )
            )

        return decoded_batch

    # =========================================================
    # Save
    # =========================================================

    def save(self, path: str):

        data = {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "max_seq_len": self.max_seq_len,
            "num_merges": self.num_merges
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"Tokenizer saved to: {path}")

    # =========================================================
    # Load
    # =========================================================

    @classmethod
    def load(cls, path: str):

        with open(path, "rb") as f:
            data = pickle.load(f)

        tokenizer = cls(
            num_merges=data["num_merges"],
            max_seq_len=data["max_seq_len"]
        )

        tokenizer.vocab = data["vocab"]
        tokenizer.special_tokens = data["special_tokens"]

        tokenizer.special_token_ids = {
            v: k
            for k, v in tokenizer.special_tokens.items()
        }

        tokenizer.bytes_to_ids = {
            tuple(value): key
            for key, value in tokenizer.vocab.items()
            if isinstance(value, list)
        }

        tokenizer.bytes_to_ids.update(
            tokenizer.special_tokens
        )

        tokenizer.is_trained = True

        print(f"Tokenizer loaded from: {path}")

        return tokenizer

    # =========================================================
    # Vocabulary Size
    # =========================================================

    @property
    def vocab_size(self):
        
        return len(self.vocab)


    # =========================================================
    # Get PAD token ID
    # =========================================================

    def get_pad_token_id(self):

        return self.special_tokens["[PAD]"]