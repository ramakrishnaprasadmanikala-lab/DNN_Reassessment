"""
clf_data_loader.py
------------------
Data loader for the REASSESSMENT classification task.

Task: Given a single text sentence, predict its position (1–5) in the story.

Labels:
    0 → position 1 (first)
    1 → position 2 (second)
    ...
    4 → position 5 (fifth)

(CrossEntropyLoss expects 0-indexed labels; we display as 1–5.)

Dataset structure:
    - train_dataset = load_dataset("daniel3303/StoryReasoning", split="train")
    - Each story has ≥5 ordered frames, each with a narrative text sentence.
    - We extract frames 0–4 (positions 1–5) from every story.

Split: 80% training / 20% validation (from the train split, since we need
       labelled position data from both halves).
"""

import os
import re
import pickle
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
NUM_POSITIONS = 5   # story positions 1–5


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.word2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2word = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.word_freq: Counter = Counter()

    def build_from_texts(self, texts):
        for text in texts:
            for token in self._tokenise(text):
                self.word_freq[token] += 1
        for word, _ in self.word_freq.most_common(self.max_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"[Vocab] Built vocabulary: {len(self.word2idx)} tokens")

    def encode(self, text: str, max_len: int = None):
        ids = [self.word2idx.get(t, UNK_IDX) for t in self._tokenise(text)]
        return ids[:max_len] if max_len else ids

    def _tokenise(self, text: str):
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── Text Extraction ─────────────────────────────────────────────────────────
def extract_frame_texts(story_str: str):
    """Extract per-frame narrative sentences from the story string."""
    matches = re.findall(r"<gdi image\d+>(.*?)</gdi>", story_str,
                         re.DOTALL | re.IGNORECASE)
    out = []
    for m in matches:
        text = re.sub(r"<[^>]+>", " ", m)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return out


# ─── Dataset ─────────────────────────────────────────────────────────────────
class StoryPositionDataset(Dataset):
    """
    Classification dataset.
    Each sample is one (text_sentence, position_label) pair.

    For every story with ≥5 text frames, we create 5 samples:
        sentence at index 0 → label 0  (displays as position 1)
        sentence at index 1 → label 1  (displays as position 2)
        ...
        sentence at index 4 → label 4  (displays as position 5)
    """

    def __init__(self, hf_subset, vocab: Vocabulary,
                 max_text_len: int = 50, split: str = ""):
        self.vocab = vocab
        self.T = max_text_len
        self.split = split
        self.samples = self._build_samples(hf_subset)
        self._print_stats()

    def _build_samples(self, hf_subset):
        samples = []
        for item in hf_subset:
            texts = extract_frame_texts(item["story"])
            if len(texts) < NUM_POSITIONS:
                continue
            for pos in range(NUM_POSITIONS):       # positions 0–4
                samples.append((texts[pos], pos))  # (sentence, label)
        return samples

    def _print_stats(self):
        total = len(self.samples)
        counts = Counter(label for _, label in self.samples)
        print(f"[Dataset/{self.split}] {total} samples total")
        print(f"  Class distribution: " +
              ", ".join(f"pos{k+1}={counts[k]}" for k in range(NUM_POSITIONS)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.vocab.encode(text, max_len=self.T)
        length = len(enc)
        # Pad to max_text_len
        enc = enc + [PAD_IDX] * (self.T - len(enc))
        tokens = torch.tensor(enc, dtype=torch.long)
        return {
            "tokens": tokens,                                  # [T]
            "length": torch.tensor(length, dtype=torch.long), # scalar
            "label":  torch.tensor(label,  dtype=torch.long), # 0–4
        }


# ─── Collate ─────────────────────────────────────────────────────────────────
def clf_collate(batch):
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
        "label":  torch.stack([b["label"]  for b in batch]),
    }


# ─── Builder ─────────────────────────────────────────────────────────────────
def build_clf_loaders(cfg, hf_train, vocab_path: str = None):
    """
    Build train/val DataLoaders for the classification task.

    Args:
        cfg       : config dict
        hf_train  : HuggingFace train split (full)
        vocab_path: path to cache / load vocabulary

    Returns:
        train_loader, val_loader, vocab
    """
    c = cfg["training"]
    T = c["max_text_len"]
    batch_size = c["batch_size"]

    # ── Subset ────────────────────────────────────────────────────────────────
    n_stories = min(c.get("train_subset", 1000), len(hf_train))
    hf_sub = hf_train.select(range(n_stories))
    print(f"[DataLoader] Using {n_stories} stories from training split")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    v_path = vocab_path or os.path.join(cfg["paths"]["checkpoint_dir"],
                                        "clf_vocab.pkl")
    if os.path.exists(v_path):
        print("[Vocab] Loading cached vocabulary")
        vocab = Vocabulary.load(v_path)
    else:
        vocab = Vocabulary(max_size=cfg["model"]["text_classifier"]["vocab_size"])
        all_texts = []
        for item in hf_sub:
            all_texts.extend(extract_frame_texts(item["story"]))
        vocab.build_from_texts(all_texts)
        vocab.save(v_path)

    # ── Full dataset → 80/20 split ────────────────────────────────────────────
    full_ds = StoryPositionDataset(hf_sub, vocab, max_text_len=T, split="full")
    n_total = len(full_ds)
    n_train = int(0.80 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[Split] Train={n_train} | Val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=clf_collate,
                              num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=clf_collate,
                              num_workers=0)
    return train_loader, val_loader, vocab
