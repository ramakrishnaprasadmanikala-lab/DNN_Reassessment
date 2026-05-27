"""
text_classifier.py
------------------
LSTM-based text classifier for story sequence position prediction.
Reassessment Task: 5-class classification (positions 1–5 in story sequence).

Architecture:
    tokens [B, T] ──► Embedding ──► LSTM ──► final hidden ──► Linear(5) ──► logits [B, 5]

Loss: CrossEntropyLoss
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMTextClassifier(nn.Module):
    """
    LSTM-based sequence classifier.
    Predicts which narrative position (1–5) a given text sentence belongs to.

    Configurable via constructor arguments so that ablation experiments
    can each change exactly ONE hyperparameter from the baseline.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 5,
        bidirectional: bool = False,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # ── Embedding ──────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[pad_idx].fill_(0)   # keep PAD embedding zero

        # ── LSTM ───────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ── Dropout ────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Classifier Head ────────────────────────────────────────────────────
        # Final hidden state dimension depends on bidirectionality
        lstm_out_dim = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens  : Tensor [B, T]   — token indices
            lengths : Tensor [B]      — actual (non-padded) sequence lengths
        Returns:
            logits  : Tensor [B, 5]   — raw class scores (NOT softmaxed)
        """
        # Embedding with dropout
        embedded = self.dropout(self.embedding(tokens))   # [B, T, embed_dim]

        # Pack padded sequences (avoids LSTM computing over padding)
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(embedded, lengths_cpu,
                                      batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)

        # Extract final hidden state
        # h_n: [num_layers * num_directions, B, hidden_dim]
        if self.bidirectional:
            # Concatenate forward and backward last-layer hidden states
            h_fwd = h_n[-2]   # [B, hidden_dim]  — forward  last layer
            h_bwd = h_n[-1]   # [B, hidden_dim]  — backward last layer
            final_hidden = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 2*hidden_dim]
        else:
            final_hidden = h_n[-1]                 # [B, hidden_dim]

        # Classify
        logits = self.classifier(self.dropout(final_hidden))   # [B, 5]
        return logits


def build_classifier(cfg: dict, vocab_size: int, **overrides) -> "LSTMTextClassifier":
    """
    Instantiate an LSTMTextClassifier from config, with optional overrides
    for ablation experiments (e.g. build_classifier(cfg, vs, hidden_dim=256)).
    """
    c = cfg["model"]["text_classifier"]
    params = dict(
        vocab_size=vocab_size,
        embed_dim=c["embed_dim"],
        hidden_dim=c["hidden_dim"],
        num_layers=c["num_layers"],
        dropout=c["dropout"],
        num_classes=c["num_classes"],
        bidirectional=c.get("bidirectional", False),
        pad_idx=0,
    )
    params.update(overrides)
    return LSTMTextClassifier(**params)
