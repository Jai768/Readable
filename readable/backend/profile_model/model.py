"""
model.py
========
DyslexiaProfiler — Multi-Output Neural Network

Architecture
------------
Input (11 features)
      ↓
Feature-Group Encoders
  ├── Eye Encoder      (5 eye features   → 32 dims)
  ├── Speech Encoder   (5 speech features → 32 dims)
  └── Acoustic Encoder (1 acoustic feature → 16 dims)
      ↓
Cross-Modal Attention Fusion  (80 → 64 dims)
      ↓
Shared Representation (64 → 32 dims)
      ↓
Five Output Heads (each 32 → 16 → 1, Sigmoid)
  ├── Reading Fluency        (0=poor, 1=good)
  ├── Decoding Difficulty    (0=easy, 1=hard)
  ├── Phonological Difficulty
  ├── Visual Difficulty
  └── Attention Difficulty
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE METADATA
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    # Eye-tracking (indices 0-4)
    "fixation_duration_ms",
    "saccade_length_deg",
    "regression_count",
    "skipped_word_rate",
    "reading_speed_wpm",
    # Speech (indices 5-9)
    "speech_rate_wps",
    "pause_duration_ms",
    "pause_frequency",
    "mispronunciation_rate",
    "repetition_rate",
    # Acoustic (index 10)
    "pitch_variation_hz",
]

OUTPUT_NAMES: List[str] = [
    "reading_fluency",
    "decoding_difficulty",
    "phonological_difficulty",
    "visual_difficulty",
    "attention_difficulty",
]

# Which input indices belong to each group
EYE_IDX     = list(range(0, 5))
SPEECH_IDX  = list(range(5, 10))
ACOUSTIC_IDX = [10]

# Min-max bounds for normalisation  (low, high)
FEATURE_BOUNDS = {
    "fixation_duration_ms": (80,   600),
    "saccade_length_deg":   (0.5,  15.0),
    "regression_count":     (0,    60),
    "skipped_word_rate":    (0.0,  0.6),
    "reading_speed_wpm":    (30,   500),
    "speech_rate_wps":      (0.3,  5.0),
    "pause_duration_ms":    (50,   1200),
    "pause_frequency":      (0,    40),
    "mispronunciation_rate":(0.0,  0.6),
    "repetition_rate":      (0.0,  0.5),
    "pitch_variation_hz":   (5,    100),
}


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Two-layer residual block with LayerNorm and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ModalityEncoder(nn.Module):
    """Encodes a feature group into a fixed-size embedding."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        hidden = max(in_dim * 4, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttention(nn.Module):
    """
    Soft attention that learns which modality matters most
    for each sample. Returns weighted sum of modality embeddings
    concatenated with the weighted combination.
    """

    def __init__(self, embed_dim: int, n_modalities: int = 3):
        super().__init__()
        self.n = n_modalities
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, 64),
            nn.Tanh(),
            nn.Linear(64, n_modalities),
        )

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        """
        embeds: list of [B, D] tensors (one per modality)
        Returns: [B, D*n] attention-weighted concatenation
        """
        cat = torch.cat(embeds, dim=-1)          # [B, D*n]
        weights = F.softmax(self.attn(cat), dim=-1)  # [B, n]

        # Weighted modality combination
        stacked = torch.stack(embeds, dim=1)     # [B, n, D]
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Return both individual embeddings and weighted fusion
        return torch.cat([cat, weighted], dim=-1)


class OutputHead(nn.Module):
    """Individual prediction head for one profile dimension."""

    def __init__(self, in_dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DyslexiaProfiler(nn.Module):
    """
    Multi-output dyslexia profile model.

    Parameters
    ----------
    eye_dim      : embedding size for eye-tracking modality
    speech_dim   : embedding size for speech modality
    acoustic_dim : embedding size for acoustic modality
    dropout      : dropout probability throughout
    """

    def __init__(
        self,
        eye_dim: int = 32,
        speech_dim: int = 32,
        acoustic_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.eye_encoder      = ModalityEncoder(len(EYE_IDX),      eye_dim,      dropout)
        self.speech_encoder   = ModalityEncoder(len(SPEECH_IDX),   speech_dim,   dropout)
        self.acoustic_encoder = ModalityEncoder(len(ACOUSTIC_IDX), acoustic_dim, dropout)

        # All modality embeddings padded to same size for attention
        self.max_dim = max(eye_dim, speech_dim, acoustic_dim)
        self.eye_proj      = nn.Linear(eye_dim,      self.max_dim)
        self.speech_proj   = nn.Linear(speech_dim,   self.max_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, self.max_dim)

        self.cross_attn = CrossModalAttention(embed_dim=self.max_dim, n_modalities=3)

        # After cross-modal attention: [cat + weighted] = max_dim*3 + max_dim = max_dim*4
        fusion_in = self.max_dim * 3 + self.max_dim  # = max_dim * 4
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(128, dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            ResidualBlock(64, dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        # Five output heads
        self.head_fluency         = OutputHead(32, dropout)
        self.head_decoding        = OutputHead(32, dropout)
        self.head_phonological    = OutputHead(32, dropout)
        self.head_visual          = OutputHead(32, dropout)
        self.head_attention       = OutputHead(32, dropout)

        # Weight initialisation
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x : [B, 11] normalised feature tensor

        Returns dict of tensors, each [B] (0-1 scores).
        """
        eye_in      = x[:, EYE_IDX]
        speech_in   = x[:, SPEECH_IDX]
        acoustic_in = x[:, ACOUSTIC_IDX]

        # Encode each modality
        eye_emb      = self.eye_proj(self.eye_encoder(eye_in))
        speech_emb   = self.speech_proj(self.speech_encoder(speech_in))
        acoustic_emb = self.acoustic_proj(self.acoustic_encoder(acoustic_in))

        # Cross-modal attention fusion
        fused = self.cross_attn([eye_emb, speech_emb, acoustic_emb])

        # Shared representation
        shared = self.fusion_net(fused)

        return {
            "reading_fluency":         self.head_fluency(shared),
            "decoding_difficulty":     self.head_decoding(shared),
            "phonological_difficulty": self.head_phonological(shared),
            "visual_difficulty":       self.head_visual(shared),
            "attention_difficulty":    self.head_attention(shared),
        }

    # ──────────────────────────────────────────────────────────────────────
    def predict_single(self, raw_features: dict,
                       scaler_params: Optional[dict] = None) -> dict:
        """
        Convenience: predict from a dict of raw (un-normalised) feature values.

        raw_features : dict with keys matching FEATURE_NAMES
        scaler_params: dict with 'mean_' and 'scale_' from a fitted StandardScaler
                       If None, uses min-max bounds from FEATURE_BOUNDS.
        Returns dict of profile scores.
        """
        self.eval()
        arr = np.array([raw_features[k] for k in FEATURE_NAMES], dtype=np.float32)

        if scaler_params:
            arr = (arr - np.array(scaler_params["mean_"])) / np.array(scaler_params["scale_"])
        else:
            # Fallback min-max normalisation
            for i, name in enumerate(FEATURE_NAMES):
                lo, hi = FEATURE_BOUNDS[name]
                arr[i] = np.clip((arr[i] - lo) / (hi - lo + 1e-8), 0, 1)

        x = torch.tensor(arr).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(x)

        return {k: float(v.item()) for k, v in out.items()}

    # ──────────────────────────────────────────────────────────────────────
    def save(self, path: str, scaler_params: Optional[dict] = None):
        """Save model weights + optional scaler parameters."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "scaler_params": scaler_params,
            "feature_names": FEATURE_NAMES,
            "output_names": OUTPUT_NAMES,
        }
        torch.save(payload, path)
        print(f"[✓] Model saved → {path}")

    @classmethod
    def load(cls, path: str, **model_kwargs) -> Tuple["DyslexiaProfiler", Optional[dict]]:
        """Load model from file. Returns (model, scaler_params)."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**model_kwargs)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("scaler_params")

    # ──────────────────────────────────────────────────────────────────────
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

class WeightedProfileLoss(nn.Module):
    """
    Weighted MSE loss across all 5 output heads.

    Weights reflect prediction difficulty:
    - Visual & Attention are harder (eye data noisier) → higher weight
    - Fluency & Decoding are more directly measurable → lower weight
    """

    DEFAULT_WEIGHTS = {
        "reading_fluency":         1.0,
        "decoding_difficulty":     1.0,
        "phonological_difficulty": 1.2,
        "visual_difficulty":       1.5,
        "attention_difficulty":    1.5,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = weights or self.DEFAULT_WEIGHTS

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns total weighted loss and per-output loss dict.
        """
        total = torch.tensor(0.0, device=next(iter(preds.values())).device)
        per_loss = {}

        for name in OUTPUT_NAMES:
            mse = F.mse_loss(preds[name], targets[name])
            w = self.weights[name]
            total = total + w * mse
            per_loss[name] = float(mse.item())

        return total, per_loss