#!/usr/bin/env python3
# ============================================================================
# LEGAL NOTICE & WATERMARK: R08 NON-COMMERCIAL RESEARCH & ETHICAL LICENSE v2.0
# Copyright (c) 2026 Thomas Seri (Sentinel). All Rights Reserved.
# Exclusive Jurisdiction: Court of Milan (Sezione Specializzata Impresa), Italy.
# Clean-room reverse engineering, distillation, commercial or military use
# is strictly prohibited and constitutes a breach of contract.
# ============================================================================
from __future__ import annotations
__author__ = "Thomas Seri (Sentinel)"
__license__ = "R08-Non-Commercial-Ethical-License-v2.0"
__jurisdiction__ = "Court of Milan, Italy"

"""Transformer Baseline 45M — Dense GPT-2 Architecture (14 Attention Layers)
Official Hugging Face / PyTorch implementation for comparative A/B benchmarking vs Sentinel R-08.
Author: Thomas Seri (Sentinel LifeOS)
"""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

import pezzi_tokenizer as pt
from vocab_stages import load_bundle, StageBundle


@dataclass
class TransformerConfig:
    vocab_size: int = 1251
    d_model: int = 512
    n_layers: int = 14
    n_heads: int = 8
    max_seq_len: int = 256
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={cfg.d_model} non divisibile per n_heads={cfg.n_heads}")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = float(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class BaselineTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        std_res = 0.02 / math.sqrt(2 * cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, mean=0.0, std=std_res)
            nn.init.normal_(blk.mlp.proj.weight, mean=0.0, std=std_res)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        pezzi_ids: torch.Tensor,
        parole_ids: Optional[torch.Tensor] = None,
        unioni_ids: Optional[torch.Tensor] = None,
        record_path: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        b, t = pezzi_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"seq_len={t} > max_seq_len={self.cfg.max_seq_len}")
        pos = torch.arange(t, device=pezzi_ids.device)
        h = self.drop(self.tok_emb(pezzi_ids) + self.pos_emb(pos))
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.ln_f(h)), None


class TransformerTokenizer:
    """Tokenizer a subword (1.251 pezzi) identico a Sentinel R-08."""

    def __init__(self, bundle: StageBundle):
        self.bundle = bundle
        self.pezzi = bundle.pezzi
        self.eos_id = int(getattr(bundle.pezzi, "eos_id", 2))
        self.pad_id = int(getattr(bundle.pezzi, "pad_id", 3))

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, Path]) -> "TransformerTokenizer":
        root = Path(pretrained_path)
        vocab_dir = root / "vocab_stages" if (root / "vocab_stages").exists() else root
        bundle = load_bundle(vocab_dir)
        return cls(bundle)

    def encode(self, text: str) -> List[int]:
        return pt.encode(text, self.pezzi)

    def decode(self, token_ids: List[int]) -> str:
        return pt.decode(token_ids, self.pezzi)


class TransformerModel(nn.Module):
    """Interfaccia Hugging Face friendly per il modello Transformer Baseline 45M."""

    def __init__(self, config: TransformerConfig, model: BaselineTransformer):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Union[str, Path],
        weights_file: str = "model.safetensors",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "TransformerModel":
        root = Path(pretrained_path)
        cfg_file = root / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"config.json non trovato in {root}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg = TransformerConfig.from_dict(cfg_dict)
        model = BaselineTransformer(cfg)

        weights_path = root / weights_file
        if not weights_path.exists():
            raise FileNotFoundError(f"{weights_file} non trovato in {root}")

        state_dict = load_file(str(weights_path))
        model.load_state_dict(state_dict)

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        model.to(torch.device(device))
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)

        model.eval()
        return cls(cfg, model)

    def forward(self, pezzi_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.model(pezzi_ids)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        tokenizer: TransformerTokenizer,
        max_new_tokens: int = 48,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        dev = next(self.model.parameters()).device
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [tokenizer.eos_id]

        new_tokens: List[int] = []
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-self.config.max_seq_len:]], dtype=torch.long, device=dev)
            logits, _ = self.model(x)
            next_logits = logits[0, -1]

            if repetition_penalty != 1.0 and new_tokens:
                for token_id in set(new_tokens):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                nxt = int(torch.multinomial(probs, num_samples=1).item())
            else:
                nxt = int(next_logits.argmax().item())

            if nxt in (tokenizer.eos_id, tokenizer.pad_id):
                break

            new_tokens.append(nxt)
            ids.append(nxt)

        return tokenizer.decode(new_tokens)
