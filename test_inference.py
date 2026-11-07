#!/usr/bin/env python3
"""Test di inferenza locale per Transformer Baseline 45M (Vocab 1.2k pezzi)."""
from pathlib import Path
from modeling_transformer import TransformerModel, TransformerTokenizer

ROOT = Path(__file__).resolve().parent

def main():
    print("1. Caricamento Tokenizer (1.2k pezzi)...")
    tokenizer = TransformerTokenizer.from_pretrained(ROOT)

    print("2. Caricamento Transformer Baseline (Midtrain 10k)...")
    model = TransformerModel.from_pretrained(ROOT, weights_file="model.safetensors")

    prompt = "La capitale d'Italia è "
    print(f"\n--- Test Prompt: '{prompt}' ---")
    gen = model.generate(prompt=prompt, tokenizer=tokenizer, max_new_tokens=30, temperature=0.0)
    print(f"Risposta Midtrain (greedy): {prompt}{gen}")

    print("\n3. Caricamento Transformer Baseline (Pretrain 185k)...")
    model_pre = TransformerModel.from_pretrained(ROOT, weights_file="model_pretrain_185k.safetensors")
    gen_pre = model_pre.generate(prompt=prompt, tokenizer=tokenizer, max_new_tokens=30, temperature=0.0)
    print(f"Risposta Pretrain 185k (greedy): {prompt}{gen_pre}")

if __name__ == "__main__":
    main()
