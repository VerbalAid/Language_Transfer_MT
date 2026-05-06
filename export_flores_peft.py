#!/usr/bin/env python3
"""
Export FLORES+ devtest translations for an NLLB + PEFT (LoRA) adapter.

Outputs one hypothesis per line (UTF-8), suitable for submission/archiving.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

SRC_LANG = "spa_Latn"
TGT_LANG = "ast_Latn"
BASE_MODEL = "facebook/nllb-200-distilled-600M"


def get_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", required=True, help="PEFT adapter directory (e.g., finetuned_multi_ast).")
    p.add_argument("--out", required=True, help="Output hypotheses file.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_length", type=int, default=128)
    args = p.parse_args()

    hf_token = get_hf_token()
    if not hf_token:
        raise SystemExit(
            "Need HF auth for gated FLORES+: export HF_TOKEN=… or run `huggingface-cli login` "
            "(accept the dataset license on the Hub first)."
        )

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter dir not found: {adapter_dir}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    print("Loading FLORES+ devtest…")
    flores_spa = load_dataset("openlanguagedata/flores_plus", SRC_LANG, split="devtest", token=hf_token)
    src_sentences = flores_spa["text"]
    print(f"  Examples: {len(src_sentences)}")

    print("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.src_lang = SRC_LANG
    forced_bos = tok.convert_tokens_to_ids(TGT_LANG)

    print("Loading base model + adapter…")
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.config.use_cache = False
    model.generation_config = GenerationConfig(
        forced_bos_token_id=forced_bos,
        max_new_tokens=args.max_length,
        num_beams=args.num_beams,
    )
    model = model.to(device)

    hyps: list[str] = []
    with torch.inference_mode():
        for i in range(0, len(src_sentences), args.batch_size):
            batch = src_sentences[i : i + args.batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            out = model.generate(**enc)
            hyps.extend(tok.batch_decode(out, skip_special_tokens=True))
            if (i // args.batch_size) % 50 == 0:
                print(f"  {min(i + args.batch_size, len(src_sentences))}/{len(src_sentences)}")

    out_path.write_text("\n".join(hyps) + "\n", encoding="utf-8")
    print(f"Wrote {len(hyps)} lines → {out_path}")


if __name__ == "__main__":
    main()

