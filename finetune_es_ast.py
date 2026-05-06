#!/usr/bin/env python3
"""
Multilingual fine-tuning: Spanish → Asturian / Galician / Portuguese
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base model : facebook/nllb-200-distilled-600M  (600M params, ≤1B ✓)
Method     : LoRA (PEFT) — ~0.5% trainable params, fits 8GB VRAM
Data       : ES-AST full 704k  +  ES-GL (OPUS-100)  +  ES-PT (OPUS-100)
Eval       : FLORES+ devtest  spa_Latn → ast_Latn  (SacreBLEU)
Hardware   : NVIDIA RTX 4060 (bf16, gradient checkpointing)

Install:
    pip install transformers datasets evaluate sacrebleu sentencepiece \
                peft accelerate torch

Usage:
    python finetune_es_ast.py                  # default run
    python finetune_es_ast.py --skip_data      # reuse cached tok data
    python finetune_es_ast.py --gl_samples 0   # AST+PT only
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import evaluate

# ── NLLB language codes ───────────────────────────────────────────
SRC_LANG = "spa_Latn"   # Spanish (source for all pairs)
AST_LANG = "ast_Latn"   # Asturian  ← primary target
GL_LANG  = "glg_Latn"   # Galician  ← cross-lingual transfer
PT_LANG  = "por_Latn"   # Portuguese ← broader Romance grounding

# ── model ─────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"
MAX_LENGTH       = 128

# ── data caps ─────────────────────────────────────────────────────
# AST: use full corpus (704k) — no cap by default
# GL / PT: capped to keep training time reasonable on RTX 4060
DEFAULT_GL_SAMPLES = 200_000
DEFAULT_PT_SAMPLES = 200_000
DEFAULT_EVAL_SAMPLES = 4_000

# ── training defaults ─────────────────────────────────────────────
# LoRA cuts VRAM dramatically: we can use a larger effective batch
# without gradient accumulation, making steps faster.
# Micro-batch tuned for ~8GB VRAM (NLLB-600M + LoRA + checkpointing + generate-eval spikes VRAM).
TRAIN_BS    = 4    # micro-batch per GPU
GRAD_ACCUM  = 4    # effective batch = TRAIN_BS × GRAD_ACCUM = 16
EVAL_BS     = 4
EPOCHS      = 1
LR          = 5e-4       # LoRA converges well with higher LR
WARMUP_RATIO = 0.05
LORA_R      = 16         # LoRA rank — 16 is a good balance
LORA_ALPHA  = 32         # = 2 × r (standard)
LORA_DROPOUT = 0.05

# Attention projection names in NLLB / M2M architecture
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]


# ─────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multilingual spa→{ast,gl,pt} fine-tuning with LoRA on NLLB-200-600M"
    )
    p.add_argument("--output_dir",   default="./mt_checkpoints_multi",
                   help="Trainer checkpoint directory.")
    p.add_argument("--save_dir",     default="./finetuned_multi_ast",
                   help="Final model save directory.")
    p.add_argument("--cache_dir",    default="./cached_multi",
                   help="Tokenized dataset cache directory.")
    p.add_argument("--gl_samples",   type=int, default=DEFAULT_GL_SAMPLES,
                   help="Max Galician training pairs (0 = skip).")
    p.add_argument("--pt_samples",   type=int, default=DEFAULT_PT_SAMPLES,
                   help="Max Portuguese training pairs (0 = skip).")
    p.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                   help="Max validation pairs (AST only).")
    p.add_argument("--epochs",       type=float, default=EPOCHS)
    p.add_argument("--lora_r",       type=int, default=LORA_R,
                   help="LoRA rank. Higher = more capacity, more VRAM.")
    p.add_argument("--skip_data",    action="store_true",
                   help="Load tokenized data from --cache_dir instead of re-processing.")
    p.add_argument("--merge_weights", action="store_true",
                   help="Merge LoRA weights into base model before saving.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────
def load_ast() -> Dataset:
    """Full projecte-aina ES-AST corpus (~704k pairs)."""
    print("Loading ES-AST corpus (full)…")
    ds = load_dataset("projecte-aina/ES-AST_Parallel_Corpus", split="train")
    rows = [
        {"source": r["es"], "target": r["ast"],
         "src_lang": SRC_LANG, "tgt_lang": AST_LANG}
        for r in ds
    ]
    print(f"  ES-AST: {len(rows):,} pairs")
    return Dataset.from_list(rows)


def load_opus(tgt_iso: str, tgt_nllb: str, max_samples: int | None) -> Dataset:
    """
    Load a Spanish-X language pair from Helsinki-NLP/opus-100.
    opus-100 pairs are stored alphabetically, so es-gl and es-pt are valid keys.
    """
    if max_samples == 0:
        return Dataset.from_list([])

    pair = f"es-{tgt_iso}" if "es" < tgt_iso else f"{tgt_iso}-es"
    print(f"Loading OPUS-100 {pair}…")

    try:
        ds = load_dataset("Helsinki-NLP/opus-100", pair, split="train",
                          trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: could not load {pair}: {e}")
        return Dataset.from_list([])

    rows = []
    for r in ds:
        t = r["translation"]
        src = t.get("es")
        tgt = t.get(tgt_iso)
        if src and tgt:
            rows.append({"source": src, "target": tgt,
                         "src_lang": SRC_LANG, "tgt_lang": tgt_nllb})

    result = Dataset.from_list(rows)
    if max_samples and len(result) > max_samples:
        result = result.shuffle(seed=42).select(range(max_samples))

    print(f"  {pair}: {len(result):,} pairs")
    return result


# ─────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────
def tokenize_group(examples: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenize a batch where all examples share the same src/tgt language.
    NLLB automatically prepends the target language token to labels when
    text_target is used — this becomes the forced_bos_token during generation.
    """
    src_lang = examples["src_lang"][0]
    tgt_lang = examples["tgt_lang"][0]

    tokenizer.src_lang = src_lang
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=examples["target"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_multilingual(ds: Dataset, tokenizer, max_length: int,
                           desc: str = "") -> Dataset:
    """
    Tokenize a multilingual dataset by processing each target language
    group separately (required so tokenizer.src_lang is consistent per batch).
    """
    parts = []
    for lang_code in [AST_LANG, GL_LANG, PT_LANG]:
        group = ds.filter(
            lambda x, lc=lang_code: x["tgt_lang"] == lc,
            num_proc=1,
        )
        if len(group) == 0:
            continue
        tok_group = group.map(
            lambda ex: tokenize_group(ex, tokenizer, max_length),
            batched=True,
            batch_size=512,
            remove_columns=group.column_names,
            num_proc=1,   # single proc: tokenizer is not fork-safe
            desc=f"Tokenising {desc} {lang_code}",
        )
        parts.append(tok_group)

    combined = concatenate_datasets(parts)
    return combined.shuffle(seed=42)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cache_dir = Path(args.cache_dir)
    tok_dir   = Path("./tokenizer_nllb")

    # ── Tokenizer ─────────────────────────────────────────────────
    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tok_dir))

    ast_lang_id = tokenizer.convert_tokens_to_ids(AST_LANG)
    print(f"  AST language token id: {ast_lang_id}")

    # ── Data ──────────────────────────────────────────────────────
    if args.skip_data and cache_dir.is_dir():
        print(f"Loading cached tokenized data from {cache_dir}…")
        tokenized = load_from_disk(str(cache_dir))
    else:
        # Load raw data
        ast_ds = load_ast()
        gl_ds  = load_opus("gl", GL_LANG, args.gl_samples)
        pt_ds  = load_opus("pt", PT_LANG, args.pt_samples)

        # Filter out empty datasets
        parts = [d for d in [ast_ds, gl_ds, pt_ds] if len(d) > 0]
        combined = concatenate_datasets(parts).shuffle(seed=42)

        print(f"\nTotal combined: {len(combined):,} pairs")
        print(f"  Language breakdown:")
        for lang in [AST_LANG, GL_LANG, PT_LANG]:
            n = sum(1 for x in combined if x["tgt_lang"] == lang)
            print(f"    {lang}: {n:,}")

        # Train / validation split
        # Val is AST-only so BLEU during training is directly comparable to FLORES+
        ast_only = combined.filter(lambda x: x["tgt_lang"] == AST_LANG, num_proc=1)
        non_ast  = combined.filter(lambda x: x["tgt_lang"] != AST_LANG, num_proc=1)

        ast_split = ast_only.train_test_split(test_size=0.01, seed=42)
        val_ds    = ast_split["test"]
        if args.eval_samples and len(val_ds) > args.eval_samples:
            val_ds = val_ds.shuffle(43).select(range(args.eval_samples))

        train_ds = concatenate_datasets([
            ast_split["train"], non_ast
        ]).shuffle(seed=42)

        print(f"\nTrain: {len(train_ds):,}  |  Val (AST only): {len(val_ds):,}")

        # Tokenize
        print("\nTokenizing…")
        tok_train = tokenize_multilingual(train_ds, tokenizer, MAX_LENGTH, "train")
        tok_val   = tokenize_multilingual(val_ds,   tokenizer, MAX_LENGTH, "val")

        tokenized = DatasetDict({"train": tok_train, "validation": tok_val})
        tokenized.save_to_disk(str(cache_dir))
        print(f"Saved tokenized data → {cache_dir}")

    print(f"\nFinal: train={len(tokenized['train']):,}  "
          f"val={len(tokenized['validation']):,}")

    # ── Model + LoRA ──────────────────────────────────────────────
    print("\nLoading NLLB-200-distilled-600M…")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_CHECKPOINT,
        torch_dtype=dtype,
    )

    # LoRA: only train a tiny fraction of the model
    # r=16 gives ~3.5M trainable out of 600M (0.58%)
    lora_config = LoraConfig(
        task_type       = TaskType.SEQ_2_SEQ_LM,
        r               = args.lora_r,
        lora_alpha      = args.lora_r * 2,
        target_modules  = LORA_TARGET_MODULES,
        lora_dropout    = LORA_DROPOUT,
        bias            = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Fix generation target to Asturian for eval
    model.generation_config = GenerationConfig(
        forced_bos_token_id = ast_lang_id,
        max_new_tokens      = MAX_LENGTH,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"\nGPU: {free_b/2**30:.2f} / {total_b/2**30:.2f} GiB free after model load")

    # ── Metric ────────────────────────────────────────────────────
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Guard against invalid ids (negatives or > vocab) which can crash fast tokenizers
        # with `OverflowError: out of range integral type conversion attempted`.
        preds = np.asarray(preds)
        valid = (preds >= 0) & (preds < tokenizer.vocab_size)
        preds = np.where(valid, preds, tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels  = np.where(labels != -100, labels, tokenizer.pad_token_id)
        refs    = [[r.strip()] for r in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        decoded = [d.strip() for d in decoded]
        result  = bleu_metric.compute(predictions=decoded, references=refs)
        return {"bleu": round(result["score"], 2)}

    # ── Training arguments ────────────────────────────────────────
    n_train  = len(tokenized["train"])
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # Eval every 10% of an epoch (at minimum every 1000 steps)
    updates_per_epoch = max(1, n_train // (TRAIN_BS * GRAD_ACCUM))
    eval_steps = max(1000, updates_per_epoch // 10)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = TRAIN_BS,
        per_device_eval_batch_size  = EVAL_BS,
        gradient_accumulation_steps = GRAD_ACCUM,
        gradient_checkpointing      = True,
        optim                       = "adafactor",
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = 0.0,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        logging_steps               = 200,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = eval_steps,
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_bleu",
        greater_is_better           = True,
        predict_with_generate       = True,
        generation_max_length       = MAX_LENGTH,
        # Python 3.14+ uses forkserver workers by default; collator holds `model` → unpicklable.
        dataloader_num_workers      = 0,
        dataloader_pin_memory       = torch.cuda.is_available(),
        report_to                   = "none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8,
    )

    # Handle tokenizer / processing_class API difference across versions
    trainer_kw: dict = dict(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized["train"],
        eval_dataset    = tokenized["validation"],
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
    )
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    trainer_kw[
        "processing_class" if "processing_class" in sig.parameters else "tokenizer"
    ] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kw)

    # ── Train ─────────────────────────────────────────────────────
    print(f"\nStarting training…")
    print(f"  Total pairs  : {n_train:,}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Effective BS : {TRAIN_BS * GRAD_ACCUM}")
    print(f"  Eval every   : {eval_steps} steps")
    print(f"  LoRA rank    : {args.lora_r}  (trainable params ≈ 0.5% of 600M)\n")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_weights:
        # Merge LoRA into base weights — larger file, no PEFT dependency at inference
        print("Merging LoRA weights into base model…")
        merged = model.merge_and_unload()
        merged.save_pretrained(str(save_dir))
        print(f"Merged model saved → {save_dir}")
    else:
        # Save LoRA adapter only — small file, requires PEFT at inference
        trainer.save_model(str(save_dir))
        print(f"LoRA adapter saved → {save_dir}")

    tokenizer.save_pretrained(str(save_dir))
    print(f"Tokenizer saved   → {save_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()

