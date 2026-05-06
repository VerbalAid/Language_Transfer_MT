#!/usr/bin/env python3
"""
Continue multilingual LoRA fine-tuning (spa→{ast,gl,pt}) from an existing adapter.

Goal: adapt the already-trained adapter with extra GL + PT while preventing AST forgetting.

Default recipe:
- 50k GL (OPUS-100 es-gl)
- 50k PT (OPUS-100 es-pt)
- 10k AST (projecte-aina ES-AST) mixed in for stability
- Lower LR than initial run (2e-4)

Run:
    python continue_multilingual.py
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import evaluate

SRC_LANG = "spa_Latn"
AST_LANG = "ast_Latn"
GL_LANG = "glg_Latn"
PT_LANG = "por_Latn"

BASE_MODEL = "facebook/nllb-200-distilled-600M"
MAX_LENGTH = 128

# 8GB-friendly micro-batch (same rationale as previous script)
TRAIN_BS = 4
GRAD_ACCUM = 4  # effective batch 16
EVAL_BS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continue LoRA adapter with extra GL/PT + AST mix-in.")
    p.add_argument("--adapter_dir", default="./finetuned_multi_ast", help="Existing LoRA adapter dir.")
    p.add_argument("--output_dir", default="./mt_checkpoints_multi_v2", help="Trainer checkpoint directory.")
    p.add_argument("--save_dir", default="./finetuned_multi_ast_v2", help="Final adapter save directory.")

    p.add_argument("--gl_samples", type=int, default=50_000)
    p.add_argument("--pt_samples", type=int, default=50_000)
    p.add_argument("--ast_mix_samples", type=int, default=10_000)
    p.add_argument("--eval_samples", type=int, default=2_000, help="AST-only eval size.")

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_ast(max_samples: int | None, seed: int) -> Dataset:
    ds = load_dataset("projecte-aina/ES-AST_Parallel_Corpus", split="train")
    rows = [
        {"source": r["es"], "target": r["ast"], "src_lang": SRC_LANG, "tgt_lang": AST_LANG}
        for r in ds
        if r.get("es") and r.get("ast")
    ]
    out = Dataset.from_list(rows)
    if max_samples and len(out) > max_samples:
        out = out.shuffle(seed=seed).select(range(max_samples))
    return out


def load_es_gl(max_samples: int | None, seed: int) -> Dataset:
    """
    Spanish→Galician parallel data.

    Note: OPUS-100 (datasets) doesn't provide es-gl config in current releases.
    We use SciELO-GL which includes an 'es-gl' config.
    """
    if max_samples == 0:
        return Dataset.from_list([])

    ds = load_dataset("ZJaume/SciELO-GL", "es-gl", split="train")
    rows = []
    for r in ds:
        src = r.get("frase_fonte")
        tgt = r.get("frase_galego")
        if src and tgt:
            rows.append({"source": src, "target": tgt, "src_lang": SRC_LANG, "tgt_lang": GL_LANG})
    out = Dataset.from_list(rows)
    if max_samples and len(out) > max_samples:
        out = out.shuffle(seed=seed).select(range(max_samples))
    return out


def load_es_pt(max_samples: int | None, seed: int) -> Dataset:
    """
    Spanish→Portuguese parallel data.

    `opus_books` includes an 'es-pt' config.
    """
    if max_samples == 0:
        return Dataset.from_list([])

    ds = load_dataset("opus_books", "es-pt", split="train")
    rows = []
    for r in ds:
        t = r.get("translation") or {}
        src = t.get("es")
        tgt = t.get("pt")
        if src and tgt:
            rows.append({"source": src, "target": tgt, "src_lang": SRC_LANG, "tgt_lang": PT_LANG})
    out = Dataset.from_list(rows)
    if max_samples and len(out) > max_samples:
        out = out.shuffle(seed=seed).select(range(max_samples))
    return out


def tokenize_group(examples: dict, tokenizer, max_length: int) -> dict:
    src_lang = examples["src_lang"][0]
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


def tokenize_multilingual(ds: Dataset, tokenizer, max_length: int, seed: int) -> Dataset:
    parts = []
    for lang_code in [AST_LANG, GL_LANG, PT_LANG]:
        group = ds.filter(lambda x, lc=lang_code: x["tgt_lang"] == lc, num_proc=1)
        if len(group) == 0:
            continue
        tok = group.map(
            lambda ex: tokenize_group(ex, tokenizer, max_length),
            batched=True,
            batch_size=512,
            remove_columns=group.column_names,
            num_proc=1,
        )
        parts.append(tok)
    return concatenate_datasets(parts).shuffle(seed=seed)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter dir not found: {adapter_dir.resolve()}")

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    ast_lang_id = tokenizer.convert_tokens_to_ids(AST_LANG)

    print("Loading GL/PT (+ AST mix-in) datasets…")
    gl_ds = load_es_gl(args.gl_samples, args.seed)
    pt_ds = load_es_pt(args.pt_samples, args.seed)
    ast_mix = load_ast(args.ast_mix_samples, args.seed)

    combined = concatenate_datasets([d for d in [gl_ds, pt_ds, ast_mix] if len(d) > 0]).shuffle(
        seed=args.seed
    )
    print(f"  GL : {len(gl_ds):,}")
    print(f"  PT : {len(pt_ds):,}")
    print(f"  AST: {len(ast_mix):,} (mix-in)")
    print(f"  Total: {len(combined):,}")

    # AST-only eval set (small, stable)
    ast_eval = load_ast(max_samples=None, seed=args.seed)
    ast_eval = ast_eval.train_test_split(test_size=0.01, seed=args.seed)["test"]
    if args.eval_samples and len(ast_eval) > args.eval_samples:
        ast_eval = ast_eval.shuffle(seed=args.seed + 1).select(range(args.eval_samples))

    print("Tokenizing…")
    tok_train = tokenize_multilingual(combined, tokenizer, MAX_LENGTH, args.seed)
    tok_eval = tokenize_multilingual(ast_eval, tokenizer, MAX_LENGTH, args.seed + 2)
    tokenized = DatasetDict({"train": tok_train, "validation": tok_eval})

    print("Loading base model + existing adapter…")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.generation_config = GenerationConfig(
        forced_bos_token_id=ast_lang_id,
        max_new_tokens=MAX_LENGTH,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"GPU: {free_b/2**30:.2f} / {total_b/2**30:.2f} GiB free after model load")

    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        valid = (preds >= 0) & (preds < tokenizer.vocab_size)
        preds = np.where(valid, preds, tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        refs = [[r.strip()] for r in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        decoded = [d.strip() for d in decoded]
        result = bleu_metric.compute(predictions=decoded, references=refs)
        return {"bleu": round(result["score"], 2)}

    n_train = len(tokenized["train"])
    use_fp16 = torch.cuda.is_available() and not use_bf16

    updates_per_epoch = max(1, n_train // (TRAIN_BS * GRAD_ACCUM))
    # Save more frequently to avoid losing progress if eval/metrics ever crashes again.
    save_steps = max(500, updates_per_epoch // 5)
    eval_steps = save_steps

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        optim="adafactor",
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.0,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    trainer_kw: dict = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    trainer_kw["processing_class" if "processing_class" in sig.parameters else "tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kw)

    print("Starting continued training…")
    print(f"  Train rows   : {n_train:,}")
    print(f"  Eval rows    : {len(tokenized['validation']):,} (AST-only)")
    print(f"  Epochs       : {args.epochs}")
    print(f"  LR           : {args.lr}")
    print(f"  Effective BS : {TRAIN_BS * GRAD_ACCUM}")
    print(f"  Save/Eval    : every {save_steps} steps")

    trainer.train()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"Updated LoRA adapter saved → {save_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

