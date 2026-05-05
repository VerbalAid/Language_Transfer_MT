#!/usr/bin/env python3
"""
Fine-tune **MarianMT** `Helsinki-NLP/opus-mt-es-ca` (Spanish→Catalan checkpoint) on
**Spanish→Asturian** parallel data (`projecte-aina/ES-AST_Parallel_Corpus`).


Default profile: subsampled train/val, 1 epoch, memory-efficient optim (Adafactor) +
gradient checkpointing so micro-batch size can stay high on ~8GB GPUs.
Use --full for the full split and 3 epochs (still slow).
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import evaluate

FAST_TRAIN_SAMPLES = 100_000
FAST_EVAL_SAMPLES = 4_000
# Defaults; VRAM auto-tune (after model is on GPU) may lower micro-batch and raise grad_accum.
FAST_TRAIN_BS = 8
FAST_GRAD_ACCUM = 2
FAST_EVAL_BS = 4
FAST_EPOCHS = 1.0
TARGET_EFFECTIVE_BATCH_FAST = 16
TARGET_EFFECTIVE_BATCH_FULL = 8


def max_microbatch_for_free_gib(free_gib: float) -> int:
    """Conservative Marian seq2seq + checkpointing; free_gib = cuda mem free after model load."""
    if free_gib >= 5.5:
        return 16
    if free_gib >= 4.0:
        return 8
    if free_gib >= 2.8:
        return 6
    if free_gib >= 2.0:
        return 4
    if free_gib >= 1.2:
        return 2
    return 1


def apply_vram_clamp(
    *,
    free_gib: float,
    train_bs: int,
    eval_bs: int,
    grad_accum: int,
    target_effective: int,
) -> tuple[int, int, int, str]:
    """Clamp micro-batches to fit free VRAM; raise grad_accum to stay near target_effective."""
    max_micro = max_microbatch_for_free_gib(free_gib)
    note_parts: list[str] = []
    if train_bs > max_micro:
        note_parts.append(
            f"train micro-batch {train_bs}→{max_micro} (~{free_gib:.2f} GiB free after model on GPU)"
        )
        train_bs = max_micro

    eval_bs = min(eval_bs, max(1, max_micro), 8)

    min_accum = max(1, (target_effective + train_bs - 1) // train_bs)
    new_accum = max(grad_accum, min_accum)
    if new_accum > grad_accum:
        note_parts.append(
            f"grad_accum {grad_accum}→{new_accum} (effective batch ≈ {train_bs * new_accum}, target {target_effective})"
        )
    grad_accum = min(new_accum, 64)

    note = (
        "Auto VRAM: " + "; ".join(note_parts)
        if note_parts
        else f"Auto VRAM: train_bs={train_bs}, grad_accum={grad_accum} (fits ~{free_gib:.2f} GiB free)."
    )
    return train_bs, eval_bs, grad_accum, note


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--full",
        action="store_true",
        help="Train on the full split (~634k train, ~70k val), 3 epochs, step-based eval (slow).",
    )
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training rows after split (default: 100k fast / no cap with --full).",
    )
    p.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Cap validation rows for BLEU during training (default: 4k fast / no cap with --full).",
    )
    p.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="Epochs (default: 1 fast / 3 with --full).",
    )
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--per_device_eval_batch_size", type=int, default=None)
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation (default: 2 fast / 1 full unless auto-tune changes it).",
    )
    p.add_argument(
        "--optim",
        type=str,
        default="adafactor",
        choices=("adafactor", "adamw_torch"),
        help="adafactor uses far less optimizer VRAM than AdamW (recommended on 8GB GPUs).",
    )
    p.add_argument("--output_dir", type=str, default="./mt_checkpoints")
    p.add_argument("--save_dir", type=str, default="./finetuned_es_ast")
    p.add_argument(
        "--skip_tokenize",
        action="store_true",
        help="Load ./tokenized_es_ast and ./raw_es_ast from disk instead of HF + map.",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint-* folder to resume training.",
    )
    p.add_argument(
        "--no_auto_batch",
        action="store_true",
        help="Do not clamp batch sizes to free VRAM (you will OOM if settings are too large).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_checkpoint = "Helsinki-NLP/opus-mt-es-ca"
    max_input_length = 128
    max_target_length = 128

    tok_dir = Path("./my_tokenizer")
    raw_disk = Path("./raw_es_ast")
    tok_disk = Path("./tokenized_es_ast")

    if args.skip_tokenize and tok_disk.is_dir() and raw_disk.is_dir():
        print("Loading cached tokenized data from disk…")
        tokenized_dataset = load_from_disk(str(tok_disk))
        dataset = load_from_disk(str(raw_disk))
        tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir) if tok_dir.is_dir() else model_checkpoint
        )
    else:
        print("Loading ES–AST parallel corpus from Hugging Face…")
        dataset = load_dataset(
            "projecte-aina/ES-AST_Parallel_Corpus", split="train"
        )
        dataset = Dataset.from_list(
            [
                {"translation": {"spa": row["es"], "ast": row["ast"]}}
                for row in dataset
            ]
        )

        print("Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_dir))

        def preprocess_function(examples):
            inputs = [ex["spa"] for ex in examples["translation"]]
            targets = [ex["ast"] for ex in examples["translation"]]
            model_inputs = tokenizer(
                inputs, max_length=max_input_length, truncation=True
            )
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        nproc = min(8, (os.cpu_count() or 2))
        print(f"Tokenizing dataset (num_proc={nproc})…")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=nproc,
        )
        tokenized_dataset.save_to_disk(str(tok_disk))
        dataset.save_to_disk(str(raw_disk))
        print(f"Saved raw → {raw_disk}, tokenized → {tok_disk}")

    print("Loading model…")
    # FP32 weights: mixed precision (fp16/bf16) is applied by the Trainer. Loading the
    # model in full float16 makes gradients FP16 and breaks GradScaler (clip_grad_norm).
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print(
        "\n  Base checkpoint: MarianMT opus-mt-es-ca (spa→ca), adapted on spa→ast data.\n"
        "  (Not NLLB / not Basque — different architecture and language pair.)\n"
    )

    free_g = 0.0
    total_g = 0.0
    if torch.cuda.is_available():
        model = model.to("cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_g, total_g = free_b / 2**30, total_b / 2**30
            _amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            print(
                f"  Model on GPU (float32 params, Trainer AMP: {_amp}). "
                f"Mem free ≈ {free_g:.2f} / {total_g:.2f} GiB.\n"
                "  Other jobs using this GPU will force a smaller train batch (auto unless "
                "--no_auto_batch).\n"
            )
        except AttributeError:
            pass

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]
        return preds, labels

    def compute_metrics(eval_preds, tokenizer=tokenizer):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < tokenizer.vocab_size, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        return {"bleu": round(result["score"], 2)}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    tokenized_split = DatasetDict(
        {"train": split["train"], "validation": split["test"]}
    )

    if args.full:
        train_limit = args.max_train_samples
        eval_limit = args.max_eval_samples
        epochs = args.num_train_epochs if args.num_train_epochs is not None else 3.0
        train_bs = args.per_device_train_batch_size or 8
        eval_bs = args.per_device_eval_batch_size or 8
        eval_strategy = "steps"
        eval_steps = 2000
        save_strategy = "steps"
        save_steps = 2000
        print("Profile: FULL (slow)")
    else:
        train_limit = (
            args.max_train_samples
            if args.max_train_samples is not None
            else FAST_TRAIN_SAMPLES
        )
        eval_limit = (
            args.max_eval_samples
            if args.max_eval_samples is not None
            else FAST_EVAL_SAMPLES
        )
        epochs = args.num_train_epochs if args.num_train_epochs is not None else FAST_EPOCHS
        train_bs = args.per_device_train_batch_size or FAST_TRAIN_BS
        eval_bs = args.per_device_eval_batch_size or FAST_EVAL_BS
        eval_strategy = "epoch"
        eval_steps = None
        save_strategy = "epoch"
        save_steps = None
        print("Profile: FAST (subsampled train/val, 1 epoch by default)")

    tr = tokenized_split["train"]
    if train_limit is not None and len(tr) > train_limit:
        tr = tr.shuffle(seed=42).select(range(train_limit))
    tokenized_split["train"] = tr

    vs = tokenized_split["validation"]
    if eval_limit is not None and len(vs) > eval_limit:
        vs = vs.shuffle(seed=43).select(range(eval_limit))
    tokenized_split["validation"] = vs

    if args.gradient_accumulation_steps is not None:
        grad_accum: int | None = args.gradient_accumulation_steps
    elif not args.full and args.per_device_train_batch_size is None:
        grad_accum = FAST_GRAD_ACCUM
    else:
        grad_accum = 1

    target_eff = TARGET_EFFECTIVE_BATCH_FULL if args.full else TARGET_EFFECTIVE_BATCH_FAST
    if torch.cuda.is_available() and not args.no_auto_batch:
        train_bs, eval_bs, grad_accum, vram_note = apply_vram_clamp(
            free_gib=free_g,
            train_bs=train_bs,
            eval_bs=eval_bs,
            grad_accum=grad_accum,
            target_effective=target_eff,
        )
    elif args.no_auto_batch and torch.cuda.is_available():
        vram_note = "VRAM auto-tune disabled (--no_auto_batch); OOM possible if batch is too large."
    else:
        vram_note = "CPU: no VRAM auto-tune."

    use_bf16 = torch.cuda.is_available() and getattr(
        torch.cuda, "is_bf16_supported", lambda: False
    )()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    use_adafactor = args.optim == "adafactor"
    weight_decay = 0.0 if use_adafactor else 0.01
    lr = 1e-4 if use_adafactor else 2e-5

    n_train = len(tokenized_split["train"])
    n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 1
    updates_per_epoch = max(1, n_train // (train_bs * grad_accum * max(1, n_dev)))
    warmup_steps = max(50, min(2000, int(0.06 * updates_per_epoch * epochs)))

    print(
        f"Train: {n_train:,} | "
        f"Val: {len(tokenized_split['validation']):,} | "
        f"epochs={epochs} | optim={args.optim} | lr={lr} | "
        f"train_bs={train_bs} × grad_accum={grad_accum} "
        f"(effective {train_bs * grad_accum}) | "
        f"checkpointing=on | warmup_steps={warmup_steps}"
    )
    print(f"  {vram_note}\n")

    ta_kw: dict = dict(
        output_dir=args.output_dir,
        optim=args.optim,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=50,
        report_to="none",
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        predict_with_generate=True,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=min(8, (os.cpu_count() or 2)),
    )
    if eval_steps is not None:
        ta_kw["eval_steps"] = eval_steps
    if save_steps is not None:
        ta_kw["save_steps"] = save_steps

    training_args = Seq2SeqTrainingArguments(**ta_kw)

    _trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_split["train"],
        eval_dataset=tokenized_split["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    _sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in _sig.parameters:
        _trainer_kw["processing_class"] = tokenizer
    else:
        _trainer_kw["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**_trainer_kw)

    print("Starting training…")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"Fine-tuned model saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
