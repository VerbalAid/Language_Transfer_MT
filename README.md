# Spanish → Asturian Neural Machine Translation
### WMT24 Low-Resource MT Shared Task

Fine-tuning `facebook/nllb-200-distilled-600M` with LoRA for Spanish→Asturian translation, developed as part of the WMT24 constrained low-resource MT shared task. Achieves **22.28 SacreBLEU** on FLORES+ devtest — above the Apertium rule-based baseline (17.0) and comparable to the best constrained submissions at WMT24 (~22).

---

## Results

| System | Description | BLEU |
|--------|-------------|------|
| Zero-shot | `opus-mt-es-ca`, no fine-tuning | 3.24 |
| Apertium | Rule-based baseline (WMT24) | 17.0 |
| System 1 | MarianMT, bilingual, 100k AST | 16.52 |
| **System 2** | **NLLB + LoRA, full 704k AST** | **22.28** |
| System 3 | Continual ML (AST + GL + PT) | 14.72 |
| Best WMT24 constrained | — | ~22 |

Evaluated on [FLORES+ devtest](https://huggingface.co/datasets/openlanguagedata/flores_plus) (`spa_Latn → ast_Latn`, 1,012 sentences).

---

## Approach

### System 1 — MarianMT Bilingual Baseline
Fine-tunes `Helsinki-NLP/opus-mt-es-ca` on a 100k-sentence subsample of the ES-AST corpus. The Catalan checkpoint was chosen for its shared SentencePiece vocabulary and structural overlap with Asturian.

### System 2 — NLLB + LoRA *(best)*
Fine-tunes `facebook/nllb-200-distilled-600M` on the full ~704k ES-AST corpus using LoRA (`r=16`, targeting all attention projections). Only ~0.34% of parameters are trained, making a full-corpus run feasible on an 8GB GPU. NLLB's built-in `ast_Latn` language token handles generation routing without any vocabulary modification.

### System 3 — Continual Multilingual Fine-Tuning
Continues the System 2 adapter on 50k Galician (SciELO-GL) + 1,327 Portuguese (opus_books) + 10k Asturian mix-in. Triggered **catastrophic forgetting** (BLEU dropped to 14.72), attributed to a ~5:1 data imbalance between new and original language data, compounded by domain mismatch (biomedical/literary corpora vs. general-domain FLORES+).

---

## Repository Structure

```
├── finetune_es_ast.py            # System 1 — MarianMT bilingual fine-tuning
├── finetune_multilingual_ast.py  # System 2 — NLLB + LoRA full training
├── continue_multilingual.py      # System 3 — continual multilingual fine-tuning
├── export_flores_peft.py         # Generate FLORES+ hypotheses from a PEFT adapter
├── flores_devtest_hypotheses_bleu22.28.txt  # System 2 outputs (best)
├── flores_devtest_hypotheses_bleu14.72.txt  # System 3 outputs
└── train_run.log                 # System 2 training log
```

---

## Setup

```bash
pip install transformers datasets evaluate sacrebleu sentencepiece peft accelerate torch
```

Tested on Python 3.10+, PyTorch 2.x, CUDA 12.x.

---

## Training

### System 1 — MarianMT baseline

```bash
python finetune_es_ast.py
```

Defaults: 100k train samples, 1 epoch, Adafactor, bf16. Run with `--full` for the full corpus.

### System 2 — NLLB + LoRA (recommended)

```bash
python finetune_multilingual_ast.py
```

Defaults: full ~704k ES-AST corpus, 1 epoch, LoRA r=16, Adafactor, bf16, effective batch 16.
Runtime: ~9 hours on RTX 4060.

On subsequent runs, skip re-tokenisation:

```bash
python finetune_multilingual_ast.py --skip_data
```

### System 3 — Continual multilingual fine-tuning

Requires System 2 adapter saved to `./finetuned_multi_ast`.

```bash
python continue_multilingual.py
```

Defaults: 50k GL + 50k PT (capped by available data) + 10k AST mix-in, lr=2e-4.

---

## Generating FLORES+ Hypotheses

```bash
export HF_TOKEN=your_token_here  # FLORES+ is gated

python export_flores_peft.py \
  --adapter_dir ./finetuned_multi_ast \
  --out flores_devtest_hypotheses.txt
```

---

## Hardware

All training was run locally on a single **NVIDIA RTX 4060 (8GB VRAM)**. LoRA + gradient checkpointing + Adafactor + bf16 keep peak VRAM usage comfortably within 8GB for all runs.

---

## Data

| Dataset | Split | Used in |
|---------|-------|---------|
| [projecte-aina/ES-AST_Parallel_Corpus](https://huggingface.co/datasets/projecte-aina/ES-AST_Parallel_Corpus) | ~704k pairs | Systems 1, 2, 3 |
| [ZJaume/SciELO-GL](https://huggingface.co/datasets/ZJaume/SciELO-GL) | 50k (es-gl) | System 3 |
| [opus_books](https://huggingface.co/datasets/opus_books) | 1,327 (es-pt) | System 3 |
| [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus) | 1,012 devtest | Evaluation |

**Note:** Systems 1 and 2 use only the official WMT24 constrained track resource (ES-AST corpus). System 3 additionally uses SciELO-GL and opus_books, which fall outside the constrained data restriction and should be treated as an out-of-constraint experiment.

---

## Key Findings

- **LoRA is sufficient** for strong low-resource MT: updating ~0.34% of NLLB's parameters on the full ES-AST corpus outperforms both the rule-based Apertium baseline and a full fine-tuned bilingual MarianMT system.
- **Catastrophic forgetting is a real risk** in continual multilingual fine-tuning. A ~5:1 data imbalance between new language pairs and the original target language was enough to cause a 7.56 BLEU regression.
- **Domain mismatch compounds forgetting**: biomedical (SciELO-GL) and literary (opus_books) corpora are poor matches for general-domain evaluation benchmarks like FLORES+.

---

## Potential Improvements

- Balanced sampling (equal AST/GL/PT proportions) for multilingual training
- Adapter merging to combine language-pair adapters without joint training
- Larger LoRA rank (r=32) for greater multilingual capacity
- General-domain Galician corpus to replace SciELO-GL
- 2–3 training epochs on the full ES-AST corpus

---

## Citation

If you use this work, please cite the WMT24 shared task findings:

```bibtex
@inproceedings{wmt24lowresource,
  title     = {Findings of the WMT24 Shared Task on Low-Resource Machine Translation},
  booktitle = {Proceedings of the Ninth Conference on Machine Translation},
  year      = {2024},
  url       = {https://aclanthology.org/2024.wmt-1.57/}
}
```
