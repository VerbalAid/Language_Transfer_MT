# Spanish → Asturian MT (Colab notebook)

Open and run `Spanish_Asturian_MT.ipynb` in Google Colab.

The notebook will:

- install deps
- fine-tune `Helsinki-NLP/opus-mt-es-ca` on ES→AST data
- evaluate on FLORES+ devtest
- write `flores_devtest_hypotheses_ast.txt` (one hypothesis per line) into your Drive folder `spanish_asturian_mt/`

This directory also contains the scripts used during development (`finetune_es_ast.py`, `eval_flores_es_ast.py`), but the notebook is the submission-ready entry point.
