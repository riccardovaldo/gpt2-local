# gpt2-local

Lightweight, educational GPT-2 (minGPT-style) implementation with optional LoRA adapters for efficient fine-tuning.


## What the project does

- Implements a compact GPT-2 model (`model.py`) compatible with Hugging Face `gpt2` weights.
- Provides a simple data loader using tiny Shakespeare (`data.py`).
- Supports Low-Rank Adaptation (LoRA) injection to replace selected linear layers (`lora.py`).
- Training and evaluation are handled in `trainer.py`.
- Inference is performed with `inference.py`, using either the pretrained model or a fine-tuned checkpoint.
- Quick verification and GPU check scripts: `gputest.py`.

## Quickstart

Requirements
- Python 3.12+
- PyTorch (matching your CUDA if you want GPU)

The project uses the dependencies listed in [pyproject.toml](pyproject.toml).

Run a GPU check

```powershell
python gputest.py
```

Train locally (small example)

```powershell
python trainer.py --batch_size 4 --block_size 128 --epochs 3
# Enable LoRA adapters for parameter-efficient tuning:
python trainer.py --use_lora --lora_rank 4 --lora_alpha 16
```

Run inference

```powershell
python inference.py --temperature 0.8
# Or load a fine-tuned checkpoint:
python inference.py --checkpoint models/model_ft_20260522_175446.pt --temperature 0.8
```

Notes
- `trainer.py` now contains the full training CLI and loop.
- `inference.py` loads the pretrained model by default and can also apply a saved fine-tuned or LoRA checkpoint.

## Key files
- [model.py](model.py) — GPT model implementation and `from_pretrained` loader.
- [lora.py](lora.py) — LoRA adapter implementation and injector.
- [data.py](data.py) — dataset loader (tiny Shakespeare) and `get_dataloaders`.
- [trainer.py](trainer.py) — training and evaluation utilities with the CLI entrypoint.
- [inference.py](inference.py) — run model inference using pretrained or checkpoint weights.
- [gputest.py](gputest.py) — prints CUDA availability and memory usage.