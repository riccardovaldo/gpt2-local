# gpt2-local

Lightweight, educational GPT-2 (minGPT-style) implementation with optional LoRA adapters for efficient fine-tuning.


## What the project does

- Implements a compact GPT-2 model (`model.py`) compatible with Hugging Face `gpt2` weights.
- Provides a simple data loader using tiny Shakespeare (`data.py`).
- Supports Low-Rank Adaptation (LoRA) injection to replace selected linear layers (`lora.py`).
- Training and evaluation helpers are included (`trainer.py`, `main.py`).
- Quick verification and GPU check scripts: `gputest.py`.

## Quickstart

Requirements
- Python 3.12+
- PyTorch (matching your CUDA if you want GPU)



This is optional — `uv sync` will read the `pyproject.toml` configuration and reinstall packages accordingly.


The project uses the dependencies listed in [pyproject.toml](pyproject.toml).

Run a GPU check

```powershell
python gputest.py
```

Train locally (small example)

```powershell
python main.py --batch_size 4 --block_size 128 --epochs 3
# Enable LoRA adapters for parameter-efficient tuning:
python main.py --use_lora --lora_rank 4 --lora_alpha 16
```

Notes
- `main.py` wires together dataloaders, model loading, optional LoRA injection and the training loop.
- The training loop is intentionally minimal; see [trainer.py](trainer.py) for the training and eval helpers.

## Key files
- [model.py](model.py) — GPT model implementation and `from_pretrained` loader.
- [lora.py](lora.py) — LoRA adapter implementation and injector.
- [data.py](data.py) — dataset loader (tiny Shakespeare) and `get_dataloaders`.
- [trainer.py](trainer.py) — training and evaluation utilities.
- [main.py](main.py) — CLI entrypoint for training and experiments.
- [gputest.py](gputest.py) — prints CUDA availability and memory usage.