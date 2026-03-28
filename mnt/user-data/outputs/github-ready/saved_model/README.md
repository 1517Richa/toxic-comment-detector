# Saved Model Directory

This directory is created automatically when you run `train.py`.

## Contents after training

| File | Description |
|---|---|
| `model_weights.pt` | Best model weights (PyTorch state dict) |
| `meta.json` | Config, tuned thresholds, and eval metrics |
| `vocab.txt` | BERT tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer configuration |
| `config.json` | BERT model configuration |

## Note on model files

`model_weights.pt` (~440 MB) is excluded from this repository via `.gitignore`.  
Run `python train.py --fast` to generate the model locally after cloning.
