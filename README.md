<<<<<<< HEAD
# Pipeline for training SimCLR Backbone and Classifier.

## Command‑Line Arguments
| Argument | Type | Description |
|---------|------|-------------|
| `--mode` | str | Operation to run (`pretrain`, `linear`) |
| `--config` | str | Path to model config file |
| `--checkpoint-dir` | str | Directory to save checkpoints |
| `--device` | str | Device to train on (`cpu` or `cuda`) |
| `--test-data` | Specify whether to test on test data after training. Applicable only for linear eval |
##

# Pretrain the backbone
```python3 main.py --mode pretrain --config config/simclr_config.yml --device cuda ```

# Train the linear classifier ontop of the saved backbone
```python3 main.py --mode linear --config config/simclr_config.yml --device cuda ```

# Train the linear classifier ontop of the saved backbone and verify against test data
```python3 main.py --mode linear --config config/simclr_config.yml --device cuda -t ```
=======
# SIMCLR
>>>>>>> e7652727d4132654592c033b08eb2125fbbdc71a
