# DNN Reassessment -- Text-Only Story Sequence Position Classifier

> **Module:** Deep Neural Networks (Reassessment Task)
> **Modality:** Text ONLY -- no images used anywhere
> **Task:** 5-class classification -- predict which position (1-5) a story sentence occupies
> **Loss:** `CrossEntropyLoss`

---

## Table of Contents

1. [Task Overview](#1-task-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Architecture](#4-architecture)
5. [How to Run (Google Colab)](#5-how-to-run-google-colab)
6. [Notebook Cell Guide](#6-notebook-cell-guide)
7. [The 5 Experiments](#7-the-5-experiments)
8. [Outputs & Results Files](#8-outputs--results-files)
9. [Configuration Reference](#9-configuration-reference)
10. [Analysis Questions Summary](#10-analysis-questions-summary)
11. [Marking Criteria Coverage](#11-marking-criteria-coverage)
12. [Dependencies](#12-dependencies)

---

## 1. Task Overview

This project addresses the **DNN Reassessment Task**: given a single text sentence extracted from a story, predict which narrative position -- **1 (first) through 5 (fifth)** -- the sentence belongs to. This is a supervised **5-class classification** problem.

| Property | Detail |
|---|---|
| Input | A single text sentence (tokenised, padded to 60 tokens) |
| Output | A probability distribution over 5 classes (positions 1-5) |
| Labels | `0` = position 1, `1` = position 2, ..., `4` = position 5 *(0-indexed for CrossEntropyLoss; displayed as 1-5)* |
| Loss function | `torch.nn.CrossEntropyLoss` |
| Evaluation metric | `Accuracy = correct_predictions / total_predictions` |
| Modality | **Text only** -- images are never loaded or used |

---

## 2. Dataset

**Source:** [`daniel3303/StoryReasoning`](https://huggingface.co/datasets/daniel3303/StoryReasoning) via HuggingFace `datasets`.

Each story contains 5 or more ordered elements. Each element has an image, a text sentence, and metadata. **Only the text sentences are used** in this project.

### How samples are constructed

For every story with at least 5 text segments, exactly **5 labelled samples** are created -- one per position:

```
Story sentence at index 0 -> label 0 (position 1)
Story sentence at index 1 -> label 1 (position 2)
Story sentence at index 2 -> label 2 (position 3)
Story sentence at index 3 -> label 3 (position 4)
Story sentence at index 4 -> label 4 (position 5)
```

### Train / Validation Split

An **80% / 20% split** is applied to the pool of labelled samples (not stories). The split uses `torch.utils.data.random_split` with `seed=42` for reproducibility. The default configuration uses 1,200 stories, producing approximately **6,000 samples** total (~4,800 train / ~1,200 val).

### Class Distribution

Because every story contributes exactly one sample per position, the dataset is **perfectly balanced** -- each of the 5 classes has the same number of samples. Random-chance accuracy is therefore exactly **20.0%**.

---

## 3. Project Structure

```
dnnls_reassessment/
|
+-- experiment_notebook.ipynb <- MAIN FILE -- run this in Google Colab
+-- config.yaml <- All hyperparameters
+-- requirements.txt <- Python dependencies
+-- README.md <- This file
|
+-- src/
| +-- __init__.py
| +-- text_classifier.py <- LSTMTextClassifier model definition
| +-- clf_data_loader.py <- Classification dataset & data loaders
| +-- text_encoder.py <- GRU text encoder (retained from original)
| +-- utils.py <- set_seed, count_parameters, AverageMeter
|
+-- results/ <- Auto-created when notebook runs
+-- checkpoints/ <- Vocabulary cache (clf_vocab.pkl)
+-- logs/
+-- training_curves.png <- 2x3 grid of loss curves (all experiments)
+-- ablation_bar.png <- Bar chart of best val accuracy per experiment
+-- results_table.csv <- Full results table
```

---

## 4. Architecture

### Baseline Model -- `LSTMTextClassifier`

```
Input: tokens [B, T=60] + lengths [B]
|
v
Embedding(vocab=5000, dim=128, padding_idx=0)
|
Dropout(p=0.3)
|
v
LSTM(input=128, hidden=128, layers=1, batch_first=True)
+- uses pack_padded_sequence to skip padding tokens
|
final hidden state [B, 128]
|
Dropout(p=0.3)
|
v
Linear(128 -> 64) -> ReLU -> Dropout(0.3) -> Linear(64 -> 5)
|
v
logits [B, 5] --> CrossEntropyLoss
```

**Key design choices:**
- `pack_padded_sequence` ensures the LSTM never processes padding tokens, giving clean final hidden states regardless of sentence length.
- The classifier head uses a two-layer MLP (128 -> 64 -> 5) rather than a single linear layer, providing a non-linear transformation before the class decision.
- `nn.init.orthogonal_` is used for LSTM weights; `xavier_uniform_` for linear layers.
- Bidirectional mode is available as a single toggle -- when enabled, forward and backward final hidden states are concatenated before the classifier head (producing a 256-dim input instead of 128).

### Training Setup

| Setting | Value |
|---|---|
| Optimiser | `Adam` |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| LR scheduler | `StepLR(step_size=4, gamma=0.5)` |
| Gradient clipping | `1.0` |
| Epochs | `10` |
| Batch size | `64` |
| Seed | `42` (applied via `set_seed` before every experiment) |

---

## 5. How to Run (Google Colab)

### Step-by-step

1. **Open Google Colab** -- [colab.research.google.com](https://colab.research.google.com)

2. **Upload the zip** -- click the Files icon () in the left sidebar -> upload `dnnls_reassessment.zip`

3. **Upload the notebook** -- go to `File -> Upload notebook` -> select `experiment_notebook.ipynb`

4. **Run Cell 0 first** (the setup cell) -- it will:
- Automatically find the zip anywhere under `/content/` or `/root/`
- Extract it and locate the project root by scanning for `src/` + `config.yaml`
- Set `os.chdir()` and `sys.path` correctly
- Install all dependencies from `requirements.txt`
- Print ` Setup complete!` and ` src/ verified` when successful

5. **Run all remaining cells in order** (Cell 1 -> Cell 14)

> **Do not skip Cell 0.** Every subsequent cell begins with a guard check:
> ```python
> if not os.path.isdir('src'):
> raise RuntimeError('src/ not found. Did you run Cell 0 first?')
> ```

### Runtime recommendation

A **CPU runtime** is sufficient -- the model is small and the dataset subset is limited to 1,200 stories. On GPU the total runtime for all 6 experiments is under 5 minutes. On CPU expect approximately 15-25 minutes.

---

## 6. Notebook Cell Guide

| Cell # | Title | What it does |
|---|---|---|
| **Cell 0** | SETUP | Finds zip, extracts, sets paths, installs dependencies |
| **Cell 1** | Imports & Configuration | Loads all libraries, reads `config.yaml`, sets device & seed |
| **Cell 2** | Load Dataset & Inspect | Downloads `StoryReasoning`, prints a sample story with position labels |
| **Cell 3** | Build Vocabulary & Data Loaders | Builds vocab (5,000 tokens), creates 80/20 split, prints class distribution |
| **Cell 4** | Helper Functions | Defines `train_one_epoch`, `evaluate`, and `run_experiment` |
| **Cell 5** | Baseline Model | Builds baseline, prints architecture, runs sanity forward pass, trains |
| **Cell 6** | Experiment 1 | Trains model with `hidden_dim=256` |
| **Cell 7** | Experiment 2 | Trains model with `dropout=0.5` |
| **Cell 8** | Experiment 3 | Trains model with `dropout=0.0` |
| **Cell 9** | Experiment 4 | Trains model with `num_layers=2` |
| **Cell 10** | Experiment 5 | Trains model with `bidirectional=True` |
| **Cell 11** | Results Table | Assembles and prints full results table, saves CSV |
| **Cell 12** | Training Curves | 2x3 subplot grid of train/val loss for all experiments |
| **Cell 13** | Accuracy Bar Chart | Bar chart of best val accuracy with random-baseline reference line |
| **Cell 14** | Final Summary | Prints best experiment, improvement over random chance |

Each experiment cell prints a table like:

```
============================================================
Experiment: Exp 1 - Larger Hidden (256) | params: 748,293
============================================================
Epoch Train Loss Val Loss Val Acc
------------------------------------------------------
1 1.6043 1.6021 0.2183
2 1.5812 1.5744 0.2401
...
```

---

## 7. The 5 Experiments

Each experiment changes **exactly one** hyperparameter from the baseline. All other settings remain identical.

| # | Experiment Name | Parameter Changed | From | To | Hypothesis |
|---|---|---|---|---|---|
| -- | **Baseline** | -- | -- | -- | LSTM hidden=128, 1 layer, dropout=0.3 |
| 1 | **Larger Hidden** | `hidden_dim` | 128 | **256** | Larger hidden state encodes richer narrative patterns |
| 2 | **Higher Dropout** | `dropout` | 0.3 | **0.5** | Stronger regularisation reduces overfitting on small data |
| 3 | **No Dropout** | `dropout` | 0.3 | **0.0** | Removing regularisation causes overfitting (train/val gap) |
| 4 | **Two LSTM Layers** | `num_layers` | 1 | **2** | Stacked LSTMs model hierarchical temporal structure |
| 5 | **Bidirectional LSTM** | `bidirectional` | False | **True** | Reading sentence in both directions improves position cues |

### Experiment code pattern

Each experiment is a single `build_classifier` call with one override:

```python
# Experiment 1 -- only hidden_dim changes
model_exp1 = build_classifier(cfg, vocab_size=len(vocab), hidden_dim=256).to(device)

# Experiment 5 -- only bidirectional changes
model_exp5 = build_classifier(cfg, vocab_size=len(vocab), bidirectional=True).to(device)
```

---

## 8. Outputs & Results Files

All outputs are saved to the `results/` directory automatically.

| File | Description |
|---|---|
| `results/training_curves.png` | 2x3 grid -- train loss (solid) and val loss (dashed) per epoch for all 6 runs |
| `results/ablation_bar.png` | Bar chart of best validation accuracy per experiment; red dashed line at 0.20 (random chance) |
| `results/results_table.csv` | Full results table with columns: Experiment, Modification, Train Loss, Val Loss, Val Accuracy, Best Val Accuracy |
| `results/checkpoints/clf_vocab.pkl` | Serialised vocabulary -- reloaded automatically on subsequent runs (no rebuild needed) |

### Results table format (CSV)

```
Experiment,Modification,Train Loss,Val Loss,Val Accuracy,Best Val Accuracy
Baseline (hidden=128, 1 layer, dropout=0.3),Baseline,...
Exp 1 - Larger Hidden (256),hidden_dim: 128 -> 256,...
Exp 2 - Higher Dropout (0.5),dropout: 0.3 -> 0.5,...
Exp 3 - No Dropout (0.0),dropout: 0.3 -> 0.0 (removed),...
Exp 4 - Two LSTM Layers,num_layers: 1 -> 2 (stacked LSTM),...
Exp 5 - Bidirectional LSTM,bidirectional: False -> True,...
```

---

## 9. Configuration Reference

All hyperparameters are controlled from `config.yaml`. No code changes are needed to adjust settings.

```yaml
model:
text_classifier:
vocab_size: 5000 # vocabulary size
embed_dim: 128 # word embedding dimension
hidden_dim: 128 # LSTM hidden state size (Exp 1 overrides to 256)
num_layers: 1 # number of LSTM layers (Exp 4 overrides to 2)
dropout: 0.3 # dropout probability (Exp 2->0.5, Exp 3->0.0)
num_classes: 5 # always 5 -- story positions 1 to 5
bidirectional: false # (Exp 5 overrides to true)

training:
batch_size: 64
learning_rate: 0.001
weight_decay: 0.0001
num_epochs: 10
max_text_len: 60 # max tokens per sentence (longer sentences are truncated)
train_subset: 1200 # number of stories to load (increase for better accuracy)
seed: 42

paths:
checkpoint_dir: results/checkpoints
log_dir: results/logs
```

> To use more data, increase `train_subset` (max: `len(ds['train'])`). Each additional 100 stories adds ~500 training samples.

---

## 10. Analysis Questions Summary

Full answers (200 words each) are written as Markdown cells in the notebook (Cells 15-20).

| Question | Key Finding |
|---|---|
| **Q1. Best modification?** | **Exp 1 (hidden=256)** -- larger hidden state captures richer narrative cues across sentence positions |
| **Q2. Which caused overfitting?** | **Exp 3 (no dropout)** -- training loss falls but val loss plateaus/rises; model memorises training stories |
| **Q3. How to detect overfitting?** | Training and val loss **diverge** across epochs; large final gap (e.g. train=0.8, val=1.5) confirms it |
| **Q4. Did bigger always help?** | No -- **Exp 4 (2 layers)** can perform worse due to vanishing gradients and overfitting on small data |
| **Q5. Why is the task hard?** | Single sentences lack absolute temporal markers; middle positions (2-4) share similar language; class imbalance between distinctive (1, 5) and ambiguous (2, 3, 4) positions |

---

## 11. Marking Criteria Coverage

| Criterion (20% each) | Where covered |
|---|---|
| **Dataset construction** | Cell 3 -- text extraction, label creation, 80/20 split, class distribution printed |
| **Working training pipeline** | Cell 4 (helpers) + Cells 5-10 (training loops with loss + accuracy per epoch) |
| **Correct loss & accuracy computation** | `CrossEntropyLoss` in `train_one_epoch`; `correct/total` in `evaluate`; both reported every epoch |
| **Five meaningful experiments** | Cells 6-10 -- each changes exactly ONE parameter; results table in Cell 11 |
| **Viva explanation** | Markdown analysis in Cells 15-20 (Q1-Q5, 200 words each) |

---

## 12. Dependencies

Installed automatically by Cell 0 from `requirements.txt`.

| Package | Purpose |
|---|---|
| `torch` | LSTM model, training, CrossEntropyLoss |
| `datasets` | HuggingFace StoryReasoning dataset loader |
| `numpy` | Array operations |
| `matplotlib` | Loss curves and bar chart |
| `pandas` | Results table formatting and CSV export |
| `pyyaml` | Load `config.yaml` |

---

*DNN Reassessment -- Text-Only Story Sequence Position Classification*
