Here is an improved version of your `README.md` — it keeps your original voice and structure but makes the language sharper, more professional, and clear for submission or public use on GitHub.

---

# transformer-word-predictor

This project is a **Next Word Prediction** model built using a **pretrained transformer** (GPT-2). It demonstrates how to fine-tune a language model and evaluate its performance using Hugging Face's `transformers` and `datasets` libraries.

---

## Problem Statement

The goal is to **design and train a transformer-based language model** that predicts the next word in a given text sequence. This task is a core problem in Natural Language Processing (NLP) and is widely used in applications like autocomplete, text generation, and writing assistants.

---

## Objectives

* Build a next-word predictor using transformer architecture (GPT-2).
* Fine-tune a pretrained language model on a textual dataset.
* Evaluate the model using **perplexity** (and optionally top-k accuracy).
* Apply best practices for tokenizer alignment, text preprocessing, and model adaptation.

---

## Dataset

Dataset used: [WikiText-2 (Hugging Face)](https://huggingface.co/datasets/mindchain/wikitext2)

This dataset contains **cleaned and structured Wikipedia articles** and is widely used for language modeling tasks.

---

## Model and Training Setup

* **Model**: GPT-2 (`GPT2LMHeadModel`)
* **Tokenizer**: GPT2TokenizerFast
* **Training**: Hugging Face `Trainer` API

Training parameters:

```python
TrainingArguments(
    output_dir="./gpt2-finetuned-wikitext2",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=1
)
```

---

## Evaluation

* **Metric used**: Perplexity
* Perplexity measures how well the model predicts the next word — **lower is better**.
* Formula to compute perplexity from loss:

```python
import math
perplexity = math.exp(loss)
```

**Note**: Top-k accuracy was skipped due to the complexity of implementing decoding-based metrics in this task.

---

## How to Run

First, install the required libraries:

```bash
pip install transformers datasets evaluate torch
```

Then run the script:

```bash
python project.py
```

Make sure you are using Python 3.10+ and have PyTorch installed.
