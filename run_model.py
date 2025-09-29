import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Configuration
MODEL_CHECKPOINT = "gpt2"
BLOCK_SIZE = 128
TRAIN_BATCH_SIZE = 8  # Increased from 2 for better GPU utilization if available
EVAL_BATCH_SIZE = 8
EPOCHS = 3           # Increased to 3 for better fine-tuning

# --- 1. Load Dataset, Tokenizer, and Model ---
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT)

# Set padding token for GPT-2 which doesn't have one by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer)) # Resize model embedding if pad token is new

# --- 2. Tokenization and Text Grouping ---
def tokenize_function(examples):
    # Process text in batches
    return tokenizer(examples["text"])

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text", "id", "title"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Drop the small remainder at the end
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    # Create the labels for language modeling
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
)

# --- 3. Define Data Collator and Training Arguments ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-wikitext2-improved",
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50, # Increased logging frequency
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(), # Use mixed precision if a GPU is available
)

# --- 4. Trainer and Training ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# --- 5. Evaluation: Compute Perplexity ---
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])

print(f"Perplexity: {perplexity:.2f}")
