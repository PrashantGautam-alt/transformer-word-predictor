from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import math
import torch

# 1. Load Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. Load Tokenizer and set padding token
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenization Function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Load Pre-trained GPT2 Model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Add pad token support

# 5. Define Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-wikitext2",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    report_to="none",
    logging_dir="./logs"
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8. Start Training
trainer.train()

# 9. Evaluation: Compute Perplexity
def compute_perplexity(eval_dataset):
    model.eval()
    losses = []
    for i in range(len(eval_dataset)):
        input_ids = torch.tensor(eval_dataset[i]['input_ids']).unsqueeze(0)  # Batch size 1
        labels = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss.item()
        losses.append(loss)
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)

# 10. Print Perplexity
print("Perplexity:", compute_perplexity(tokenized_datasets["validation"]))
