# dataset --> tokenizes it to numeric i.e [32, 123290] --> add labels (crucial, predicts next token) --> set training args --> train 
# Feed token IDs into GPT-2. GPT-2 predicts the next token probabilities.
# Compare prediction with actual next token (labels) and compute loss.
# Backpropagate → slightly adjust GPT-2’s weights. Trainer does this.

# gpt2 with some random weights from wikitext-2-raw-v1 dataset.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

tokenizer.pad_token = tokenizer.eos_token

# tokenize the dataset
def tokenize_function(examples):
    inputs =  tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./finetune-gpt2/results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./finetune-gpt2/logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# to resume --> resume_from_checkpoint='./finetune-gpt2/results/checkpoint-5000'
trainer.train()

# save the model and tokenizer explicitly
model_output_dir = './finetune-gpt2/results/model'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)