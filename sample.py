# testing the sample. (passing max length)

from datasets import load_dataset
from transformers import AutoTokenizer

# standalone inspect script — only load one example and tokenize it (no full dataset mapping)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# inspect only the first train example (fast)
text = next(ex['text'] for ex in dataset['train'] if ex['text'].strip()) # type: ignore[arg-type]
tokenized = tokenizer(text, truncation=True, max_length=128, padding='max_length')
labels = tokenized['input_ids'].copy()  # labels == input_ids for causal LM

print("text (first 200 chars):", text[:200]) # type: ignore[arg-type]
print("non-pad token count:", sum(1 for t in tokenized['input_ids'] if t != tokenizer.pad_token_id))
print("input_ids[:40]:", tokenized['input_ids'][:40])
print("labels[:40]:", labels[:40])
