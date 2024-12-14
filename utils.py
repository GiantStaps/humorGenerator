import os
from datasets import Dataset

def save_data(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for entry in data:
            f.write(f"{entry}\n")

def load_data(file_path):
    with open(file_path, "r") as f:
        return [eval(line.strip()) for line in f.readlines()]

def tokenize_and_set_labels(batch, tokenizer, max_length):
    """
    Handle batched inputs for tokenization and label preparation.
    """
    # Tokenize instructions and responses separately
    instructions = tokenizer(
        batch["instruction"], truncation=True, max_length=max_length // 2, add_special_tokens=True, padding="max_length"
    )
    responses = tokenizer(
        batch["response"], truncation=True, max_length=max_length // 2, add_special_tokens=False, padding="max_length"
    )

    # Concatenate instructions and responses
    input_ids = [
        instruction + response
        for instruction, response in zip(instructions["input_ids"], responses["input_ids"])
    ]
    attention_mask = [
        instruction_mask + response_mask
        for instruction_mask, response_mask in zip(instructions["attention_mask"], responses["attention_mask"])
    ]

    # Set labels: ignore instruction tokens in the loss
    labels = [
        [-100] * len(instruction) + response
        for instruction, response in zip(instructions["input_ids"], responses["input_ids"])
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def prepare_dataset(file_path, tokenizer, max_length):
    data = load_data(file_path)
    dataset = Dataset.from_list(data)

    return dataset.map(
        lambda batch: tokenize_and_set_labels(batch, tokenizer, max_length),
        batched=True  # Process in batches
    )
