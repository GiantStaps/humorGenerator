from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from config import Config
import os

def train_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)

    # Load datasets
    data_files = {"train": Config.CLEANED_TRAIN, "validation": Config.CLEANED_VAL}
    dataset = load_dataset("text", data_files=data_files)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.MODEL_OUTPUT_DIR,
        evaluation_strategy="steps",
        logging_dir="./logs",
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        num_train_epochs=Config.MAX_EPOCHS,
        logging_steps=Config.LOGGING_STEPS,
        save_steps=Config.SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        load_best_model_at_end=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    trainer.save_model(Config.MODEL_OUTPUT_DIR)
    print("Model training completed.")

if __name__ == "__main__":
    train_model()
