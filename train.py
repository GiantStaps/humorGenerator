from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from config import Config
from utils import prepare_dataset

def train_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

    # Prepare datasets
    train_dataset = prepare_dataset(f"{Config.DATA_DIR}/train.jsonl", tokenizer, Config.MAX_SEQ_LEN)
    val_dataset = prepare_dataset(f"{Config.DATA_DIR}/val.jsonl", tokenizer, Config.MAX_SEQ_LEN)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{Config.OUTPUT_DIR}/logs",
        load_best_model_at_end=True
    )

    # Initialize and train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    trainer.save_model(Config.OUTPUT_DIR)
    print(f"Model saved to {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    Config.ensure_dirs()
    train_model()
