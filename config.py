import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "data", "shortjokes.csv")
    CLEANED_TRAIN = os.path.join(BASE_DIR, "data", "train.txt")
    CLEANED_VAL = os.path.join(BASE_DIR, "data", "val.txt")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "model")
    
    # Hugging Face Model
    MODEL_NAME = "meta-llama/LLaMA-3-8B"
    TOKENIZER_NAME = "meta-llama/LLaMA-3-8B"

    # Training Hyperparameters
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    MAX_EPOCHS = 3
    SAVE_STEPS = 1000
    LOGGING_STEPS = 100

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
