import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    TOKENIZER_NAME = "meta-llama/Llama-3.2-3B-Instruct"

    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    MAX_SEQ_LEN = 256

    # Generation parameters
    MAX_GEN_LEN = 100
    TOP_P = 0.9
    TOP_K = 50
    TEMPERATURE = 0.7

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
