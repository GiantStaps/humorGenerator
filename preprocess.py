import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config
from utils import save_data

def preprocess_data(input_file):
    # Load jokes dataset
    jokes = pd.read_csv(input_file)["Joke"].dropna().str.strip()

    # Add instructions to each joke
    instruction_template = "Tell me a joke."
    data = [{"instruction": instruction_template, "response": joke} for joke in jokes]

    # Split into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save processed data
    save_data(train_data, f"{Config.DATA_DIR}/train.jsonl")
    save_data(val_data, f"{Config.DATA_DIR}/val.jsonl")

    print(f"Preprocessing complete. Train and validation data saved to {Config.DATA_DIR}/")

if __name__ == "__main__":
    Config.ensure_dirs()
    preprocess_data(f"{Config.DATA_DIR}/shortjokes.csv")
