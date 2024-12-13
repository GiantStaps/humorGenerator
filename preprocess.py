import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config
import os

def preprocess_data():
    # Load data
    data = pd.read_csv(Config.DATASET_PATH)
    
    # Clean and prepare jokes
    data['Joke'] = data['Joke'].str.strip()
    data = data.drop_duplicates(subset='Joke')
    data = data[data['Joke'].notnull() & (data['Joke'].str.len() > 0)]
    
    # Split into train/validation
    train_data, val_data = train_test_split(data['Joke'], test_size=0.2, random_state=42)
    
    # Save to text files
    os.makedirs(os.path.dirname(Config.CLEANED_TRAIN), exist_ok=True)
    train_data.to_csv(Config.CLEANED_TRAIN, index=False, header=False)
    val_data.to_csv(Config.CLEANED_VAL, index=False, header=False)
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
