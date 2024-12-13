from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config

def generate_joke(prompt):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_OUTPUT_DIR)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)

    # Generate joke
    outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    joke = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return joke

if __name__ == "__main__":
    prompt = input("Enter a joke prompt: ")
    print("Generated Joke:", generate_joke(prompt))
