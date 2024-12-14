from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config

def generate_joke(prompt):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(Config.OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(Config.OUTPUT_DIR)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate joke
    outputs = model.generate(
        inputs["input_ids"],
        max_length=Config.MAX_GEN_LEN,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        top_k=Config.TOP_K
    )

    # Decode the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Tell me a joke."
    joke = generate_joke(prompt)
    print(joke)
