import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main(input_text):
    # Load the tokenizer and model from the saved directory
    model_path = './finetune-gpt2/results/model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Calculate the Number of Parameters in the model being used for inference
    total_params = get_model_parameters(model)
    print(f"Total number of paramerers: {total_params}")

    # Prepare the input text you want to generate predictions for
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate Text
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    parser.add_argument('input_text', type=str, help="The input text to generate from.")
    args = parser.parse_args()
    main(args.input_text)