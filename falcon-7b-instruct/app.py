from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_musa

# Load the tokenizer and model
model_path = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Function for inference
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Write me a poem"
response = generate_text(prompt)
print(response)

