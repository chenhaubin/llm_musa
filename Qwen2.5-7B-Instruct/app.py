import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_musa

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@spaces.GPU
def generate(prompt, history):
    messages = [
        {"role": "system", "content": "Je bent een vriendelijke, behulpzame assistent."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response



chat_interface = gr.ChatInterface(
    fn=generate,
)
chat_interface.launch(
    server_name="0.0.0.0",  # 使服务器可以从任何地方访问
    server_port=7860,  # 设置端口
    share=True,  # 如果你想分享一个公开链接，设置为True
)
