import torch
import torch_musa
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 配置模型权重路径和参数
MODEL_PATH = "Phind/Phind-CodeLlama-34B-v2"  # 替换为模型文件的路径
DEVICE = "musa" if torch.musa.is_available() else "cpu"
OFFLOAD_FOLDER = "./offload_weights"  # 设置权重卸载的文件夹

print("use device", DEVICE)
# 创建卸载文件夹（如果不存在）
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

def load_model():
    """加载模型和分词器"""
    print("Loading model...")
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 提高效率

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)  # 使用慢速分词器避免 tiktoken 问题

    # 使用 accelerate 进行模型加载
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分配设备
            offload_folder=OFFLOAD_FOLDER  # 设置权重卸载路径
        )

    model = load_checkpoint_and_dispatch(
        model,
        MODEL_PATH,
        device_map="auto",
        offload_folder=OFFLOAD_FOLDER
    )

    print("Model loaded successfully.")
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    """生成模型响应"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def qa_interface(prompt):
    """Gradio 接口函数"""
    response = generate_response(prompt, tokenizer, model)
    return response

# 加载模型和分词器
tokenizer, model = load_model()

# 定义 Gradio 接口
description = """
### Phind-CodeLlama-34B 模型服务

输入问题，模型将生成答案。
"""
iface = gr.Interface(
    fn=qa_interface,
    inputs=gr.Textbox(lines=2, label="输入问题"),
    outputs=gr.Textbox(label="模型回答"),
    title="Phind-CodeLlama-34B",
    description=description,
)

# 启动 Gradio 服务
iface.launch(server_name="0.0.0.0", server_port=7860)

