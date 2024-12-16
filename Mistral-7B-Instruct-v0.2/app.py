from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# 设置 pad_token_id 为 eos_token_id
tokenizer.pad_token = tokenizer.eos_token  # 将 pad_token 设置为 eos_token
model.config.pad_token_id = tokenizer.pad_token_id  # 更新模型配置

# 引入 torch_musa 并移动模型到 MUSA 设备
import torch_musa

device = torch.device("musa" if torch_musa.is_available() else "cpu")
print("device is",device)
model = model.half().to(device)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')  # 获取用户输入的消息
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # 将用户输入进行编码
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 生成响应
    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask', None),
        max_new_tokens=50,
        do_sample=True
    )

    # 解码生成的响应
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)

