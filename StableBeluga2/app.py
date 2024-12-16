import torch
import torch_musa
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

# 初始化Flask应用
app = Flask(__name__)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga2", use_fast=False)

# 设置 pad_token 为 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型并转移到 MUSA 设备
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/StableBeluga2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"  # 自动分配设备，确保模型适应GPU显存
)

print(f"Using device: {model.device}")

# 定义系统提示
system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # 获取用户消息
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # 构建提示语
        prompt = f"{system_prompt}### User: {user_message}\n\n### Assistant:\n"

        # 对输入进行分词处理，并限制最大长度为 256 tokens
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=256).to("musa")

        # 生成模型输出
        with torch.no_grad():  # 关闭梯度计算
            output = model.generate(
                **inputs,
                do_sample=True,
                top_p=0.95,
                top_k=0,
                max_new_tokens=64,
                temperature=0.7  # 适度控制生成的多样性
            )

        # 解码生成的文本并返回
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 运行 Flask 服务
    app.run(host='0.0.0.0', port=5000, debug=True)

