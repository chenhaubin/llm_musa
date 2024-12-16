from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch_musa

# 初始化 FastAPI 应用
app = FastAPI()

# 加载 Yi-1.5-34B-Chat 模型和分词器
model_name = "01-ai/Yi-1.5-34B"  # 替换为你的本地模型路径

device = torch.device("musa")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义请求体格式
class ChatRequest(BaseModel):
    user_input: str

# 聊天接口
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input

    # 编码用户输入
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    # 生成回复
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # 解码生成的文本
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"response": bot_response}

# 启动 FastAPI 服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

