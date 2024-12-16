from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import torch_musa

# 加载 GPT-Neo 模型和 tokenizer
model_name = "EleutherAI/gpt-neo-125m"  # 也可以直接加载本地路径，如果你本地有模型
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 检查模型所在的设备（GPU/CPU）
device = "musa" if torch.musa.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# 配置 FastAPI
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
async def generate(request: GenerateRequest):
    # 编码输入的 prompt，并将其移动到正确的设备
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)

    # 生成文本
    with torch.no_grad():
        output = model.generate(input_ids, max_length=request.max_length, num_return_sequences=1)

    # 解码输出并返回
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

# 运行 FastAPI 服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

