import torch
import torch_musa
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# 初始化 FastAPI 应用
app = FastAPI()

# 加载模型和分词器
model_name = "mosaicml/mpt-7b"  # 替换为你的本地模型路径或 Hugging Face 模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 设置设备（如果有GPU）
device = torch.device("musa")
model.to(device)

# 处理生成请求的数据结构
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    num_return_sequences: int = 1

# API 请求处理逻辑
@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # 设置生成配置
        generation_config = GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_length=request.max_length,
            pad_token_id=tokenizer.eos_token_id,  # 设定 pad_token_id 以避免警告
        )

        # 生成模型输出
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            num_return_sequences=request.num_return_sequences,
        )

        # 解码并返回生成的文本
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"generated_text": decoded_outputs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务（默认监听8000端口）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

