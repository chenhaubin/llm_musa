import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch_musa

# 定义 FastAPI 应用
app = FastAPI()

# 加载预训练模型和 tokenizer
model_name = "bigscience/bloom"  # 请替换为你的模型路径或名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 打印当前使用的设备
device = "musa" if torch.musa.is_available() else "cpu"
print(f"当前使用的设备: {device}")

# 如果 pad_token 和 eos_token 是相同的，需要调整 tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token

# 定义请求体的结构
class ChatRequest(BaseModel):
    prompt: str  # 接收用户输入的 prompt

class ChatResponse(BaseModel):
    output: str  # 返回模型的输出

# 定义后端服务的接口
@app.post("/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest):
    try:
        # 打印请求数据，以调试问题
        print(f"收到请求: {request.dict()}")

        # 对输入文本进行编码，并指定 max_length
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)  # max_length 设定为 512，根据模型的最大输入长度调整
        
        # 生成 attention_mask，防止 pad_token 和 eos_token 重复带来的问题
        attention_mask = inputs['attention_mask']

        # 使用模型进行推理
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=50)

        # 解码模型的输出
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return ChatResponse(output=decoded_output)
    
    except Exception as e:
        print(f"发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"发生错误: {str(e)}")

# 启动 FastAPI 服务，监听所有网络接口
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

