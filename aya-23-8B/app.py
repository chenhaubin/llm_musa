import torch
import torch_musa
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型目录 ./aya-23-8b
MODEL_DIR = 'CohereForAI/aya-23-8B'

# 这里我们使用 Hugging Face Transformers 来加载模型
try:
    # 加载模型和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.eval()  # 切换为评估模式
    model.to('musa')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

# FastAPI 设置
app = FastAPI()

# 请求和响应格式
class MessageRequest(BaseModel):
    message: str
    conversation_id: str = None  # 允许传入会话 ID

class MessageResponse(BaseModel):
    conversation_id: str
    response: str

# 用于存储会话的字典
active_conversations = {}

@app.post("/generate", response_model=MessageResponse)
async def generate_response(request: MessageRequest):
    global model, tokenizer
    # 获取请求中的消息和会话 ID
    message = request.message
    conversation_id = request.conversation_id or str(uuid.uuid4())  # 如果没有会话 ID，则生成一个新的 ID

    if model is None or tokenizer is None:
        return {"error": "Model not loaded successfully"}

    # 处理消息生成的逻辑
    try:
        # 对消息进行 Tokenizer 编码
        inputs = tokenizer(message, return_tensors="pt")

        # 使用模型进行推理
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=100)  # 可根据需要调整最大长度

        # 解码模型输出
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 更新会话状态（如果需要）
        active_conversations[conversation_id] = response_text

        # 返回响应数据
        return MessageResponse(conversation_id=conversation_id, response=response_text)
    except Exception as e:
        return {"error": f"Error during model inference: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # 启动 FastAPI 服务，暴露到 0.0.0.0 上，允许外部访问
    uvicorn.run(app, host="0.0.0.0", port=8001)

