import gradio as gr
import requests

# 假设你的聊天服务通过 HTTP POST 请求与模型交互
CHAT_SERVICE_URL = "http://127.0.0.1:8000/chat"

def chat_with_model(user_input):
    # 发送用户输入到聊天服务，假设服务返回一个 JSON 响应
    response = requests.post(CHAT_SERVICE_URL, json={"prompt": user_input})  # 确保字段是 "prompt"
    
    # 如果请求成功，返回模型的回答
    if response.status_code == 200:
        model_output = response.json().get("output", "没有返回内容")
        return model_output
    else:
        return f"请求失败，状态码: {response.status_code}"

# 创建 Gradio 接口，去掉 live=True，添加一个按钮
iface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(label="输入你的问题"),  # 用户输入框
    outputs=gr.Textbox(label="模型回复"),
    live=False  # 禁用实时输入触发
)

# 启动 Gradio 服务，并使其支持外网访问
iface.launch(share=True, server_name="0.0.0.0", server_port=7860)

