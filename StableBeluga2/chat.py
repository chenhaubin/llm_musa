import gradio as gr
import requests

# 定义一个函数来调用后端的 /chat 接口
def chat_with_model(input_text):
    url = "http://127.0.0.1:5000/chat"  # 后端服务地址
    payload = {"message": input_text}  # 发送的请求数据
    response = requests.post(url, json=payload)  # 发送 POST 请求
    
    # 检查是否成功返回响应
    if response.status_code == 200:
        return response.json().get('response', '没有返回消息')
    else:
        return "请求失败，错误代码: " + str(response.status_code)

# 创建一个 Gradio 界面，包含输入框、按钮和输出框
with gr.Blocks() as iface:
    input_text = gr.Textbox(label="请输入消息")  # 输入框
    output_text = gr.Textbox(label="模型响应")  # 输出框
    submit_btn = gr.Button("发送消息")  # 按钮

    # 定义按钮点击时的事件处理
    submit_btn.click(chat_with_model, inputs=input_text, outputs=output_text)

# 启动前端界面并设置暴露的 IP 和端口
iface.launch(
    server_name="0.0.0.0",  # 监听所有 IP 地址
    server_port=7860,  # 设置端口为 7860
    share=False  # 设置为 False，避免公开 URL（根据需求决定）
)

