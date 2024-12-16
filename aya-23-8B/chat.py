import gradio as gr
import requests

# 定义请求函数，发送 POST 请求到你本地的 API
def query_api(message):
    url = "http://127.0.0.1:8001/generate"
    headers = {"Content-Type": "application/json"}
    data = {"message": message}
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()['response']  # 假设返回的是 {'response': 'response_text'}
    else:
        return f"Error: {response.status_code}"

# 定义 Gradio 界面
def create_interface():
    with gr.Blocks() as demo:
        # 输入框
        message_input = gr.Textbox(label="Enter your message", placeholder="Type a message here...")
        
        # 按钮
        button = gr.Button("Send Request")
        
        # 输出框
        response_output = gr.Textbox(label="Response from Model", interactive=False)
        
        # 定义按钮点击时的行为
        button.click(fn=query_api, inputs=message_input, outputs=response_output)

    return demo

# 启动 Gradio 应用，设置服务器绑定的 IP 和端口
interface = create_interface()

# 将 Gradio 界面暴露到外部IP和端口7860
interface.launch(server_name="0.0.0.0", server_port=7860)

