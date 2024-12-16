import gradio as gr
import requests

# 这里是你 FastAPI 服务的地址，确保 FastAPI 服务监听外网地址
api_url = "http://127.0.0.1:8000/chat"  # 替换为你的服务器 IP 地址

# 定义聊天函数，调用 FastAPI 服务的接口
def chat_with_model(user_input):
    response = requests.post(api_url, json={"user_input": user_input})
    
    # 确保返回的内容正确
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: Unable to get a response from the server."

# 创建 Gradio 接口
iface = gr.Interface(fn=chat_with_model, 
                     inputs=gr.Textbox(lines=2, placeholder="输入你的问题..."), 
                     outputs="text",
                     title="Yi-1.5-34B-Chat 聊天机器人",
                     description="这是一个基于 Yi-1.5-34B-Chat 模型的聊天接口。你可以输入任何问题，模型将生成回答。")

# 启动 Gradio 前端界面，监听外网
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

