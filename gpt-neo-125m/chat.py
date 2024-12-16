import gradio as gr
import requests

# 模型接口 URL
model_url = "http://127.0.0.1:8001/generate"

def generate_response(user_message):
    try:
        # 向模型服务发送POST请求
        response = requests.post(model_url, json={"prompt": user_message})
        
        if response.status_code == 200:
            # 获取返回的 JSON 数据
            response_data = response.json()
            # 获取生成的文本
            generated_text = response_data.get("generated_text", "No result returned")
            return generated_text
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"

# Gradio 界面
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("### GPT-Neo 125M 聊天模型")
        
        # 输入框
        user_message = gr.Textbox(lines=2, placeholder="输入您的消息...", label="用户输入")
        
        # 生成按钮
        submit_button = gr.Button("生成响应")
        
        # 输出框
        chatbot = gr.Textbox(lines=5, label="生成的文本", interactive=False)
        
        # 按钮点击时生成响应
        submit_button.click(fn=generate_response, inputs=user_message, outputs=chatbot)

demo.launch(server_name="0.0.0.0", server_port=7860)

