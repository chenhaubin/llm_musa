import gradio as gr
import requests

# 模型接口URL
model_url = "http://127.0.0.1:8001/generate"

def generate_response(user_message):
    try:
        # 向模型服务发送POST请求
        response = requests.post(model_url, json={"prompt": user_message})
        
        if response.status_code == 200:
            # 打印返回的JSON数据
            response_data = response.json()
            print(f"Response JSON: {response_data}")
            # 获取生成的文本
            generated_text = response_data.get("generated_text", ["No result returned"])[0]
            return generated_text
        else:
            return "Error: " + response.text
    except Exception as e:
        return f"Request failed: {str(e)}"

# Gradio界面
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("### MPT-7b模型生成对话")
        
        # 输入框
        user_message = gr.Textbox(lines=2, placeholder="Enter your message...", label="User Input")
        
        # 生成按钮
        submit_button = gr.Button("Generate Response")
        
        # 输出框
        chatbot = gr.Textbox(lines=5, label="Generated Text", interactive=False)
        
        # 按钮点击时生成响应
        submit_button.click(fn=generate_response, inputs=user_message, outputs=chatbot)

demo.launch(server_name="0.0.0.0", server_port=7860)

