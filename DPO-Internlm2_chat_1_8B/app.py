import gradio as gr
import llama_cpp
import copy

# 初始化模型，确保模型路径正确
model_path = "./dpo-internlm2-1_8b.Q4_K_M.gguf"  # 这里使用你提供的模型路径
llm = llama_cpp.Llama(model_path=model_path)

# 定义最大历史记录长度
MAX_HISTORY_LENGTH = 5  # 保留最近的 5 次对话

def generate_text(
    message,
    history,
    system_message,
    top_p,  # top_p 值传递时确保是 float 类型
    max_tokens,
    temperature=0.7,  # 设置默认值
):
    # 确保历史记录不会过长，保留最近的 MAX_HISTORY_LENGTH 次对话
    history = history[-MAX_HISTORY_LENGTH:]

    input_prompt = f"<s> <|im_start|>system:{system_message}<|im_end|>"
    for interaction in history:
        input_prompt = input_prompt + "<|im_start|>user:" + str(interaction[0]) + "<|im_end|><|im_start|>" + "assistant:" + str(interaction[1]) + "<|im_end|>"

    input_prompt = input_prompt + "<|im_start|>user:" + str(message) + "<|im_end|><|im_start|>assistant:"

    # 确保 top_p 是 float 类型
    top_p = float(top_p)

    # 使用Llama模型生成文本
    output = llm(
        input_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        repeat_penalty=1.2,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "[UNUSED_TOKEN_145]"],
        stream=False,  # 不使用流式生成
    )
    
    # 获取模型生成的文本
    generated_text = output["choices"][0]["text"]
    
    # 更新历史记录
    history.append((message, generated_text))

    # 返回生成的文本和更新后的历史记录
    return generated_text, history


# Gradio的界面配置
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="User Input", lines=2),
        gr.State(value=[]),  # 用于传递历史记录
        gr.Textbox(label="System Message", value="You are a helpful assistant."),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Top-p", value=0.9),
        gr.Slider(minimum=1, maximum=500, label="Max Tokens", value=150),
        gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label="Temperature", value=0.7),
    ],
    outputs=[
        gr.Textbox(label="Assistant Response"),
        gr.State(value=[])  # 对应历史记录的输出
    ],
    live=False,  # 禁用实时更新，用户输入后才会生成响应
    allow_flagging="never",  # 禁用标记功能
    theme="default",  # 设置主题
    title="AI Assistant",  # 设置界面标题
    description="A chatbot interface powered by Llama model",  # 界面描述
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 使服务器可以从任何地方访问
        server_port=7861,  # 设置端口
        share=True,  # 如果你想分享一个公开链接，设置为True
    )

