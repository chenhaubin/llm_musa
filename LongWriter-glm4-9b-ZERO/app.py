import torch
import torch_musa
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

torch.jit._state.disable()

MODEL = "THUDM/LongWriter-glm4-9b"

TITLE = "<h1><center>LongWriter-glm4-9b</center></h1>"

PLACEHOLDER = """
<center>
<p>Hi! I'm LongWriter-glm4-9b, capable of generating 10,000+ words. How can I assist you today?</p>
</center>
"""

CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
"""

# Explicitly set device to MUSA:0 or fallback to CPU
try:
    if torch.musa.is_available():
        device = "musa"  # Use MUSA if available
    else:
        device = "cpu"  # Fallback to CPU
except Exception as e:
    print(f"Error initializing MUSA device: {e}")
    device = "cpu"  # Fallback to CPU if MUSA is not available

if torch.musa.is_available():
    print("MUSA is available.")
else:
    print("MUSA is not available.")

# Load tokenizer and model with explicit device mapping
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Explicitly set device_map to None and load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=None  # Disable device_map to control device manually
)

# Ensure the model is moved to the correct device explicitly
model = model.to(device)
print(f"Using device: {device}")

def generate_response(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    top_p: float = 1.0,
    top_k: int = 50,
):
    """
    Generate a response from the model.
    """
    print(f"Message: {message}")
    print(f"History: {history}")

    # Initialize chat history
    chat_history = history or []

    try:
        # Ensure input tensors are also on the correct device
        inputs = tokenizer(message, return_tensors="pt").to(device)
        print(f"Using device: {device}")
        # Generate response from model using the `chat` method
        response, chat_history = model.chat(
            tokenizer,
            message,
            history=chat_history,
            max_new_tokens=max_new_tokens,  # Explicitly use only max_new_tokens
            max_length=None,  # Ensure max_length is not set
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Append the latest conversation
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})

        return chat_history, chat_history
    except Exception as e:
        print(f"Error: {e}")
        return [{"role": "assistant", "content": f"An error occurred: {e}"}], history

with gr.Blocks(css=CSS, theme="soft") as demo:
    gr.HTML(TITLE)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")
    chatbot = gr.Chatbot(type="messages", height=600, placeholder=PLACEHOLDER)
    state = gr.State([])  # Initialize state

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                lines=3,
            )
            submit_button = gr.Button("Submit")

            system_prompt = gr.Textbox(
                value="You are a helpful assistant capable of generating long-form content.",
                label="System Prompt",
                visible=False,
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.1,
                value=0.7,
                label="Temperature",
            )
            max_new_tokens = gr.Slider(
                minimum=512,
                maximum=8192,
                step=512,
                value=2048,
                label="Max new tokens",
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=1.0,
                label="Top p",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=50,
                label="Top k",
            )

    submit_button.click(
        generate_response,
        inputs=[user_input, state, system_prompt, temperature, max_new_tokens, top_p, top_k],
        outputs=[chatbot, state],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7871,
        share=True,
    )

