# 0 pull code
git clone https://github.com/chenhaubin/llm_musa.git

# 1. proxy
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

# 插入代理，并保存
vim ~/.bashrc

# 此模型为受限模型库，需通过环境变量设置令牌。
export HF_TOKEN="your_token_here"

# 2. Prepare model
python app.py

# 3. Test
bash run.sh

# port
7748