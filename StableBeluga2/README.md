# 0 pull code
git clone https://github.com/chenhaubin/llm_musa.git

# 1. proxy
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

# 插入代理，并保存
vim ~/.bashrc

# 2. 下载依赖
pip install SentencePiece

pip install 'accelerate>=0.26.0'

# 3. Prepare model
python app.py

# 4. Test
bash run.sh

# port
7747