from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_musa

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./", device_map="auto", trust_remote_code=True)

inputs = tokenizer('The winter in Beijing is', return_tensors='pt')
inputs = inputs.to('musa')

# 使用 model.generate 生成预测
pred = model.generate(**inputs)

# 打印解码后的结果
print(tokenizer.decode(pred[0], skip_special_tokens=True))