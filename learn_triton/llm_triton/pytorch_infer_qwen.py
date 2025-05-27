# 直接调用vllm来运行模型。

# model_path = "E:\my_project\DeepSeek-R1-Distill-Qwen-1.5B"

import torch
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 从本地加载预训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "E:\my_project\DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True, padding_side='left')

# 模型输入
input_text = "介绍一下悉尼这座城市。"

# 对输入文本分词
input_ids = tokenizer(input_text, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(model_path,device_map=device)  
# 设置 device_map="auto" 会自动使用所有多卡
print(f"model： {model}")

# 加载 tokenizer（分词器）
# 分词器负责将句子分割成更小的文本片段 (词元) 并为每个词元分配一个称为输入 id 的值（数字），因为模型只能理解数字。
# 每个模型都有自己的分词器词表，因此使用与模型训练时相同的分词器很重要，否则它会误解文本。

# add_eos_token=True: 可选参数，表示在序列的末尾添加一个结束标记（end-of-sequence token），这有助于模型识别序列的结束。
# padding_side='left': 可选参数，表示 padding 应该在序列的哪一边进行，确保所有序列的长度一致。



# print input_ids shape
# input_ids["input_ids"].shape: torch.Size([1, 12])，表示输入文本被分割成了 12 个 token

# return_tensors="pt": 指定返回的数值序列的数据类型。"pt"代表 PyTorch Tensor，表示分词器将返回一个PyTorch而不是TensorFlow对象

# 生成文本回答
# max_new_tokens：模型生成的新的 token 的最大数量为 200

outputs = model.generate(input_ids["input_ids"], max_new_tokens=200)

prompt = "Once upon a time"
start = time.time()

outputs = model.generate(input_ids["input_ids"], max_new_tokens=200)

end = time.time()
num_tokens = outputs.shape[1]
elapsed = end - start
print(f"{num_tokens} tokens in {elapsed:.2f}s -> {num_tokens / elapsed:.2f} tokens/s")


print(f"type(outputs) = {type(outputs)}")   # <class 'torch.Tensor'>
print(f"outputs.shape = {outputs.shape}")   # torch.Size([1, 95])，outputs.shape是随机的，是不超过200的数

# 将输出token解码为文本
decoded_outputs = tokenizer.decode(outputs[0])
print(f"decoded_outputs： {decoded_outputs}")
