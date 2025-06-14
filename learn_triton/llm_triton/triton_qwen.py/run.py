'''
主程序
'''
import numpy
from transformers import AutoTokenizer
from layer import Qwen2Model, Qwen2Config
import torch
import tqdm

# load param to model
bos_token_id = 151643
eos_token_id = 151643
qw2_config = Qwen2Config(
    vocab_size=151936, 
    hidden_size = 1536, 
    num_hidden_layers = 28, 
    num_attention_heads = 12, 
    num_key_value_heads = 2, 
    intermediate_size = 8960, 
    hidden_act="silu", 
    layer_norm_eps = 1e-06, 
    max_position_embeddings=131072
)
model = Qwen2Model(qw2_config)

model.load()

max_new_tokens=200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "E:\my_project\DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True, padding_side='left')

# generate attention mask, max_length=200
attention_mask = torch.tril(torch.ones(max_new_tokens, max_new_tokens)).to(device)  # torch.Size([200, 200])

print(f"attention_mask: {attention_mask.shape}")  # torch.Size([1, 200])
print(attention_mask)  # torch.Size([1, 200])

input_text = "国家主席是谁？"

# 对输入文本分词
input_ids = tokenizer(input_text, return_tensors="pt").to(device)
input_ids = input_ids["input_ids"].tolist()[0]

init_input_len = len(input_ids)

position_ids = torch.arange(init_input_len).unsqueeze(0).to(device)  # torch.Size([1, 200])

# begin to generate
count = init_input_len
out_0 = model.forward(input_ids=input_ids, attention_mask=None, position_ids=position_ids)

max_indices = torch.argmax(out_0.squeeze(), dim=0).numpy()
input_ids.append(max_indices)  # append the first token
print(max_indices)
# TODO: decode 0

token_out = input_text + "\n"
for _ in tqdm.tqdm(range(max_new_tokens), desc="Decoding", unit="token"):
    # print(token_out, end="")
    # new position_ids with value = count
    # position_ids2 中的值是数字 6， shap是1x1
    position_ids2 = torch.tensor([[count]]).to(device)  # torch.Size([1, 1])

    out_0 = model.forward(input_ids=[max_indices], attention_mask=None, position_ids=position_ids2)
    count += 1
    max_indices = torch.argmax(out_0.squeeze(), dim=0).numpy()

    #
    if max_indices == eos_token_id:
        break
    token_out = token_out + tokenizer.decode([max_indices])
    # token_out = token_out
    print(token_out, end="", flush=True)
    # tqdm.tqdm.write(token_out, end="")
    # print(max_indices)
    # input_ids.append(max_indices)
    

# token_output = numpy.concatenate((input_ids, max_indices), axis=0)
# print(token_output)  # [  1  12  13  14  15  16  17  18  19   0   0   0   ...]
# print(f"token_output.shape: {token_output.shape}")  # (200,)

# 将输出token解码为文本
decoded_outputs = tokenizer.decode(input_ids)
print(f"decoded_outputs: {decoded_outputs}")  # decoded_outputs: 介绍一下悉尼这座城市。
    # TODO: decode 0

    # 对输出的out 进行解码，然后再将out作为输入


# for i in range(len(input_ids["input_ids"][0])):
#     input_ids["input_ids"][0][i] = input_ids["input_ids"][0][i] + 1
