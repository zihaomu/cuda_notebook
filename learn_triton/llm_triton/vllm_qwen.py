from vllm import LLM, SamplingParams
import time

# 模型路径（支持 Hugging Face 本地路径或模型名）
model_path = r"/mnt/e/my_project/DeepSeek-R1-Distill-Qwen-1.5B"


prompts = [
    "介绍一下悉尼这座城市。",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

llm = LLM(model=model_path, tensor_parallel_size=1)

# warm up
outputs = llm.generate(prompts, sampling_params)

start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()

# output_text = outputs[0].outputs[0].text
# num_tokens = outputs[0].outputs[0].token_count  # ✅ 这是 vLLM 返回的真实 token 数量


# elapsed = end - start
# speed = num_tokens / elapsed

# # 输出结果
# print(f"Generated {num_tokens} tokens in {elapsed:.2f}s -> {speed:.2f} tokens/s")
# print("Output text:")
# print(output_text)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text # outputs[0] 是因为 n=1 (每个 prompt 生成 1 个结果)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# # 初始化 vLLM 模型（默认使用 GPU）
# llm = LLM(model=model_path)

# # 设置生成参数
# sampling_params = SamplingParams(max_tokens=200)

# # 输入文本
# input_text = "介绍一下悉尼这座城市。"

# # 推理
# start = time.time()
# outputs = llm.generate(prompt=input_text, sampling_params=sampling_params)
# end = time.time()

# # 提取文本（vLLM 会返回多个候选）
# generated_text = outputs[0].outputs[0].text

# # 输出
# num_tokens = len(generated_text.split())  # 粗略估计 token 数
# elapsed = end - start
# print(f"{num_tokens} tokens in {elapsed:.2f}s -> {num_tokens / elapsed:.2f} tokens/s")

# print("generated_text:")
# print(generated_text)

