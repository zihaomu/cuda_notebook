# speed test
本项目用 DeepSeek-R1-Distill-Qwen-1.5B 为基准，测试自己手写的算子和vllm的区别。

手写算子包含以下几方面：
- triton 手写算子
- c++ 手写算子，通过绑定到python中
- 不同情况的量化分支

4090
直接使用 vllm运行1.5B速度：speed input: 13.88 toks/s, output: 138.81 toks/s
使用pytoch运行速度：46.49 tokens/s

可见，pytorch速度还是比vllm慢很多