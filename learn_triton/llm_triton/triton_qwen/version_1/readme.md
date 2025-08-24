v1

优化点：
1. Linear 层，直接保存转置之后的tensor，减少重复计算，从42 token/s 增加到45 token/sm，如果使用内部sdpa算法，则可以从45上升到50 token/s
2. 在Linear 层中，使用triton的matmul_fp32和matmul_add_fp32替换原始的torch.matmul。替换完之后,速度从45变成33
3. 使用matmul_fp16 kernel 替换上一步的fp32,从33进一步降低到32.
4. 通过修改kernel，将matmul kernel 分支中，增加一个1xK 的分支，能够从33增速到37 ms。