# Softmax cuda 优化

cpu版本 softmax 计算
是二维softmax MxN的计算。

需要实现计算的准确性

Softmax理论带宽计算。

输入维度为：[m x n], D = m x n.
按照softmax 公式，需要做以下计算：
step 1: ReduceMax, Read 为D，write 为 m
step 2: BroadcastSub，Read为D+m，write 为 D
step 3: Exp，Read为D，write为D
step 4: ReduceSum，Read为D，write为m
step 5: BroadcastDiv，Read为D，write为D

总共需要：**8\*D+4\*m.** 简化为 8*D。
举例：对于3090显卡，理论带宽为936GB/s，那么理论softmax 带宽为**936/8 = 117GB/s**.
