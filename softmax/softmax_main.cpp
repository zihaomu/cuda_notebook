#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "profile_utils.cuh"
#include "softmax.hpp"

using namespace std;

int main() {
    print_device_info();
    const size_t rows = 4096;
    const size_t cols = 4096; // 实际中，这个数据量太大了，应该跑不起来。
    std::vector<float> input(rows * cols, 0); // 二维vector也不是一个高效
    std::vector<float> output(rows * cols, 0);// 的数据结构，不建议。

    const size_t num_repeats{1U};
    const size_t num_warmups{1U};

    const size_t m = rows;
    const size_t n = cols;
    const size_t ldm{(n + 16U - 1U) / 16U * 16U};

    float fp32_abs_tol{1.0e-4f};
    double const fp32_rel_tol{0.0e-4f};

    // 填充输入数组（示例）
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            input[i * cols + j] = static_cast<float>(i * cols + j); // 示例数据
        }
    }

    std::vector<std::pair<
    std::string,
    std::function<void(size_t, size_t, float*, float*, cudaStream_t)>>> const
    softmax_kernel_launch_functions{
        {"Naive Softmax Kernel V00", cuda_softmax_v0<float>},
        // {"Naive Softmax Kernel V01: warp reduce", cuda_softmax_v1<float>},
        {"Naive Softmax Kernel V02: warp reduce", cuda_softmax_v2<float>},
    };

    for (auto const& softmax_kernel_launch_function : softmax_kernel_launch_functions)
    {
        std::cout << softmax_kernel_launch_function.first << std::endl; // 输出 kernel 名称
        std::pair<float, float> const softmax_kernel_profile_result{
            profile_softmax<float>(
                m, n, softmax_kernel_launch_function.second,
                fp32_abs_tol, fp32_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }

    return(0);
} 