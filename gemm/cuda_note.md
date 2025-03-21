# CUDA 开发记录

'cuh' 文件和h文件一样，也是属于头文件，满足CMake的'include_directories'规则。

具体编译方便：cu和cuh一样由nvcc编译，而其他文件则由c++编译器编译。

目前的CMAKE是已经可用在windows平台跑通，但是只能编译静态库，不能编译动态库



kernel文件开发步骤：
1. 制作profile方法，如何准确的测试cuda kernel的运行时间？
2. 制作统一的结果check方法，验证gpu结果和cpu的结果是否一致。
3. 制作gpu统一的输入输出流程，保证方便的后续流畅的多版本kernel开发。


零散知识记录：

1. std::function说明

std::function 时一种通用、多态的封装函数。其可用对任何可用调用的目标实体进行存储、复制和调用。封装的目标实体包括：lambda表达式、函数指针、一般函数以及函数对象。
```.cpp
typedef std::function<int(int)> F0; // 声明一个函数模板
int test(int a)
{
    return a + 1;
}

auto lambda = [](int a)->int{return a;};

class Functor
{

public:
    int operator() (int a)
    {
        return a;
    }

};

int main()
{
    F0 f0 = test;

    int res = f0(1);

    f0 = lambda;
    res = f0(2);

    f0 = Functor;
    res = f0(3);
}

```

2. cuda 线程组成，容易被忽略的点

threadIdx.x threadIdx.y threadIdx.z 这三个的关系，对应的维度和直觉相反。[x, y, z] 其中x是最内层，而y是外层。
解释参考：https://face2ai.com/CUDA-F-5-2-%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%9A%84%E6%95%B0%E6%8D%AE%E5%B8%83%E5%B1%80/里的图片。

为什么会第一个数是x，但是进去之后对应的是最内层，而不是我们直觉的[y, x]。其中y是外层，而x是内存。
`dim3 const block_dim{32U, 31U, 30U};`
`std::cout<<"Block Dim: "<<block_dim.x<<" x "<<block_dim.y<<" x "<<block_dim.z<<std::endl;`

输出是：Block Dim:  32 x 31 x 30.

3. kernel 函数的理解
```
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void checkIndex(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}
int main(int argc,char **argv)
{
  int nElem=6;
  dim3 block(3);
  dim3 grid((nElem+block.x-1)/block.x);
  printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();
  cudaDeviceReset();
  return 0;
}
```

上面的checkIndex是一个核函数，前面有个修饰符`__global__`。
除了global（设备执行，可以从主机或者设备调用）,还有device和host，分别代表只能从设备或主机调用。
核函数的限制：
1. 只能访问设备内存
2. 必须有void返回类型
3. 不支持可变数量的参数
4. 不支持静态变量
5. 显示异步行为

除此之外，后续的`<<<grid, block>>>`前面的是将thread划分成多少块，后面的是每一个块继续怎么划分。
其中block应当小于1024. grid和block维度都是三维。
grid对应的值是：
blockIdx.x
blockIdx.y
blockIdx.z
block对应的值是：
threadIdx.x
threadIdx.y
threadIdx.z
在kernel中也可以拿到这三个字段的宽度，不过这个常量是运行时，而不是编译时
gridDim.x,gridDim.y,gridDim.z
blockDim.x
blockDim.y
blockDim.z.

4. assert用法
static_assert和assert
static_assert是编译时断言，而assert是运行时断言，而且只能在debug中使用。

5. 怎么测试kernel的耗时
通常使用cudaEventCreate()、cudaEventRecord() 和 cudaEventElapsedTime()这三个函数来测试执行时间，记下来就行了。可以认为是一个范式。
例子代码：
```
__global__ void kernel(...) {}

void test()
{
  // ...

  // 方法1，默认stream场景----------------------------
  // 定义 CUDA 事件
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 记录开始时间
  cudaEventRecord(start, 0);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(...); // 调用

  // 记录结束时间
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);  // 确保 Kernel 运行完成

  // 计算时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

  // 方法2，指定stream场景----------------------------
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // 记录开始时间
  cudaStreamSynchronize(stream); // 同步
  cudaEventRecord(start, stream);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(...); // 调用

  // 记录结束时间
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);  // 确保 Kernel 运行完成

  // 计算时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
  // ...
}
```

6. Explicit instantiation的作用
在src的每个.cu文件的最后都会将模板代码进行显示实例化，这是为啥？
不加实例化，在vs2022中会有如下报错：
'''
unresolved external symbol "void __cdecl cuda_gemm_cutlass<struct __half>(unsigned __int64,unsigned __int64,unsigned __int64,struct __half *,struct __half *,struct __half *,struct __half,struct __half,struct CUstream_st *)" (??$cuda_gemm_cutlass@U__half@@@@YAX_K00PEAU__half@@11U0@2PEAUCUstream_st@@@Z) referenced in function main

'''
这里也涉及到自己的知识盲区。
C++模板代码在编译和链接过程中，模板的实例化是延迟到使用时才发生的。当编译器处理main_fp32.cpp时, 它会看到'cuda_gemm_v0<float>'的调用，然而定义在gemm_0.cu中。gemm_0.cu是一个独立的编译单元，这就会导致在编译时找不到定义。

对此常规的解决方案有两种：
  - 将模板定义放到头文件中，让调用方能在编译时获取定义
  - 显示实例化，明确告诉编译器生成模板

7. windows上加入符号导出的作用'__declspec(dllexport)'
在问题6中，我们已经解决了unresolved external symbol的报错，但是，我尝试将CMakeList.txt中的cuda_gemm改为动态库时，这个烦人的unresolved external symbol的报错又出现了。原来是符号的可见性和连接方式在静态库和动态库中行为不同。
对于静态库：会将连接时所有符号（包括实例化模板的代码）直接嵌入到可执行文件中。
对于动态库：是在运行时加载，所有函数符号都需要显示的导出，才能让其他库进行调用。

解决方案，在需要导出的函数前面加上显示导出符号，对于windows是'__declspec(dllexport)'。

8. 模板和特例化模板
'''
template <typename T>
T func(T a, T b)
{
  return a + b; 
}

// 正确的实现 一个特例化模板
template <>
float func<float>(float a, float b)
{
  return a + b;
}

// 错误的实现
float func(float a, float b)
{
  return a + b;
}
'''

9. extern __shared__
在gemm_wmma_1中，核函数开始部分加了一个'extern __shared__'，

加上这个extern声明修饰符和不加是完全不一样的。
两者的定义：
- 对于'__shared__':静态分配，在 编译时 确定大小
- 对于'extern __shared__':动态分配，在 运行时 由 cudaLaunchKernel() 指定大小

两者适用于什么情况？
- 对于'__shared__':大小提前就知道，适用于小规模使用
- 对于'extern __shared__':大小不知道，需要大规模使用共享内存，一般最大96KB，Kernel需要更通用，支持不同大小的共享内存。

举个'extern __shared__'例子：
'''
__global__ void kernel_dynamic(int N) {
    extern __shared__ float sharedData[];  // 运行时决定大小
    int tid = threadIdx.x;
    sharedData[tid] = tid * 2.0f;
    __syncthreads();
}

void call_test()
{
  int N = 1024;
  size_t sharedMemSize = N * sizeof(float);
  kernel_dynamic<<<gridDim, blockDim, sharedMemSize>>>(N); // 在这里指定决定大小
}
'''

10. deviceProp.multiProcessorCount
这个值是返回当前设备的SM硬件的数量。