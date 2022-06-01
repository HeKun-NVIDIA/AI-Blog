# CUDA中一些关键字说明

## 1 函数执行空间说明符
函数执行空间说明符表示函数是在主机上执行还是在设备上执行，以及它是可从主机调用还是从设备调用。

### 1.1 \_\_global\_\_
`__global__` 执行空间说明符将函数声明为内核。 它的功能是：

* 在设备上执行，
* 可从主机调用，
* 可在计算能力为 3.2 或更高的设备调用（有关更多详细信息，请参阅 [CUDA 动态并行性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism)）。
`__global__` 函数必须具有 void 返回类型，并且不能是类的成员。

对 `__global__` 函数的任何调用都必须指定其执行配置，如[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)中所述。

对 `__global__` 函数的调用是异步的，这意味着它在设备完成执行之前返回。

### 1.2 \_\_device\_\_
`__device__` 执行空间说明符声明了一个函数：

* 在设备上执行，
* 只能从设备调用。
`__global__` 和 `__device__` 执行空间说明符不能一起使用。

###  1.3 \_\_host\_\_
`__host__` 执行空间说明符声明了一个函数：

* 在主机上执行，
* 只能从主机调用。
相当于声明一个函数只带有 `__host__` 执行空间说明符，或者声明它没有任何 `__host__` `、__device__` 或 `__global__` 执行空间说明符； 在任何一种情况下，该函数都仅为主机编译。

`__global__` 和 `__host__` 执行空间说明符不能一起使用。

但是， `__device__` 和 `__host__` 执行空间说明符可以一起使用，在这种情况下，该函数是为主机和设备编译的。 Application Compatibility 中引入的 `__CUDA_ARCH__ `宏可用于区分主机和设备之间的代码路径：
```C++
__host__ __device__ func()
{
#if __CUDA_ARCH__ >= 800
   // Device code path for compute capability 8.x
#elif __CUDA_ARCH__ >= 700
   // Device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600
   // Device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500
   // Device code path for compute capability 5.x
#elif __CUDA_ARCH__ >= 300
   // Device code path for compute capability 3.x
#elif !defined(__CUDA_ARCH__) 
   // Host code path
#endif
}
```

###  1.4 Undefined behavior
在以下情况下，“跨执行空间”调用具有未定义的行为：
* `__CUDA_ARCH__` 定义了, 从 `__global__` 、 `__device__` 或 `__host__ __device__` 函数到 `__host__` 函数的调用。
* `__CUDA_ARCH__` 未定义，从 `__host__` 函数内部调用 `__device__` 函数。

####  1.5 `__noinline__` and `__forceinline__`

编译器在认为合适时内联任何 `__device__` 函数。

`__noinline__` 函数限定符可用作提示编译器尽可能不要内联函数。

`__forceinline__` 函数限定符可用于强制编译器内联函数。

`__noinline__` 和 `__forceinline__` 函数限定符不能一起使用，并且两个函数限定符都不能应用于内联函数。

##  2 Variable Memory Space Specifiers

变量内存空间说明符表示变量在设备上的内存位置。

在设备代码中声明的没有本节中描述的任何 `__device__`、`__shared__` 和 `__constant__` 内存空间说明符的自动变量通常驻留在寄存器中。 但是，在某些情况下，编译器可能会选择将其放置在本地内存中，这可能会产生不利的性能后果，如[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)中所述。

###  2.1 \_\_device\_\_
`__device__` 内存空间说明符声明了一个驻留在设备上的变量。

在接下来的三个部分中定义的其他内存空间说明符中最多有一个可以与 `__device__` 一起使用，以进一步表示变量属于哪个内存空间。 如果它们都不存在，则变量：

* 驻留在全局内存空间中，
* 具有创建它的 CUDA 上下文的生命周期，
* 每个设备都有一个不同的对象，
* 可从网格内的所有线程和主机通过运行时库 (`cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()`) 访问。

###  2.2. \_\_constant\_\_
`__constant__` 内存空间说明符，可选择与 `__device__` 一起使用，声明一个变量：

* 驻留在常量的内存空间中，
* 具有创建它的 CUDA 上下文的生命周期，
* 每个设备都有一个不同的对象，
* 可从网格内的所有线程和主机通过运行时库 (`cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()`) 访问。

###  2.3 \_\_shared\_\_

`__shared__` 内存空间说明符，可选择与 `__device__` 一起使用，声明一个变量：

* 驻留在线程块的共享内存空间中，
* 具有块的生命周期，
* 每个块有一个不同的对象，
* 只能从块内的所有线程访问，
* 没有固定地址。

将共享内存中的变量声明为外部数组时，例如:
```C++
extern __shared__ float shared[];
```
数组的大小在启动时确定（请参阅[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)）。 以这种方式声明的所有变量都从内存中的相同地址开始，因此必须通过偏移量显式管理数组中变量的布局。 例如，如果想要在动态分配的共享内存中等价于，
```C++
short array0[128];
float array1[64];
int   array2[256];
```
可以通过以下方式声明和初始化数组：
```C++
extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array; 
    float* array1 = (float*)&array0[128];
    int*   array2 =   (int*)&array1[64];
}
```
#### 请注意，指针需要与它们指向的类型对齐，因此以下代码不起作用，因为 array1 未对齐到 4 个字节。
```C++
extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array; 
    float* array1 = (float*)&array0[127];
}
```
[表 4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types__alignment-requirements-in-device-code) 列出了内置向量类型的对齐要求。

###  2.4. __managed__
`__managed__` 内存空间说明符，可选择与 `__device__` 一起使用，声明一个变量：

* 可以从设备和主机代码中引用，例如，可以获取其地址，也可以直接从设备或主机功能读取或写入。
* 具有应用程序的生命周期。
有关更多详细信息，请参阅 [`__managed__` 内存空间](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#managed-specifier)说明符。

###  2.5. __restrict__
nvcc 通过 `__restrict__` 关键字支持受限指针。

C99中引入了受限指针，以缓解存在于c类型语言中的混叠问题，这种问题抑制了从代码重新排序到公共子表达式消除等各种优化。

下面是一个受混叠问题影响的例子，使用受限指针可以帮助编译器减少指令的数量：
```C++
void foo(const float* a,
         const float* b,
         float* c)
{
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    c[3] = a[0] * a[1];
    c[4] = a[0] * b[0];
    c[5] = b[0];
    ...
}
```

此处的效果是减少了内存访问次数和减少了计算次数。 这通过由于“缓存”负载和常见子表达式而增加的寄存器压力来平衡。

由于寄存器压力在许多 CUDA 代码中是一个关键问题，因此由于占用率降低，使用受限指针会对 CUDA 代码产生负面性能影响。

##  3. Built-in Vector Types

###  3.1. char, short, int, long, longlong, float, double
这些是从基本整数和浮点类型派生的向量类型。 它们是结构，第一个、第二个、第三个和第四个组件可以分别通过字段 `x、y、z 和 w` 访问。 它们都带有 `make_<type name> `形式的构造函数； 例如，
```C++
int2 make_int2(int x, int y);
```
它创建了一个带有 `value(x, y)` 的 `int2` 类型的向量。
向量类型的对齐要求在下表中有详细说明。

|Type|	Alignment|
|----|----|
|char1, uchar1|	1|
|char2, uchar2|	2|
|char3, uchar3|	1|
|char4, uchar4	|4|
|short1, ushort1|	2|
|short2, ushort2|	4|
|short3, ushort3|	2|
|short4, ushort4|	8|
|int1, uint1	|4|
|int2, uint2	|8|
|int3, uint3	|4|
|int4, uint4|	16|
|long1, ulong1|	4 if sizeof(long) is equal to sizeof(int) 8, otherwise|
|long2, ulong2|	8 if sizeof(long) is equal to sizeof(int), 16, otherwise|
|long3, ulong3|	4 if sizeof(long) is equal to sizeof(int), 8, otherwise|
|long4, ulong4	|16|
|longlong1, ulonglong1|	8|
|longlong2, ulonglong2	|16|
|longlong3, ulonglong3|	8|
|longlong4, ulonglong4	|16|
|float1	|4|
|float2	|8|
|float3	|4|
|float4	|16|
|double1	|8|
|double2	|16|
|double3	|8|
|double4	|16|

###  3.2. dim3
此类型是基于 uint3 的整数向量类型，用于指定维度。 定义 dim3 类型的变量时，任何未指定的组件都将初始化为 1。

##  4. Built-in Variables

###  4.1. gridDim
该变量的类型为 `dim3`（请参阅[ dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)）并包含网格的尺寸。

###  4.2. blockIdx
该变量是 `uint3` 类型（请参见 [char、short、int、long、longlong、float、double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types)）并包含网格内的块索引。

###  4.3. blockDim
该变量的类型为 `dim3`（请参阅 [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)）并包含块的尺寸。

###  4.4. threadIdx
此变量是 `uint3` 类型（请参见 [char、short、int、long、longlong、float、double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types) ）并包含块内的线程索引。

###  4.5. warpSize
该变量是 `int` 类型，包含线程中的 `warp` 大小（有关 `warp` 的定义，请参见 [SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)）。
