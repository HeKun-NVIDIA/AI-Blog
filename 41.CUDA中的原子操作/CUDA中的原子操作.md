# CUDA中的原子操作.md


原子函数对驻留在全局或共享内存中的一个 32 位或 64 位字执行读-修改-写原子操作。 例如，`atomicAdd()` 在全局或共享内存中的某个地址读取一个字，向其中加一个数字，然后将结果写回同一地址。 该操作是原子的，因为它保证在不受其他线程干扰的情况下执行。 换句话说，在操作完成之前，没有其他线程可以访问该地址。 原子函数不充当内存栅栏，也不意味着内存操作的同步或排序约束（有关内存栅栏的更多详细信息，请参阅[内存栅栏函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)）。 原子函数只能在设备函数中使用。

原子函数仅相对于特定集合的线程执行的其他操作是原子的：

* 系统范围的原子：当前程序中所有线程的原子操作，包括系统中的其他 CPU 和 GPU。 这些以 `_system` 为后缀，例如 `atomicAdd_system`。
* 设备范围的原子：当前程序中所有 CUDA 线程的原子操作，在与当前线程相同的计算设备中执行。 这些没有后缀，只是以操作命名，例如 `atomicAdd`。
* Block-wide atomics：当前程序中所有 CUDA 线程的原子操作，在与当前线程相同的线程块中执行。 这些以 _block 为后缀，例如 `atomicAdd_block`。

在以下示例中，CPU 和 GPU 都以原子方式更新地址 `addr` 处的整数值：
```C++
__global__ void mykernel(int *addr) {
  atomicAdd_system(addr, 10);       // only available on devices with compute capability 6.x
}

void foo() {
  int *addr;
  cudaMallocManaged(&addr, 4);
  *addr = 0;

   mykernel<<<...>>>(addr);
   __sync_fetch_and_add(addr, 10);  // CPU atomic operation
}
```

请注意，任何原子操作都可以基于 `atomicCAS()`（Compare And Swap）来实现。 例如，用于双精度浮点数的 atomicAdd() 在计算能力低于 6.0 的设备上不可用，但可以按如下方式实现：
```C++
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
```

以下设备范围的原子 API 有系统范围和块范围的变体，但以下情况除外：

* 计算能力低于 6.0 的设备只支持设备范围的原子操作，
* 计算能力低于 7.2 的 Tegra 设备不支持系统范围的原子操作。

#  1. Arithmetic Functions
##   1.1. atomicAdd()
```C++
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);
__half2 atomicAdd(__half2 *address, __half2 val);
__half atomicAdd(__half *address, __half val);
__nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val);
__nv_bfloat16 atomicAdd(__nv_bfloat16 *address, __nv_bfloat16 val);
```
读取位于全局或共享内存中地址 `address` 的 16 位、32 位或 64 位字 `old`，计算 `(old + val)`，并将结果存储回同一地址的内存中。这三个操作在一个原子事务中执行。该函数返回`old`。

`atomicAdd()` 的 32 位浮点版本仅受计算能力 2.x 及更高版本的设备支持。

`atomicAdd()` 的 64 位浮点版本仅受计算能力 6.x 及更高版本的设备支持。

`atomicAdd()` 的 32 位 `__half2` 浮点版本仅受计算能力 6.x 及更高版本的设备支持。 `__half2` 或 `__nv_bfloat162` 加法操作的原子性分别保证两个 `__half` 或 `__nv_bfloat16` 元素中的每一个；不保证整个 `__half2` 或 `__nv_bfloat162` 作为单个 32 位访问是原子的。

`atomicAdd()` 的 16 位 `__half` 浮点版本仅受计算能力 7.x 及更高版本的设备支持。

`atomicAdd()` 的 16 位 `__nv_bfloat16` 浮点版本仅受计算能力 8.x 及更高版本的设备支持。

####   1.2. atomicSub()
```C++
int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address,
                       unsigned int val);
```
读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `(old - val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

##   1.3. atomicExch()
```C++
int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address,
                        unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);
float atomicExch(float* address, float val);
```
读取位于全局或共享内存中地址address的 32 位或 64 位字 `old` 并将 `val` 存储回同一地址的内存中。 这两个操作在一个原子事务中执行。 该函数返回`old`。

##   1.4. atomicMin()
```C++
int atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMin(long long int* address,
                                long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `old` 和 `val` 的最小值，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicMin()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

##   1.5. atomicMax()
```C++
int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMax(long long int* address,
                                 long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `old` 和 `val` 的最大值，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicMax()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

##   1.6. atomicInc()
```C++
unsigned int atomicInc(unsigned int* address,
                       unsigned int val);
```

读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `((old >= val) ? 0 : (old+1))`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

##   1.7. atomicDec()
```C++
unsigned int atomicDec(unsigned int* address,
                       unsigned int val);
```
读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `(((old == 0) || (old > val)) ? val : (old-1) )`，并将结果存储回同一个地址的内存。 这三个操作在一个原子事务中执行。 该函数返回`old`。

##   1.8. atomicCAS()
```C++
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);
unsigned short int atomicCAS(unsigned short int *address, 
                             unsigned short int compare, 
                             unsigned short int val);
```
读取位于全局或共享内存中地址`address`的 16 位、32 位或 64 位字 `old`，计算 `(old == compare ? val : old)` ，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`（Compare And Swap）。

#   2. Bitwise Functions

##   2.1. atomicAnd()
```C++
int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old & val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicAnd()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

##   2.2. atomicOr()
```C++
int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old | val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicOr()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

##  2.3. atomicXor()
```C++
int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old ^ val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicXor()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。
