# CUDA中动态Global Memory分配和操作


动态全局内存分配和操作仅受计算能力 2.x 及更高版本的设备支持。
```C++
__host__ __device__ void* malloc(size_t size);
__device__ void *__nv_aligned_device_malloc(size_t size, size_t align);
__host__ __device__  void free(void* ptr);
```
从全局内存中的固定大小的堆中动态分配和释放内存。
```C++
__host__ __device__ void* memcpy(void* dest, const void* src, size_t size);
```
从 `src` 指向的内存位置复制 `size` 个字节到 `dest` 指向的内存位置。

```C++
__host__ __device__ void* memset(void* ptr, int value, size_t size);
```
将 `ptr` 指向的内存块的 `size` 字节设置为 `value`（解释为无符号字符）。

CUDA 内核中的 `malloc()` 函数从设备堆中分配至少 `size` 个字节，并返回一个指向已分配内存的指针，如果没有足够的内存来满足请求，则返回 NULL。返回的指针保证与 16 字节边界对齐。

内核中的 CUDA `__nv_aligned_device_malloc()` 函数从设备堆中分配至少 `size` 个字节，并返回一个指向已分配内存的指针，如果内存不足以满足请求的大小或对齐，则返回 NULL。分配内存的地址将是 `align` 的倍数。 `align` 必须是 2 的非零幂。

CUDA 内核中的 `free()` 函数释放 `ptr` 指向的内存，该内存必须由先前对 `malloc()` 或 `__nv_aligned_device_malloc()` 的调用返回。如果 `ptr` 为 NULL，则忽略对 `free()` 的调用。使用相同的 `ptr` 重复调用 `free()` 具有未定义的行为。

给定 CUDA 线程通过 `malloc()` 或 `__nv_aligned_device_malloc()` 分配的内存在 CUDA 上下文的生命周期内保持分配状态，或者直到通过调用 `free()` 显式释放。它可以被任何其他 CUDA 线程使用，即使在随后的内核启动时也是如此。任何 CUDA 线程都可以释放由另一个线程分配的内存，但应注意确保不会多次释放同一指针。

##  1. Heap Memory Allocation
设备内存堆具有固定大小，必须在任何使用 `malloc()、__nv_aligned_device_malloc() 或 free()` 的程序加载到上下文之前指定该大小。 如果任何程序在没有明确指定堆大小的情况下使用 `malloc() 或 __nv_aligned_device_malloc()` ，则会分配 8 MB 的默认堆。

以下 API 函数获取和设置堆大小：
* `cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)`
* `cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)`

授予的堆大小至少为 `size` 个字节。 `cuCtxGetLimit() 和 cudaDeviceGetLimit()` 返回当前请求的堆大小。

当模块被加载到上下文中时，堆的实际内存分配发生，或者显式地通过 CUDA 驱动程序 API（参见[模块](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#module)），或者隐式地通过 CUDA 运行时 API（参见 [CUDA 运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime)）。 如果内存分配失败，模块加载会产生 `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` 错误。

一旦发生模块加载，堆大小就无法更改，并且不会根据需要动态调整大小。

除了通过主机端 CUDA API 调用（例如 `cudaMalloc()`）分配为设备堆保留的内存之外。

##  2. Interoperability with Host Memory API
通过设备 `malloc()` 或 `__nv_aligned_device_malloc()` 分配的内存不能使用运行时释放（即，通过从设备内存调用任何空闲内存函数）。

同样，通过运行时分配的内存（即，通过从设备内存调用任何内存分配函数）不能通过 `free()` 释放。

此外，在设备代码中调用 `malloc()` 或 `__nv_aligned_device_malloc()` 分配的内存不能用于任何运行时或驱动程序 API 调用（即 `cudaMemcpy`、`cudaMemset` 等）。

##  3. Examples
###  3.1. Per Thread Allocation

```C++
#include <stdlib.h>
#include <stdio.h>

__global__ void mallocTest()
{
    size_t size = 123;
    char* ptr = (char*)malloc(size);
    memset(ptr, 0, size);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}

int main()
{
    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
```
上面的代码将会输出:
```
Thread 0 got pointer: 00057020
Thread 1 got pointer: 0005708c
Thread 2 got pointer: 000570f8
Thread 3 got pointer: 00057164
Thread 4 got pointer: 000571d0
```
注意每个线程如何遇到 `malloc()` 和 `memset()` 命令，从而接收和初始化自己的分配。 （确切的指针值会有所不同：这些是说明性的。）

###  3.2. Per Thread Block Allocation
```C++
#include <stdlib.h>

__global__ void mallocTest()
{
    __shared__ int* data;

    // The first thread in the block does the allocation and then
    // shares the pointer with all other threads through shared memory,
    // so that access can easily be coalesced.
    // 64 bytes per thread are allocated.
    if (threadIdx.x == 0) {
        size_t size = blockDim.x * 64;
        data = (int*)malloc(size);
    }
    __syncthreads();

    // Check for failure
    if (data == NULL)
        return;

    // Threads index into the memory, ensuring coalescence
    int* ptr = data;
    for (int i = 0; i < 64; ++i)
        ptr[i * blockDim.x + threadIdx.x] = threadIdx.x;

    // Ensure all threads complete before freeing 
    __syncthreads();

    // Only one thread may free the memory!
    if (threadIdx.x == 0)
        free(data);
}

int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<10, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

###  3.3. Allocation Persisting Between Kernel Launches
```C++
#include <stdlib.h>
#include <stdio.h>

#define NUM_BLOCKS 20

__device__ int* dataptr[NUM_BLOCKS]; // Per-block pointer

__global__ void allocmem()
{
    // Only the first thread in the block does the allocation
    // since we want only one allocation per block.
    if (threadIdx.x == 0)
        dataptr[blockIdx.x] = (int*)malloc(blockDim.x * 4);
    __syncthreads();

    // Check for failure
    if (dataptr[blockIdx.x] == NULL)
        return;

    // Zero the data with all threads in parallel
    dataptr[blockIdx.x][threadIdx.x] = 0;
}

// Simple example: store thread ID into each element
__global__ void usemem()
{
    int* ptr = dataptr[blockIdx.x];
    if (ptr != NULL)
        ptr[threadIdx.x] += threadIdx.x;
}

// Print the content of the buffer before freeing it
__global__ void freemem()
{
    int* ptr = dataptr[blockIdx.x];
    if (ptr != NULL)
        printf("Block %d, Thread %d: final value = %d\n",
                      blockIdx.x, threadIdx.x, ptr[threadIdx.x]);

    // Only free from one thread!
    if (threadIdx.x == 0)
        free(ptr);
}

int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    // Allocate memory
    allocmem<<< NUM_BLOCKS, 10 >>>();

    // Use memory
    usemem<<< NUM_BLOCKS, 10 >>>();
    usemem<<< NUM_BLOCKS, 10 >>>();
    usemem<<< NUM_BLOCKS, 10 >>>();

    // Free memory
    freemem<<< NUM_BLOCKS, 10 >>>();

    cudaDeviceSynchronize();

    return 0;
}
```