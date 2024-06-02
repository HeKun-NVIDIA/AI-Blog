#! https://zhuanlan.zhihu.com/p/689580387
# 高效 CUDA 调试：将 NVIDIA Compute Sanitizer 与 NVIDIA 工具扩展结合使用并创建自定义工具

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/efficient-cuda-debugging-3-featured.png)


[NVIDIA Compute Sanitizer](https://developer.nvidia.com/nvidia-compute-sanitizer) 是一款功能强大的工具，可以节省您的时间和精力，同时提高 CUDA 应用程序的可靠性和性能。 在 CUDA 环境中调试代码既具有挑战性又耗时，尤其是在处理数千个线程时。 计算消毒剂可以提供帮助！

在本系列的第一篇文章《[高效 CUDA 调试：如何使用 NVIDIA Compute Sanitizer 寻找错误](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/)》中，我们讨论了如何开始使用一些 Compute Sanitizer 工具来在调试代码时检查内存泄漏和竞争条件。

在第二篇文章《[高效 CUDA 调试：使用 NVIDIA Compute Sanitizer 进行内存初始化和线程同步](https://developer.nvidia.com/blog/efficient-cuda-debugging-memory-initialization-and-thread-synchronization-with-nvidia-compute-sanitizer/)》中，我们探索了用于检查内存初始化和线程同步的工具。

在这篇文章中，我们重点介绍了 Compute Sanitizer 的一些其他功能，即它与 NVIDIA 工具扩展 (NVTX) 的集成，用于标记代码以方便更直接地使用 Compute Sanitizer。 我们还讨论了 Compute Sanitizer 本身的 API，以便创建更多用于调试 CUDA 应用程序的工具。

在这篇文章中，我们重点介绍了 Compute Sanitizer 的一些其他功能，即它与 NVIDIA 工具扩展 (NVTX) 的集成，用于标记代码以方便更直接地使用 Compute Sanitizer。 我们还讨论了 Compute Sanitizer 本身的 API，以便创建更多用于调试 CUDA 应用程序的工具。

### NVIDIA Compute Sanitizer
Compute Sanitizer 是一套工具，可以对代码的功能正确性执行不同类型的检查。 主要有四种工具：

* `memcheck`：内存访问错误和泄漏检测。
* `racecheck`：共享内存数据访问危险检测工具。
* `initcheck`：未初始化设备全局内存访问检测工具。
* `synccheck`：线程同步危险检测。

除了这些工具之外，NVIDIA Compute Sanitizer 还具有更多功能：

* 一个 API，用于创建针对 CUDA 应用程序的清理和跟踪工具
* 与 NVIDIA 工具扩展 (NVTX) 集成
* Coredump 支持与 CUDA-GDB 一起使用
* 用于管理工具输出的抑制功能

## 将 Compute Sanitizer 与 NVTX 结合使用
NVTX 是一个基于 C 的 API，用于注释程序中的代码范围、事件和资源。 此注释允许在应用程序运行时收集更多信息，这可用于在分析和分析代码时改进数据表示。 Compute Sanitizer 和 NVTX 之间的集成使您能够使用 NVTX 来注释代码，以帮助 Compute Sanitizer 捕获错误。

有关 NVTX 注释的更多信息，请参阅以下帖子：

* C/C++ 和 NVTX：[CUDA 专业提示：使用 NVTX 生成自定义应用程序配置文件时间线](https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/)
Python 和 NVTX：[NVIDIA 工具扩展 API：用于分析 Python 和 C/C++ 代码的注释工具](https://developer.nvidia.com/blog/nvidia-tools-extension-api-nvtx-annotation-tool-for-profiling-code-in-python-and-c-c/)
Fortran 和 NVTX：[使用 NVTX 自定义 CUDA Fortran 分析](https://developer.nvidia.com/blog/customize-cuda-fortran-profiling-nvtx/)

NVTX 内存 API 使 CUDA 程序能够向 Compute Sanitizer 通知内存限制，例如内存池管理或权限限制以及内存标签。

### 内存池管理
NVTX 与 Compute Sanitizer 集成的第一个示例来自 NVTX 内存 API 的子分配部分。

使用 API，您可以将内存分配注释为内存池。 Compute Sanitizer 知道这些池，并且可以检测实际使用特定分配的哪些部分。 然后，如果在代码执行期间访问内存池的任何未注册部分，则通过 `Compute Sanitizer memcheck` 工具检测到这些访问。

这是基本内存池的示例 `mempool_example.cu。`
```cpp
#include <stdio.h>
 
__global__ void populateMemory(int* chunk) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  chunk[i] = i;
}
 
int main(int argc, char **argv) {
  int poolSize   = 4096 * sizeof(int);
  int numThreads = 63;
  // int bucketSize = numThreads * sizeof(int); // You need this later ...
 
  void *pool;
  cudaMallocManaged(&pool, poolSize); // Create your memory pool
 
  // Assign part of the memory pool to the bucket
  auto bucket = (int *)pool + 16; // Address of bucket is 16 bytes into the pool
 
  // Set values in bucket
  populateMemory<<<1, numThreads>>>(bucket);
  cudaDeviceSynchronize();
  printf("After populateMemory 1: bucket 0, 1 ..  62: %d %d .. %d\n", bucket[0], bucket[1], bucket[numThreads-1]);
 
  // Set some more values in bucket
  populateMemory<<<1, numThreads + 1>>>(bucket);
  cudaDeviceSynchronize();
  printf("After populateMemory 2: bucket 0, 1 ..  63: %d %d .. %d\n", bucket[0], bucket[1], bucket[numThreads]);
 
  cudaFree(pool);
  exit(0);
}

```
在代码示例中，您创建了一个大小为 4096 个整数的内存池（称为池！）。 然后，您分配该池的一部分（由变量存储桶标记），从距池开头 16 字节的地址开始。

您打算将存储桶设置为具有 `numThreads` 个元素，在本例中为 `63，由变量bucketSize` 确定。 然后，您可以使用 GPU 内核 `populateMemory` 用一些值填充存储桶。 块数设置为 1，线程数设置为 numThreads，这意味着 `populateMemory` 执行 `1×63` 次，按预期设置存储桶中的每个值。

但是，您随后尝试通过 `populateMemory` 内核再次填充存储桶。 这次，您将线程数设置为 `numThreads+1 (64)`。 您的意图是让存储桶具有 `63` 个值，但现在您尝试分配 `64` 个值。但是，这不会导致错误。 例如，您没有访问任何越界内存，因为存储桶所属的实际内存池足够大，可以容纳额外的元素。

编译它并通过 `memcheck` 运行它，以确认这个潜在的错误没有被发现。 我们在 NVIDIA V100 GPU 上运行，因此我们将 GPU 架构设置为 `sm_70`。 您可能需要更改此设置，具体取决于您运行的内容。

```bash
$ nvcc -o mempool.exe mempool_example.cu -arch=sm_70
$ ./mempool.exe
After populateMemory 1: bucket 0, 1 ..  62: 0 1 .. 62
After populateMemory 2: bucket 0, 1 ..  63: 0 1 .. 63
 
$compute-sanitizer --tool memcheck ./mempool.exe
========= COMPUTE-SANITIZER
After populateMemory 1: bucket 0, 1 ..  62: 0 1 .. 62
After populateMemory 2: bucket 0, 1 ..  63: 0 1 .. 63
========= ERROR SUMMARY: 0 errors
```
这就是 NVTX API 可以提供帮助的地方。 您可以使用 NVTX 的内存堆寄存器函数 nvtxMemHeapRegister 注册任何 cudaMalloc 内存分配。 这将内存注册为一个堆，代表可以进一步细分为多个区域的内存范围。 您可以通过以下过程在此代码中执行此操作。

首先，完成将 NVTX 与 Compute Sanitizer 结合使用所需的四个步骤。

对于 C 和 C++，NVTX 是一个仅包含头文件的库，没有依赖项，因此您必须获取 NVTX 头文件才能包含。 通常，这些会随您首选的 CUDA 下载一起提供，例如[工具包](https://developer.nvidia.com/cuda-toolkit)或 [HPC SDK](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nvhpc)。 但是，NVTX 内存 API 相对较新，因此现在可以从 [/NVIDIA/NVTX](https://github.com/NVIDIA/NVTX.git) GitHub 存储库获取它。 将来，它将作为工具包的一部分包含在内。

特别是，`nvToolsExtMem.h` 标头尚无法通过其他方法获得，因此在克隆 NVTX GitHub 分支 dev-mem-api 后检查它是否存在：

```bash
$ git clone --branch dev-mem-api https://github.com/NVIDIA/NVTX.git
…
$ ls NVTX/c/include/nvtx3/
nvToolsExtCuda.h    nvToolsExt.h           nvToolsExtMem.h     nvToolsExtSync.h  nvtxDetail
nvToolsExtCudaRt.h  nvToolsExtMemCudaRt.h  nvToolsExtOpenCL.h  nvtx3.hpp         nvtxExtDetail
```
现在您可以在源代码的开头包含 NVTX 和 NVTX API 头文件：
```bash
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtMem.h>
```
Compute Sanitizer 要求在任何 NVTX 调用之前初始化 CUDA 运行时。 无论如何，这可能会发生在您的代码中，具体取决于您开始使用 NVTX 的位置，但您可以使用 `cudaFree` 强制执行它，例如：

```cpp
// Forces CUDA runtime initialization.
cudaFree(0);

```
最后，创建一个 NVTX 域。 这些是调用 API 所必需的。 目前，这些域没有特定的功能，但将用于未来的 Compute Sanitizer 版本。
```cpp
// Create the NVTX domain
auto mynvtxDomain = nvtxDomainCreateA("my-domain");

```

好的，这就是完成的先决步骤。 现在，使用 NVTX 将池分配注册为内存池或堆：
```cpp
nvtxMemVirtualRangeDesc_t myPoolRangeDesc = {}; // Descriptor for the
                                                // range memory pool
myPoolRangeDesc.size = poolSize; // Size of the range memory pool
myPoolRangeDesc.ptr  = pool;     // Pointer to the pool itself
 
nvtxMemHeapDesc_t myHeapDesc = {}; // Descriptor for the heap
 
myHeapDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
myHeapDesc.structSize = sizeof(nvtxMemHeapDesc_t);
myHeapDesc.usage = NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR;
myHeapDesc.type = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
myHeapDesc.typeSpecificDescSize = sizeof(nvtxMemVirtualRangeDesc_t);
myHeapDesc.typeSpecificDesc = &myPoolRangeDesc;
 
auto mynvtxPool = nvtxMemHeapRegister(mynvtxDomain, &myHeapDesc);

```
这些步骤已注册该池并将其分配给变量 `mynvtxPool。` 要在前面的示例中使用它，您现在必须在池中创建一个子分配来代表存储桶。 语法与分配池本身的方式没有什么不同，但这次使用区域描述符而不是堆描述符：

```cpp
nvtxMemVirtualRangeDesc_t mySubRangeDesc = {}; // Descriptor for the range
mySubRangeDesc.size = bucketSize; // Size of your suballocation (in bytes)
mySubRangeDesc.ptr  = bucket;     // Pointer to the suballocation
 
nvtxMemRegionsRegisterBatch_t myRegionsDesc = {};
myRegionsDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
myRegionsDesc.structSize  = sizeof(nvtxMemRegionsRegisterBatch_t);
myRegionsDesc.regionType  = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
myRegionsDesc.heap = mynvtxPool; // The heap you registered earlier
myRegionsDesc.regionCount = 1;
myRegionsDesc.regionDescElementSize = sizeof(nvtxMemVirtualRangeDesc_t);
myRegionsDesc.regionDescElements = &mySubRangeDesc;
 
nvtxMemRegionsRegister(mynvtxDomain, &myRegionsDesc);

```
这既是您的内存池，又是现在已向 NVTX 注册的子分配存储桶。 这意味着 Compute Sanitizer 可以将它们的属性作为其检查的一部分。 现在，看看它是否检测到将存储桶填充到其预期范围之外的错误尝试。

以下是现在具有 NVTX 注册的基本内存池的完整代码示例，包括 NVTX 注册，`mempool_nvtx_example.cu`。

```cpp
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtMem.h>
 
#include <stdio.h>
 
__global__ void populateMemory(int* chunk) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  chunk[i] = i;
}
 
int main(int argc, char **argv) {
  int poolSize   = 4096 * sizeof(int);
  int numThreads = 63;
  int bucketSize = numThreads * sizeof(int);
 
  // Forces CUDA runtime initialization.
  cudaFree(0);
 
  // Create the NVTX domain
  auto mynvtxDomain = nvtxDomainCreateA("my-domain");
 
 
  void *pool;
  cudaMallocManaged(&pool, poolSize); // Create your memory pool
 
  // Register the pool with NVTX
  nvtxMemVirtualRangeDesc_t myPoolRangeDesc = {}; // Descriptor for the
                                                  // range memory pool
  myPoolRangeDesc.size = poolSize; // Size of the range memory pool
  myPoolRangeDesc.ptr  = pool;     // Pointer to the pool itself
 
  nvtxMemHeapDesc_t myHeapDesc = {}; // Descriptor for the heap
 
  myHeapDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
  myHeapDesc.structSize = sizeof(nvtxMemHeapDesc_t);
  myHeapDesc.usage = NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR;
  myHeapDesc.type = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
  myHeapDesc.typeSpecificDescSize = sizeof(nvtxMemVirtualRangeDesc_t);
  myHeapDesc.typeSpecificDesc = &myPoolRangeDesc;
 
  auto mynvtxPool = nvtxMemHeapRegister(mynvtxDomain, &myHeapDesc);
 
  // Assign part of the memory pool to the bucket
  auto bucket = (int *)pool + 16; // Address of bucket is 16 bytes into the pool
 
  // Register bucket as a suballocated region in NVTX
  nvtxMemVirtualRangeDesc_t mySubRangeDesc = {}; // Descriptor for the range
  mySubRangeDesc.size = bucketSize; // Size of your suballocation (in bytes)
  mySubRangeDesc.ptr  = bucket;     // Pointer to the suballocation
 
  nvtxMemRegionsRegisterBatch_t myRegionsDesc = {};
  myRegionsDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
  myRegionsDesc.structSize  = sizeof(nvtxMemRegionsRegisterBatch_t);
  myRegionsDesc.regionType  = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
  myRegionsDesc.heap = mynvtxPool; // The heap you registered earlier
  myRegionsDesc.regionCount = 1;
  myRegionsDesc.regionDescElementSize = sizeof(nvtxMemVirtualRangeDesc_t);
  myRegionsDesc.regionDescElements = &mySubRangeDesc;
 
  nvtxMemRegionsRegister(mynvtxDomain, &myRegionsDesc);
 
  // Set values in bucket
  populateMemory<<<1, numThreads>>>(bucket);
  cudaDeviceSynchronize();
  printf("After populateMemory 1: bucket 0, 1 ..  62: %d %d .. %d\n", bucket[0], bucket[1], bucket[numThreads-1]);
 
  // Set some more values in bucket
  populateMemory<<<1, numThreads + 1>>>(bucket);
  cudaDeviceSynchronize();
  printf("After populateMemory 2: bucket 0, 1 ..  63: %d %d .. %d\n", bucket[0], bucket[1], bucket[numThreads]);
 
  cudaFree(pool);
  exit(0);
}

```
编译它并再次通过 Compute Sanitizer 运行它。 编译步骤中的 include 语句应指向您安装 NVTX 标头的位置。

```cpp
$ nvcc -I ./NVTX/c/include -o mempool_nvtx.exe mempool_nvtx_example.cu -arch=sm_70
$ compute-sanitizer --tool memcheck --destroy-on-device-error=kernel ./mempool_nvtx.exe
========= COMPUTE-SANITIZER
After populateMemory 1: bucket 0, 1 ..  62: 0 1 .. 62
========= Invalid __global__ write of size 4 bytes
=========     at populateMemory(int *)+0x70
=========     by thread (63,0,0) in block (0,0,0)
=========     Address 0x7f2a9800013c is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x7f2a98000040 of size 252 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
. . .
=========
After populateMemory 2: bucket 0, 1 ..  63: 0 1 .. 0
========= ERROR SUMMARY: 1 error

```

Compute Sanitizer 确实捕获了写入指定数组末尾的元素的尝试：
```cpp
Invalid __global__ write of size 4 bytes

```
现在，如果您有部分内存池需要调整大小甚至销毁怎么办？ NVTX 内存 API 还提供了类似的方法来对其 NVTX 注册执行此操作。

要调整大小，请返回之前的池和存储桶示例。 如果您想要将存储桶的大小从 63 个元素调整为 64 个元素，请使用以下代码修改前面的代码示例，也会调整您的 NVTX 注册存储桶的大小以反映这一点：

```cpp
// Resizing the sub-allocation within the memory pool
 // You reuse mySubRangeDesc from earlier
 mySubRangeDesc.size = bucketSize + 4; // You want one extra int (4B) element
 mySubRangeDesc.ptr  = bucket;
 
 nvtxMemRegionsResizeBatch_t myNewRegionsDesc = {};
 myNewRegionsDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
 myNewRegionsDesc.structSize = sizeof(mySubRangeDesc);
 myNewRegionsDesc.regionType = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
 myNewRegionsDesc.regionDescCount = 1;
 myNewRegionsDesc.regionDescElementSize = sizeof(mySubRangeDesc);
 myNewRegionsDesc.regionDescElements = &mySubRangeDesc;
 
 nvtxMemRegionsResize(mynvtxDomain, &myNewRegionsDesc);

```

正如您所看到的，它与子分配的初始声明类似，但最后使用了函数 nvtxMemRegionsResize。

仔细检查 Compute Sanitizer 是否对现在为调整大小的存储桶分配值的合法尝试感到满意。 在示例代码中两次调用populateMemory之间添加调整大小注册代码，然后编译运行。
```cpp
$ nvcc -I./NVTX/c/include -o mempool_resize.exe mempool_resize_example.cu -arch=sm_70
$ compute-sanitizer --tool memcheck --destroy-on-device-error=kernel ./mempool_resize.exe
========= COMPUTE-SANITIZER
After populateMemory 1: bucket 0, 1 ..  62: 0 1 .. 62
After populateMemory 2: bucket 0, 1 ..  63: 0 1 .. 63
========= ERROR SUMMARY: 0 errors

```
希望您能看到类似此示例的内容。 当您调整了子分配注册的大小后，现在不再有关于尝试访问添加到末尾的新元素的投诉。

需要相当多的代码行来注册和调整池和子分配的大小，以便将它们与 NVTX 和 Compute Sanitizer 一起使用。 对于更复杂的代码来说，这可能会变得很麻烦，因此将这些步骤封装到一个单独的类中可能会很有用。 很方便，/NVIDIA/compute-sanitizer-samples GitHub 存储库中提供了这种方法的一个很好的示例，因此这是您自己的代码的一个很好的起点。

还有两个 NVTX API 需要提及：

* 命名 API：使区域或子分配具有与其关联的 ASCII 名称。 然后，它可以用于在错误报告中通过名称来引用分配，目前支持泄漏和未使用的内存报告。
* 权限 API：允许将分配访问权限限制为只读或原子等。


## Compute Sanitizer API 用于创建您自己的工具
Compute Sanitizer 附带 API，使您能够创建自己的清理和跟踪工具来定位 CUDA 应用程序。 它是一组函数，可用于与 Compute Sanitizer 交互以进行控制和配置、启用或禁用其功能以及访问其结果。

该 API 还为您提供了一种将 Compute Sanitizer 集成到您的开发工作流程中的便捷方法，因为它可以轻松集成到现有的 CUDA 应用程序中。 借助 Compute Sanitizer API，您可以直接利用强大的调试功能并提高 CUDA 应用程序的可靠性和性能。

它由以下子 API 组成：

* 回调：使您能够在用户代码中注册回调，其中回调可以与相关 CUDA 函数或事件组关联，例如 memcpy 操作或驱动程序函数。 然后订阅者可以使用这些回调，例如用于事件跟踪。
* 修补：允许加载修补函数并将其插入到 GPU 上执行的设备代码中。 然后，它们可以用作检测点，这意味着每当执行修补事件时都会执行修补函数，例如，设置回调，例如进行内存访问的设备代码。
* 内存：提供标准 CUDA 内存 API 的替换函数。 可以从 Compute Sanitizer 回调中安全地调用替换，例如使用替换 sanitizerAlloc 而不是 cudaMalloc。

结合起来，这些 API 使您能够将 Compute Sanitizer 功能合并到您自己的工具中。

有关更多信息和一些示例代码，请参阅[ NVIDIA Compute Sanitizer API 指南](https://docs.nvidia.com/compute-sanitizer/SanitizerApiGuide/index.html)。


























