# CUDA中的异步数据拷贝

CUDA 11 引入了带有 `memcpy_async` API 的异步数据操作，以允许设备代码显式管理数据的异步复制。 `memcpy_async` 功能使 CUDA 内核能够将计算与数据传输重叠。

##  1. memcpy_async API接口
`memcpy_async` API 在 `cuda/barrier、cuda/pipeline` 和`cooperative_groups/memcpy_async.h` 头文件中提供。

`cuda::memcpy_async` API 与 `cuda::barrier` 和 `cuda::pipeline` 同步原语一起使用，而`cooperative_groups::memcpy_async` 使用 `coopertive_groups::wait` 进行同步。

这些 API 具有非常相似的语义：将对象从 `src` 复制到 `dst`，就好像由另一个线程执行一样，在完成复制后，可以通过 `cuda::pipeline、cuda::barrier` 或`cooperative_groups::wait` 进行同步。

[`libcudacxx`](https://nvidia.github.io/libcudacxx) API 文档和一些示例中提供了 `cuda::barrier` 和 `cuda::pipeline` 的 `cuda::memcpy_async` 重载的完整 API 文档。

`Cooperation_groups::memcpy_async` 的 API 文档在[文档的合作组部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)中提供。

使用 `cuda::barrier` 和 `cuda::pipeline` 的 `memcpy_async` API 需要 7.0 或更高的计算能力。在具有 8.0 或更高计算能力的设备上，从全局内存到共享内存的 `memcpy_async` 操作可以受益于硬件加速。

##  2. 拷贝和计算模式 - 利用Shared Memory逐步处理存储

CUDA 应用程序通常采用一种***copy and compute*** 模式：
* 从全局内存中获取数据，
* 将数据存储到共享内存中，
* 对共享内存数据执行计算，并可能将结果写回全局内存。
  
以下部分说明了如何在使用和不使用` memcpy_async` 功能的情况下表达此模式：
* [没有 memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#without_memcpy_async) 部分介绍了一个不与数据移动重叠计算并使用中间寄存器复制数据的示例。
* [使用 memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#with_memcpy_async) 部分改进了前面的示例，引入了`cooperation_groups::memcpy_async` 和 `cuda::memcpy_async` API 直接将数据从全局复制到共享内存，而不使用中间寄存器。
* 使用 `cuda::barrier` 的[异步数据拷贝](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memcpy_async_barrier)部分显示了带有协作组和屏障的 memcpy
* [单步异步数据拷贝](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#with-memcpy_async-pipeline-pattern-single)展示了利用单步`cuda::pipeline`的memcpy
* [多步异步数据拷贝](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#with-memcpy_async-pipeline-pattern-multi)展示了使用`cuda::pipeline`多步memcpy

##  3. 不使用 memcpy_async
如果没有 `memcpy_async`，复制和计算模式的复制阶段表示为 `shared[local_idx] = global[global_idx]`。 这种全局到共享内存的复制被扩展为从全局内存读取到寄存器，然后从寄存器写入共享内存。

当这种模式出现在迭代算法中时，每个线程块需要在 `shared[local_idx] = global[global_idx]` 分配之后进行同步，以确保在计算阶段开始之前对共享内存的所有写入都已完成。 线程块还需要在计算阶段之后再次同步，以防止在所有线程完成计算之前覆盖共享内存。 此模式在以下代码片段中进行了说明。

```C++
#include <cooperative_groups.h>
__device__ void compute(int* global_out, int const* shared_in) {
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory.
}

__global__ void without_memcpy_async(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // Compute the index of the current batch for this block in global memory:
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    size_t global_idx = block_batch_idx + threadIdx.x;
    shared[local_idx] = global_in[global_idx];

    block.sync(); // Wait for all copies to complete

    compute(global_out + block_batch_idx, shared); // Compute and write result to global memory

    block.sync(); // Wait for compute using shared memory to finish
  }
} 
```

##  4. 使用memcpy_async
使用 `memcpy_async`，从全局内存中分配共享内存
```C++
shared[local_idx] = global_in[global_idx];
```
替换为来自[合作组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)的异步复制操作

```C++
cooperative_groups::memcpy_async(group, shared, global_in + batch_idx, sizeof(int) * block.size());
```

`cooperation_groups::memcpy_async` API 将 `sizeof(int) * block.size()` 字节从 `global_in + batch_idx` 开始的全局内存复制到共享数据。 这个操作就像由另一个线程执行一样发生，在复制完成后，它与当前线程对`cooperative_groups::wait` 的调用同步。 在复制操作完成之前，修改全局数据或读取写入共享数据会引入数据竞争。

在具有 8.0 或更高计算能力的设备上，从全局内存到共享内存的 `memcpy_async` 传输可以受益于硬件加速，从而避免通过中间寄存器传输数据。

```C++
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__device__ void compute(int* global_out, int const* shared_in);

__global__ void with_memcpy_async(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    // Whole thread-group cooperatively copies whole batch to shared memory:
    cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size());

    cooperative_groups::wait(block); // Joins all threads, waits for all copies to complete

    compute(global_out + block_batch_idx, shared);

    block.sync();
  }
}}      
```

##  5. 使用 cuda::barrier异步拷贝内存

`cuda::memcpy_async` 的 `cuda::barrier` 重载允许使用屏障同步异步数据传输。 此重载执行复制操作，就好像由绑定到屏障的另一个线程执行：在创建时增加当前阶段的预期计数，并在完成复制操作时减少它，这样屏障的阶段只会前进, 当所有参与屏障的线程都已到达，并且绑定到屏障当前阶段的所有 memcpy_async 都已完成时。 以下示例使用block范围的屏障，所有块线程都参与其中，并将等待操作与屏障到达和等待交换，同时提供与前一个示例相同的功能：
```C++
#include <cooperative_groups.h>
#include <cuda/barrier>
__device__ void compute(int* global_out, int const* shared_in);

__global__ void with_barrier(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  // Create a synchronization object (C++20 barrier)
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
  if (block.thread_rank() == 0) {
    init(&barrier, block.size()); // Friend function initializes barrier
  }
  block.sync();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    cuda::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size(), barrier);

    barrier.arrive_and_wait(); // Waits for all copies to complete

    compute(global_out + block_batch_idx, shared);

    block.sync();
  }
}
```
##  6.  memcpy_async使用指南
对于计算能力 8.x，pipeline机制在同一 CUDA warp中的 CUDA 线程之间共享。 这种共享会导致成批的 memcpy_async 纠缠在warp中，这可能会在某些情况下影响性能。

本节重点介绍 warp-entanglement 对提交、等待和到达操作的影响。 有关各个操作的概述，请参阅[pipeline接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface)和[pipeline基元接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-primitives-interface)。

###  6.1. 对齐

在具有计算能力 8.0 的设备上，`cp.async` 系列指令允许将数据从全局异步复制到共享内存。 这些指令支持一次复制 4、8 和 16 个字节。 如果提供给 `memcpy_async` 的大小是 4、8 或 16 的倍数，并且传递给 `memcpy_async` 的两个指针都对齐到 4、8 或 16 对齐边界，则可以使用专门的异步内存操作来实现 `memcpy_async`。

此外，为了在使用 `memcpy_async` API 时获得最佳性能，需要为共享内存和全局内存对齐 128 字节。

对于指向对齐要求为 1 或 2 的类型值的指针，通常无法证明指针始终对齐到更高的对齐边界。 确定是否可以使用 `cp.async` 指令必须延迟到运行时。 执行这样的运行时对齐检查会增加代码大小并增加运行时开销。

`cuda::aligned_size_t<size_t Align>(size_t size)Shape `可用于证明传递给 `memcpy_async `的两个指针都与 `Align` 边界对齐，并且大小是 `Align` 的倍数，方法是将其作为参数传递，其中 `memcpy_async` API 需要一个 `Shape`：
```C++
cuda::memcpy_async(group, dst, src, cuda::aligned_size_t<16>(N * block.size()), pipeline);
```
如果验证不正确，则行为未定义。 

###  6.2. Trivially copyable

在具有计算能力 8.0 的设备上，`cp.async` 系列指令允许将数据从全局异步复制到共享内存。 如果传递给 `memcpy_async` 的指针类型不指向 `TriviallyCopyable` 类型，则需要调用每个输出元素的复制构造函数，并且这些指令不能用于加速 `memcpy_async`。

###  6.3. Warp Entanglement - Commit

`memcpy_async` 批处理的序列在 warp 中共享。 提交操作被合并，使得对于调用提交操作的所有聚合线程，序列增加一次。 如果warp完全收敛，则序列加1； 如果warp完全发散，则序列增加 32。

* 设 PB 为 warp-shared pipeline的实际批次序列. 
   
  `PB = {BP0, BP1, BP2, …, BPL}`

* 令 TB 为线程感知的批次序列，就好像该序列仅由该线程调用提交操作增加。

    `TB = {BT0, BT1, BT2, …, BTL}`

    `pipeline::producer_commit()` 返回值来自线程感知的批处理序列。

* 线程感知序列中的索引始终与实际warp共享序列中的相等或更大的索引对齐。 仅当从聚合线程调用所有提交操作时，序列才相等。

    `BTn ≡ BPm 其中 n <= m`

例如，当warp完全发散时：

* warp共享pipeline的实际顺序是：PB = {0, 1, 2, 3, ..., 31} (PL=31)。
* 该warp的每个线程的感知顺序将是：
  * `Thread 0: TB = {0} (TL=0)`
  * `Thread 1: TB = {0} (TL=0)`
  
    `…`
  * `Thread 31: TB = {0} (TL=0)`

###  6.4. Warp Entanglement - Wait
CUDA 线程调用 `pipeline_consumer_wait_prior<N>()` 或 `pipeline::consumer_wait()` 以等待感知序列 TB 中的批次完成。 注意 `pipeline::consumer_wait()` 等价于 `pipeline_consumer_wait_prior<N>()`，其中 `N = PL`。

`pipeline_consumer_wait_prior<N>()` 函数等待实际序列中的批次，至少达到并包括 `PL-N`。 由于 `TL <= PL`，等待批次达到并包括 `PL-N` 包括等待批次 `TL-N`。 因此，当 `TL < PL` 时，线程将无意中等待更多的、更新的批次。

在上面的极端完全发散的warp示例中，每个线程都可以等待所有 32 个批次。

###  6.5. Warp Entanglement - Arrive-On

`Warp-divergence` 影响到达 `on(bar)` 操作更新障碍的次数。 如果调用 warp 完全收敛，则屏障更新一次。 如果调用 warp 完全发散，则将 32 个单独的更新应用于屏障。

###  6.6. Keep Commit and Arrive-On Operations Converged

建议提交和到达调用由聚合线程进行：

* 通过保持线程的感知批次序列与实际序列对齐，不要过度等待，并且
* 以最小化对屏障对象的更新。


当这些操作之前的代码分支线程时，应该在调用提交或到达操作之前通过 `__syncwarp` 重新收敛warp。