# 使用CUDA_Pipeline异步数据拷贝


CUDA 提供 `cuda::pipeline` 同步对象来管理异步数据移动并将其与计算重叠。

`cuda::pipeline` 的 API 文档在 [libcudacxx API](https://nvidia.github.io/libcudacxx) 中提供。 流水线对象是一个具有头尾的双端 N 阶段队列，用于按照先进先出 (FIFO) 的顺序处理工作。 管道对象具有以下成员函数来管理管道的各个阶段。

|Pipeline Class Member Function	|Description|
|----|----|
|`producer_acquire`|	Acquires an available stage in the pipeline internal queue.|
|`producer_commit`	|Commits the asynchronous operations issued after the producer_acquire call on the currently acquired stage of the pipeline.|
|`consumer_wait`|	Wait for completion of all asynchronous operations on the oldest stage of the pipeline.|
|`consumer_release`|	Release the oldest stage of the pipeline to the pipeline object for reuse. The released stage can be then acquired by the producer.|

##  1. 使用`cuda::pipeline`进行单个阶段的异步拷贝

在前面的示例中，我们展示了如何使用[`cooperative_groups`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#collectives-cg-wait)和 [`cuda::barrier`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier) 进行异步数据传输。 在本节中，我们将使用带有单个阶段的 `cuda::pipeline` API 来调度异步拷贝。 稍后我们将扩展此示例以显示多阶段重叠计算和复制。

```C++
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
        
__device__ void compute(int* global_out, int const* shared_in);
__global__ void with_single_stage(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 1; // Pipeline with one stage
    // One batch must fit in shared memory:
    extern __shared__ int shared[];  // block.size() * sizeof(int) bytes
    
    // Allocate shared storage for a two-stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t batch = 0; batch < batch_sz; ++batch) {
        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared, global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations committed to the
        // previous `compute` stage to complete:
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }
}
```

 ## 2. 使用`cuda::pipeline`多个阶段的异步拷贝

在前面带有`cooperative_groups::wait` 和`cuda::barrier` 的示例中，内核线程立即等待数据传输到共享内存完成。 这避免了数据从全局内存传输到寄存器，但不会通过重叠计算隐藏 `memcpy_async` 操作的延迟。

为此，我们在以下示例中使用 CUDA [***pipeline***](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface) 功能。 它提供了一种管理 `memcpy_async` 批处理序列的机制，使 CUDA 内核能够将内存传输与计算重叠。 以下示例实现了一个将数据传输与计算重叠的两级管道。 它：

* 初始化管道共享状态（更多下文）
* 通过为第一批调度 `memcpy_async` 来启动管道。
* 循环所有批次：它为下一个批次安排 `memcpy_async`，在完成上一个批次的 `memcpy_async` 时阻塞所有线程，然后将上一个批次的计算与下一个批次的内存的异步副本重叠。
* 最后，它通过对最后一批执行计算来排空管道。

请注意，为了与 `cuda::pipeline` 的互操作性，此处使用来自 `cuda/pipeline` 头文件的 `cuda::memcpy_async`。
```C++
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

__device__ void compute(int* global_out, int const* shared_in);
__global__ void with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 2; // Pipeline with two stages
    // Two batches must fit in shared memory:
    extern __shared__ int shared[];  // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count] = { 0, block.size() }; // Offsets to each batch

    // Allocate shared storage for a two-stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return block.group_index().x * block.size() + grid.size() * batch;
    };

    // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for the block:
    if (batch_sz == 0) return;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, shared + shared_offset[0], global_in + block_batch(0), sizeof(int) * block.size(), pipeline);
    pipeline.producer_commit();

    // Pipelined copy/compute:
    for (size_t batch = 1; batch < batch_sz; ++batch) {
        // Stage indices for the compute and copy stages:
        size_t compute_stage_idx = (batch - 1) % 2;
        size_t copy_stage_idx = batch % 2;

        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared + shared_offset[copy_stage_idx], global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations commited to the
        // previous `compute` stage to complete:
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared + shared_offset[compute_stage_idx]);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }

    // Compute the data fetch by the last iteration
    pipeline.consumer_wait();
    compute(global_out + block_batch(batch_sz-1), shared + shared_offset[(batch_sz - 1) % 2]);
    pipeline.consumer_release();
}
```

[***pipeline***](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface) 对象是一个带有头尾的双端队列，用于按照先进先出 (FIFO) 的顺序处理工作。 生产者线程将工作提交到管道的头部，而消费者线程从管道的尾部提取工作。 在上面的示例中，所有线程都是生产者和消费者线程。 线程首先提交 `memcpy_async` 操作以获取下一批，同时等待上一批 `memcpy_async` 操作完成。
* 将工作提交到pipeline阶段包括：
    * 使用 `pipeline.producer_acquire()` 从一组生产者线程中集体获取pipeline头。
    * 将 `memcpy_async` 操作提交到pipeline头。
    * 使用 `pipeline.producer_commit()` 共同提交（推进）pipeline头。
* 使用先前提交的阶段包括：
    * 共同等待阶段完成，例如，使用 pipeline.consumer_wait() 等待尾部（最旧）阶段。
    * 使用 `pipeline.consumer_release()` 集体释放阶段。

`cuda::pipeline_shared_state<scope, count> `封装了允许管道处理多达 `count` 个并发阶段的有限资源。 如果所有资源都在使用中，则 `pipeline.producer_acquire()` 会阻塞生产者线程，直到消费者线程释放下一个管道阶段的资源。
通过将循环的 `prolog` 和 `epilog` 与循环本身合并，可以以更简洁的方式编写此示例，如下所示：
```C++
template <size_t stages_count = 2 /* Pipeline with stages_count stages */>
__global__ void with_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * block.size();

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    // compute_batch: next batch to process
    // fetch_batch:  next batch to fetch from global memory
    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            size_t block_batch_idx = block_batch(batch_idx);
            cuda::memcpy_async(block, shared + shared_offset[shared_idx], global_in + block_batch_idx, sizeof(int) * block.size(), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}
```
上面使用的 `pipeline<thread_scope_block>` 原语非常灵活，并且支持我们上面的示例未使用的两个特性：块中的任意线程子集都可以参与管道，并且从参与的线程中，任何子集都可以成为生产者 ，消费者，或两者兼而有之。 在以下示例中，具有“偶数”线程等级的线程是生产者，而其他线程是消费者：
```C++
__device__ void compute(int* global_out, int shared_in); 

template <size_t stages_count = 2>
__global__ void with_specialized_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // In this example, threads with "even" thread rank are producers, while threads with "odd" thread rank are consumers:
    const cuda::pipeline_role thread_role 
      = block.thread_rank() % 2 == 0? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;

    // Each thread block only has half of its threads as producers:
    auto producer_threads = block.size() / 2;

    // Map adjacent even and odd threads to the same id:
    const int thread_idx = block.thread_rank() / 2;

    auto elements_per_batch = size / batch_sz;
    auto elements_per_batch_per_block = elements_per_batch / grid.group_dim().x;

    extern __shared__ int shared[]; // stages_count * elements_per_batch_per_block * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * elements_per_batch_per_block;

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    cuda::pipeline pipeline = cuda::make_pipeline(block, &shared_state, thread_role);

    // Each thread block processes `batch_sz` batches.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return elements_per_batch * batch + elements_per_batch_per_block * blockIdx.x;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            if (thread_role == cuda::pipeline_role::producer) {
                // Only the producer threads schedule asynchronous memcpys:
                pipeline.producer_acquire();
                size_t shared_idx = fetch_batch % stages_count;
                size_t batch_idx = fetch_batch;
                size_t global_batch_idx = block_batch(batch_idx) + thread_idx;
                size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
                cuda::memcpy_async(shared + shared_batch_idx, global_in + global_batch_idx, sizeof(int), pipeline);
                pipeline.producer_commit();
            }
        }
        if (thread_role == cuda::pipeline_role::consumer) {
            // Only the consumer threads compute:
            pipeline.consumer_wait();
            size_t shared_idx = compute_batch % stages_count;
            size_t global_batch_idx = block_batch(compute_batch) + thread_idx;
            size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
            compute(global_out + global_batch_idx, *(shared + shared_batch_idx));
            pipeline.consumer_release();
        }
    }
} 
```

管道执行了一些优化，例如，当所有线程既是生产者又是消费者时，但总的来说，支持所有这些特性的成本不能完全消除。 例如，流水线在共享内存中存储并使用一组屏障进行同步，如果块中的所有线程都参与流水线，这并不是真正必要的。

对于块中的所有线程都参与管道的特殊情况，我们可以通过使用`pipeline<thread_scope_thread>` 结合 `__syncthreads()` 做得比`pipeline<thread_scope_block>` 更好：
```C++
template<size_t stages_count>
__global__ void with_staging_scope_thread(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    auto thread = cooperative_groups::this_thread();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * block.size();

    // No pipeline::shared_state needed
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            // Each thread fetches its own data:
            size_t thread_batch_idx = block_batch(batch_idx) + threadIdx.x;
            // The copy is performed by a single `thread` and the size of the batch is now that of a single element:
            cuda::memcpy_async(thread, shared + shared_offset[shared_idx] + threadIdx.x, global_in + thread_batch_idx, sizeof(int), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync(); // __syncthreads: All memcpy_async of all threads in the block for this stage have completed here
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}
```
如果计算操作只读取与当前线程在同一 warp 中的其他线程写入的共享内存，则 `__syncwarp()` 就足够了。

##  3. Pipeline 接口
[libcudacxx](https://nvidia.github.io/libcudacxx) API 文档中提供了 `cuda::memcpy_async` 的完整 API 文档以及一些示例。

`pipeline`接口需要

* 至少 CUDA 11.0，
* 至少与 ISO C++ 2011 兼容，例如，使用 -std=c++11 编译，
* `#include <cuda/pipeline>`。
对于类似 C 的接口，在不兼容 ISO C++ 2011 的情况下进行编译时，请参阅 [Pipeline Primitives Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-primitives-interface)。

##  4. Pipeline 原语接口
`pipeline`原语是用于 `memcpy_async` 功能的类 C 接口。 通过包含 <cuda_pipeline.h> 头文件，可以使用`pipeline`原语接口。 在不兼容 ISO C++ 2011 的情况下进行编译时，请包含 `<cuda_pipeline_primitives.h>` 头文件。

##  4.1. memcpy_async 原语

```C++
void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                             const void* __restrict__ src_global,
                             size_t size_and_align,
                             size_t zfill=0);
```
* 请求提交以下操作以进行异步评估：
```C++
  size_t i = 0;
  for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_shared)[i]; /* copy */
  for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* zero-fill */
```
* 需要:
    * `dst_shared` 必须是指向 `memcpy_async` 的共享内存目标的指针。
    * `src_global` 必须是指向 `memcpy_async` 的全局内存源的指针。
    * `size_and_align` 必须为 4、8 或 16。
    * `zfill <= size_and_align`.
    * `size_and_align` 必须是 `dst_shared` 和 `src_global` 的对齐方式。

* 任何线程在等待 `memcpy_async` 操作完成之前修改源内存或观察目标内存都是一种竞争条件。 在提交 `memcpy_async` 操作和等待其完成之间，以下任何操作都会引入竞争条件：
    * 从 `dst_shared` 加载。
    * 存储到 `dst_shared` 或 `src_global。`
    * 对 `dst_shared` 或 `src_global` 应用原子更新。

###  4.2. Commit 原语
```C++
void __pipeline_commit();
```
* 将提交的 `memcpy_async` 作为当前批次提交到管道。

###  4.3. Wait 原语
```C++
void __pipeline_wait_prior(size_t N);
```
* 令 `{0, 1, 2, ..., L}` 为与给定线程调用 `__pipeline_commit()` 相关联的索引序列。
* 等待批次完成，至少包括 `L-N`。

###  4.4. Arrive On Barrier 原语
```C++
void __pipeline_arrive_on(__mbarrier_t* bar);
```
* `bar` 指向共享内存中的屏障。
* 将屏障到达计数加一，当在此调用之前排序的所有 `memcpy_async` 操作已完成时，到达计数减一，因此对到达计数的净影响为零。 用户有责任确保到达计数的增量不超过 `__mbarrier_maximum_count()`。