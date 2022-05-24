# 使用Grid-Stride循环编写灵活的内核

![](GPU-Pro-Tip.png)

CUDA 编程中最常见的任务之一是使用内核并行化循环。 举个例子，让我们用我们的老朋友 SAXPY。 这是使用 for 循环的基本顺序实现。 为了有效地并行化，我们需要启动足够多的线程来充分利用 GPU。

```C++
void saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}
```
常见的 CUDA 指导是为每个数据元素启动一个线程，这意味着要并行化上述 SAXPY 循环，我们编写一个内核，假设我们有足够的线程来覆盖数组大小。

```C++
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        y[i] = a * x[i] + y[i];
}
```
我将这种风格的内核称为单片内核，因为它假设一个大型线程网格可以一次性处理整个数组。 您可以使用以下代码启动 saxpy 内核来处理一百万个元素。

```C++
// Perform SAXPY on 1M elements
saxpy<<<4096,256>>>(1<<20, 2.0, x, y);
```

我建议使用网格步长循环，而不是在并行计算时完全消除循环，如下面的内核。

```C++
__global__
void saxpy(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}
```
这个内核不是假设线程网格足够大以覆盖整个数据数组，而是一次循环一个网格大小的数据数组。

请注意，循环的步幅是 `blockDim.x * gridDim.x`，它是网格中的线程总数。 因此，如果网格中有 `1280` 个线程，线程 0 将计算元素 `0、1280、2560` 等。这就是我称之为网格步长循环的原因。 通过使用步长等于网格大小的循环，我们确保了 warp 内的所有寻址都是单位步长，因此我们获得了最大的内存合并，就像在单片版本中一样。

当使用足够大的网格启动以覆盖循环的所有迭代时，网格步长循环应该具有与单片内核中的 if 语句基本相同的指令开销，因为只有在循环条件计算为真时才会计算循环增量。

使用网格步长循环有几个好处。

1. 可扩展性和线程重用。 通过使用循环，您可以支持任何问题大小，即使它超过了您的 CUDA 设备支持的最大网格大小。 此外，您可以限制用于调整性能的块数。 例如，启动设备上多处理器数量的倍数的块通常很有用，以平衡利用率。 例如，我们可能会像这样启动内核的循环版本。

    当您限制网格中的块数时，线程被重用于多个计算。 线程重用分摊线程创建和销毁成本以及内核在循环之前或之后可能执行的任何其他处理（例如线程私有或共享数据初始化）。
```C++
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
// Perform SAXPY on 1M elements
saxpy<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
```



2. 调试。 通过使用循环而不是单片内核，您可以通过使用一个线程启动一个块来轻松切换到串行处理。   
    这使得更容易模拟串行主机实现来验证结果，并且可以通过序列化打印顺序使 `printf` 调试更容易。 序列化计算还允许您消除因运行顺序变化而导致的数值变化，从而帮助您在调整并行版本之前验证您的数值是否正确。
```C++
saxpy<<<1,1>>>(1<<20, 2.0, x, y);
```

3. 可移植性和可读性。`grid-stride`循环代码比单片内核代码更像原始的顺序循环代码，使其对其他用户更清晰。 事实上，我们可以很容易地编写一个内核版本，它可以作为 GPU 上的并行 CUDA 内核或作为 CPU 上的顺序循环编译和运行。 [Hemi](https://developer.nvidia.com/blog/parallelforall/simple-portable-parallel-c-hemi-2/) 库提供了一个 `grid_stride_range()` 帮助器，它使用基于 C++11 范围的 for 循环使这个变得微不足道。

    我们可以使用此代码启动内核，它会在为 CUDA 编译时生成内核启动，或者在为 CPU 编译时生成函数调用。

```C++
HEMI_LAUNCHABLE
void saxpy(int n, float a, float *x, float *y)
{
  for (auto i : hemi::grid_stride_range(0, n)) {
    y[i] = a * x[i] + y[i];
  }
}
```

```C++
hemi::cudaLaunch(saxpy, 1<<20, 2.0, x, y);
```

网格步长循环是使您的 CUDA 内核灵活、可扩展、可调试甚至可移植的好方法。 虽然本文中的示例都使用了 `CUDA C/C++`，但相同的概念也适用于其他 CUDA 语言，例如 CUDA Fortran。










