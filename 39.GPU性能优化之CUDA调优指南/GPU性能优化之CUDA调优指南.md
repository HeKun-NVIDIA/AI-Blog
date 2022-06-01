# GPU性能优化之CUDA调优指南

## 1 整体性能优化策略
性能优化围绕四个基本策略：

* 最大化并行执行以实现最大利用率；
* 优化内存使用，实现最大内存吞吐量；
* 优化指令使用，实现最大指令吞吐量；
* 尽量减少内存抖动。
  
哪些策略将为应用程序的特定部分产生最佳性能增益取决于该部分的性能限值； 例如，优化主要受内存访问限制的内核的指令使用不会产生任何显着的性能提升。 因此，应该通过测量和监控性能限制来不断地指导优化工作，例如使用 CUDA 分析器。 此外，将特定内核的浮点运算吞吐量或内存吞吐量（以更有意义的为准）与设备的相应峰值理论吞吐量进行比较表明内核还有多少改进空间。

## 2 最大化利用率
为了最大限度地提高利用率，应用程序的结构应该尽可能多地暴露并行性，并有效地将这种并行性映射到系统的各个组件，以使它们大部分时间都处于忙碌状态。

### 2.1 应用程序层次
在高层次上，应用程序应该通过使用异步函数调用和[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)中描述的流来最大化主机、设备和将主机连接到设备的总线之间的并行执行。它应该为每个处理器分配它最擅长的工作类型：主机的串行工作负载；设备的并行工作负载。

对于并行工作负载，在算法中由于某些线程需要同步以相互共享数据而破坏并行性的点，有两种情况： 这些线程属于同一个块，在这种情况下，它们应该使用 `__syncthreads ()` 并在同一个内核调用中通过共享内存共享数据，或者它们属于不同的块，在这种情况下，它们必须使用两个单独的内核调用通过全局内存共享数据，一个用于写入，一个用于从全局内存中读取。第二种情况不太理想，因为它增加了额外内核调用和全局内存流量的开销。因此，应该通过将算法映射到 CUDA 编程模型以使需要线程间通信的计算尽可能在单个线程块内执行，从而最大限度地减少它的发生。

### 2.2 设备层次
在较低级别，应用程序应该最大化设备多处理器之间的并行执行。

多个内核可以在一个设备上并发执行，因此也可以通过使用流来启用足够多的内核来实现最大利用率，如[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)中所述。

### 2.3 多处理器层次
在更低的层次上，应用程序应该最大化多处理器内不同功能单元之间的并行执行。

如硬件多线程中所述，GPU 多处理器主要依靠线程级并行性来最大限度地利用其功能单元。因此，利用率与常驻warp的数量直接相关。在每个指令发出时，warp 调度程序都会选择一条准备好执行的指令。该指令可以是同一warp的另一条独立指令，利用指令级并行性，或者更常见的是另一个warp的指令，利用线程级并行性。如果选择了准备执行指令，则将其发布到 warp 的活动线程。一个warp准备好执行其下一条指令所需的时钟周期数称为延迟，并且当所有warp调度程序在该延迟期间的每个时钟周期总是有一些指令要为某个warp发出一些指令时，就可以实现充分利用，或者换句话说，当延迟完全“隐藏”时。隐藏 L 个时钟周期延迟所​​需的指令数量取决于这些指令各自的吞吐量（有关各种算术指令的吞吐量，请参见[算术指令](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)）。如果我们假设指令具有最大吞吐量，它等于： 

* 4L 用于计算能力 5.x、6.1、6.2、7.x 和 8.x 的设备，因为对于这些设备，多处理器在一个时钟周期内为每个 warp 发出一条指令，一次四个 warp，如[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)中所述。
* 2L 用于计算能力 6.0 的设备，因为对于这些设备，每个周期发出的两条指令是两条不同warp的一条指令。
* 8L 用于计算能力 3.x 的设备，因为对于这些设备，每个周期发出的八条指令是四对，用于四个不同的warp，每对都用于相同的warp。

warp 未准备好执行其下一条指令的最常见原因是该指令的输入操作数尚不可用。

如果所有输入操作数都是寄存器，则延迟是由寄存器依赖性引起的，即，一些输入操作数是由一些尚未完成的先前指令写入的。在这种情况下，延迟等于前一条指令的执行时间，warp 调度程序必须在此期间调度其他 warp 的指令。执行时间因指令而异。在计算能力 7.x 的设备上，对于大多数算术指令，它通常是 4 个时钟周期。这意味着每个多处理器需要 16 个活动 warp（4 个周期，4 个 warp 调度程序）来隐藏算术指令延迟（假设 warp 以最大吞吐量执行指令，否则需要更少的 warp）。如果各个warp表现出指令级并行性，即在它们的指令流中有多个独立指令，则需要更少的warp，因为来自单个warp的多个独立指令可以背靠背发出。

如果某些输入操作数驻留在片外存储器中，则延迟要高得多：通常为数百个时钟周期。在如此高的延迟期间保持 warp 调度程序繁忙所需的 warp 数量取决于内核代码及其指令级并行度。一般来说，如果没有片外存储器操作数的指令（即大部分时间是算术指令）与具有片外存储器操作数的指令数量之比较低（这个比例通常是称为程序的算术强度）。 


warp 未准备好执行其下一条指令的另一个原因是它正在某个内存栅栏（[内存栅栏函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)）或同步点（[同步函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)）处等待。随着越来越多的warp等待同一块中的其他warp在同步点之前完成指令的执行，同步点可以强制多处理器空闲。在这种情况下，每个多处理器拥有多个常驻块有助于减少空闲，因为来自不同块的warp不需要在同步点相互等待。


对于给定的内核调用，驻留在每个多处理器上的块和warp的数量取决于调用的执行配置（[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)）、多处理器的内存资源以及内核的资源需求，如[硬件多线程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading)中所述。使用 `--ptxas-options=-v` 选项编译时，编译器会报告寄存器和共享内存的使用情况。


一个块所需的共享内存总量等于静态分配的共享内存量和动态分配的共享内存量之和。


内核使用的寄存器数量会对驻留warp的数量产生重大影响。例如，对于计算能力为 6.x 的设备，如果内核使用 64 个寄存器并且每个块有 512 个线程并且需要很少的共享内存，那么两个块（即 32 个 warp）可以驻留在多处理器上，因为它们需要 2x512x64 个寄存器，它与多处理器上可用的寄存器数量完全匹配。但是一旦内核多使用一个寄存器，就只能驻留一个块（即 16 个 warp），因为两个块需要 2x512x65 个寄存器，这比多处理器上可用的寄存器多。因此，编译器会尽量减少寄存器的使用，同时保持寄存器溢出（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)）和最少的指令数量。可以使用 `maxrregcount` 编译器选项或启动边界来控制寄存器的使用，如[启动边界](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds)中所述。

寄存器文件组织为 32 位寄存器。因此，存储在寄存器中的每个变量都需要至少一个 32 位寄存器，例如双精度变量使用两个 32 位寄存器。

对于给定的内核调用，执行配置对性能的影响通常取决于内核代码。因此建议进行实验。应用程序还可以根据寄存器文件大小和共享内存大小参数化执行配置，这取决于设备的计算能力，以及设备的多处理器数量和内存带宽，所有这些都可以使用运行时查询（参见参考手册）。

每个块的线程数应选择为 warp 大小的倍数，以避免尽可能多地在填充不足的 warp 上浪费计算资源。

#### 2.3.1 占用率计算
存在几个 API 函数来帮助程序员根据寄存器和共享内存要求选择线程块大小。

* 占用计算器 API，`cudaOccupancyMaxActiveBlocksPerMultiprocessor`，可以根据内核的块大小和共享内存使用情况提供占用预测。此函数根据每个多处理器的并发线程块数报告占用情况。
* * **请注意，此值可以转换为其他指标。乘以每个块的warp数得出每个多处理器的并发warp数；进一步将并发warp除以每个多处理器的最大warp得到占用率作为百分比。**
* 基于占用率的启动配置器 API，`cudaOccupancyMaxPotentialBlockSize` 和 `cudaOccupancyMaxPotentialBlockSizeVariableSMem`，启发式地计算实现最大多处理器级占用率的执行配置。

以下代码示例计算 MyKernel 的占用率。然后，它使用并发warp与每个多处理器的最大warp之间的比率报告占用率。
```C++
/ Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
    
    return 0;
}
```

下面的代码示例根据用户输入配置了一个基于占用率的内核启动MyKernel。
```C++
// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount)
{
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)MyKernel,
        0,
        arrayCount);

    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
    cudaDeviceSynchronize();

    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor

    return 0;
}
```

CUDA 工具包还在 `<CUDA_Toolkit_Path>/include/cuda_occupancy.h` 中为任何不能依赖 CUDA 软件堆栈的用例提供了一个自记录的独立占用计算器和启动配置器实现。 还提供了占用计算器的电子表格版本。 电子表格版本作为一种学习工具特别有用，它可以可视化更改影响占用率的参数（块大小、每个线程的寄存器和每个线程的共享内存）的影响。


## 3 最大化存储吞吐量
最大化应用程序的整体内存吞吐量的第一步是最小化低带宽的数据传输。

这意味着最大限度地减少主机和设备之间的数据传输，如[主机和设备之间的数据传输](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#data-transfer-between-host-and-device)中所述，因为它们的带宽比全局内存和设备之间的数据传输低得多。

这也意味着通过最大化片上内存的使用来最小化全局内存和设备之间的数据传输：共享内存和缓存（即计算能力 2.x 及更高版本的设备上可用的 L1 缓存和 L2 缓存、纹理缓存和常量缓存 适用于所有设备）。

共享内存相当于用户管理的缓存：应用程序显式分配和访问它。 如 [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime) 所示，典型的编程模式是将来自设备内存的数据暂存到共享内存中； 换句话说，拥有一个块的每个线程：
* 将数据从设备内存加载到共享内存，
* 与块的所有其他线程同步，以便每个线程可以安全地读取由不同线程填充的共享内存位置，
处理共享内存中的数据，
* 如有必要，再次同步以确保共享内存已使用结果更新，
* 将结果写回设备内存。

对于某些应用程序（例如，全局内存访问模式依赖于数据），传统的硬件管理缓存更适合利用数据局部性。如 Compute Capability 3.x、Compute Capability 7.x 和 Compute Capability 8.x 中所述，对于计算能力 3.x、7.x 和 8.x 的设备，相同的片上存储器用于 L1 和共享内存，以及有多少专用于 L1 与共享内存，可针对每个内核调用进行配置。

内核访问内存的吞吐量可能会根据每种内存类型的访问模式而变化一个数量级。因此，最大化内存吞吐量的下一步是根据[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)中描述的最佳内存访问模式尽可能优化地组织内存访问。这种优化对于全局内存访问尤为重要，因为与可用的片上带宽和算术指令吞吐量相比，全局内存带宽较低，因此非最佳全局内存访问通常会对性能产生很大影响。

### 3.1 设备与主机之间的数据传输
应用程序应尽量减少主机和设备之间的数据传输。 实现这一点的一种方法是将更多代码从主机移动到设备，即使这意味着运行的内核没有提供足够的并行性以在设备上全效率地执行。 中间数据结构可以在设备内存中创建，由设备操作，并在没有被主机映射或复制到主机内存的情况下销毁。

此外，由于与每次传输相关的开销，将许多小传输批处理为单个大传输总是比单独进行每个传输执行得更好。

在具有前端总线的系统上，主机和设备之间的数据传输的更高性能是通过使用页锁定主机内存来实现的，如[页锁定主机内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)中所述。

此外，在使用映射页锁定内存（[Mapped Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory)）时，无需分配任何设备内存，也无需在设备和主机内存之间显式复制数据。 每次内核访问映射内存时都会隐式执行数据传输。 为了获得最佳性能，这些内存访问必须与对全局内存的访问合并（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)）。 假设它们映射的内存只被读取或写入一次，使用映射的页面锁定内存而不是设备和主机内存之间的显式副本可以提高性能。

在设备内存和主机内存在物理上相同的集成系统上，主机和设备内存之间的任何拷贝都是多余的，应该使用映射的页面锁定内存。 应用程序可以通过检查集成设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-enumeration)）是否等于 1 来查询设备是否集成。

### 3.2 设备内存访问
访问可寻址内存（即全局、本地、共享、常量或纹理内存）的指令可能需要多次重新发出，具体取决于内存地址在 warp 内线程中的分布。 分布如何以这种方式影响指令吞吐量特定于每种类型的内存，在以下部分中进行描述。 例如，对于全局内存，一般来说，地址越分散，吞吐量就越低。

**全局内存**

全局内存驻留在设备内存中，设备内存通过 32、64 或 128 字节内存事务访问。这些内存事务必须自然对齐：只有32字节、64字节或128字节的设备内存段按其大小对齐(即，其第一个地址是其大小的倍数)才能被内存事务读取或写入。

当一个 warp 执行一条访问全局内存的指令时，它会将 warp 内的线程的内存访问合并为一个或多个内存事务，具体取决于每个线程访问的大小以及内存地址在整个线程中的分布。线程。一般来说，需要的事务越多，除了线程访问的字之外，传输的未使用字也越多，相应地降低了指令吞吐量。例如，如果为每个线程的 4 字节访问生成一个 32 字节的内存事务，则吞吐量除以 8。

需要多少事务以及最终影响多少吞吐量取决于设备的计算能力。 [Compute Capability 3.x、Compute Capability 5.x、Compute Capability 6.x、Compute Capability 7.x 和 Compute Capability 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0) 提供了有关如何为各种计算能力处理全局内存访问的更多详细信息。

为了最大化全局内存吞吐量，因此通过以下方式最大化合并非常重要：

* 遵循基于 Compute Capability 3.x、Compute Capability 5.x、Compute Capability 6.x、Compute Capability 7.x 和 Compute Capability 8.x 的最佳访问模式
* 使用满足以下“尺寸和对齐要求”部分中详述的大小和对齐要求的数据类型，
* 在某些情况下填充数据，例如，在访问二维数组时，如下面的二维数组部分所述。

**尺寸和对齐要求**

全局内存指令支持读取或写入大小等于 1、2、4、8 或 16 字节的字。 当且仅当数据类型的大小为 1、2、4、8 或 16 字节并且数据为 对齐（即，它的地址是该大小的倍数）。

如果未满足此大小和对齐要求，则访问将编译为具有交错访问模式的多个指令，从而阻止这些指令完全合并。 因此，对于驻留在全局内存中的数据，建议使用满足此要求的类型。

[内置矢量类型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types)自动满足对齐要求。

对于结构，大小和对齐要求可以由编译器使用对齐说明符 `__align__(8)` 或 `__align__(16)` 强制执行，例如:
```C++
struct __align__(8) {
    float x;
    float y;
};

struct __align__(16) {
    float x;
    float y;
    float z;
};
```

驻留在全局内存中, 或由驱动程序, 或运行时 API 的内存分配例程之一返回的变量的任何地址始终与至少 256 字节对齐。

读取非自然对齐的 8 字节或 16 字节字会产生不正确的结果（相差几个字），因此必须特别注意保持这些类型的任何值或数组值的起始地址对齐。 一个可能容易被忽视的典型情况是使用一些自定义全局内存分配方案时，其中多个数组的分配（多次调用 `cudaMalloc()` 或 `cuMemAlloc()`）被单个大块内存的分配所取代分区为多个数组，在这种情况下，每个数组的起始地址都与块的起始地址有偏移。

**二维数组**

一个常见的全局内存访问模式是当索引 (tx,ty) 的每个线程使用以下地址访问一个宽度为 width 的二维数组的一个元素时，位于 type* 类型的地址 BaseAddress （其中 type 满足最大化中描述的使用要求 ）：

BaseAddress + width * ty + tx

为了使这些访问完全合并，线程块的宽度和数组的宽度都必须是 warp 大小的倍数。

特别是，这意味着如果一个数组的宽度不是这个大小的倍数，如果它实际上分配了一个宽度向上舍入到这个大小的最接近的倍数并相应地填充它的行，那么访问它的效率会更高。 参考手册中描述的 `cudaMallocPitch()` 和 `cuMemAllocPitch()` 函数以及相关的内存复制函数使程序员能够编写不依赖于硬件的代码来分配符合这些约束的数组。

**本地内存**

本地内存访问仅发生在[可变内存空间说明符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers)中提到的某些自动变量上。 编译器可能放置在本地内存中的变量是：

* 无法确定它们是否以常数索引的数组，
* 会占用过多寄存器空间的大型结构或数组，
* 如果内核使用的寄存器多于可用寄存器（这也称为寄存器溢出），则为任何变量。

检查 PTX 汇编代码（通过使用 `-ptx` 或 `-keep` 选项进行编译）将判断在第一个编译阶段是否已将变量放置在本地内存中，因为它将使用 `.local` 助记符声明并使用 ld 访问`.local` 和 `st.local` 助记符。即使没有，后续编译阶段可能仍会做出其他决定，但如果他们发现它为目标体系结构消耗了过多的寄存器空间：使用 `cuobjdump` 检查 `cubin` 对象将判断是否是这种情况。此外，当使用 `--ptxas-options=-v` 选项编译时，编译器会报告每个内核 (`lmem`) 的总本地内存使用量。请注意，某些数学函数具有可能访问本地内存的实现路径。

本地内存空间驻留在**设备内存**中，因此本地内存访问与全局内存访问具有相同的高延迟和低带宽，并且与设备内存访问中所述的内存合并要求相同。然而，本地存储器的组织方式是通过连续的线程 ID 访问连续的 32 位字。因此，只要一个 warp 中的所有线程访问相同的相对地址（例如，数组变量中的相同索引，结构变量中的相同成员），访问就会完全合并。

在某些计算能力 3.x 的设备上，本地内存访问始终缓存在 L1 和 L2 中，其方式与全局内存访问相同（请参阅[计算能力 3.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0)）。

在计算能力 5.x 和 6.x 的设备上，本地内存访问始终以与全局内存访问相同的方式缓存在 L2 中（请参阅[计算能力 5.x 和计算能力 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x)）。

**共享内存**

因为它是片上的，所以共享内存比本地或全局内存具有更高的带宽和更低的延迟。

为了实现高带宽，共享内存被分成大小相等的内存模块，称为banks，可以同时访问。因此，可以同时处理由落在 n 个不同存储器组中的 n 个地址构成的任何存储器读取或写入请求，从而产生的总带宽是单个模块带宽的 n 倍。

但是，如果一个内存请求的两个地址落在同一个内存 bank 中，就会发生 bank 冲突，访问必须串行化。硬件根据需要将具有bank冲突的内存请求拆分为多个单独的无冲突请求，从而将吞吐量降低等于单独内存请求数量的总数。如果单独的内存请求的数量为 n，则称初始内存请求会导致 n-way bank 冲突。

因此，为了获得最佳性能，重要的是要了解内存地址如何映射到内存组，以便调度内存请求，从而最大限度地减少内存组冲突。这在[计算能力 3.x、计算能力 5.x、计算能力 6.x、计算能力 7.x 和计算能力 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0) 中针对计算能力 3.x、5.x、6.x 7.x 和 8.x 的设备分别进行了描述。

**常量内存**

常量内存空间驻留在设备内存中，并缓存在常量缓存中。

然后，一个请求被拆分为与初始请求中不同的内存地址一样多的单独请求，从而将吞吐量降低等于单独请求数量的总数。

然后在缓存命中的情况下以常量缓存的吞吐量为结果请求提供服务，否则以设备内存的吞吐量提供服务。

**纹理和表面记忆**  

纹理和表面内存空间驻留在设备内存中并缓存在纹理缓存中，因此纹理提取或表面读取仅在缓存未命中时从设备内存读取一次内存，否则只需从纹理缓存读取一次。 纹理缓存针对 2D 空间局部性进行了优化，因此读取 2D 中地址靠近在一起的纹理或表面的同一 warp 的线程将获得最佳性能。 此外，它专为具有恒定延迟的流式提取而设计； 缓存命中会降低 DRAM 带宽需求，但不会降低获取延迟。

通过纹理或表面获取读取设备内存具有一些优势，可以使其成为从全局或常量内存读取设备内存的有利替代方案：
* 如果内存读取不遵循全局或常量内存读取必须遵循以获得良好性能的访问模式，则可以实现更高的带宽，前提是纹理提取或表面读取中存在局部性；
* 寻址计算由专用单元在内核外部执行；
* 打包的数据可以在单个操作中广播到单独的变量；
* 8 位和 16 位整数输入数据可以选择转换为 [0.0, 1.0] 或 [-1.0, 1.0] 范围内的 32 位浮点值（请参阅[纹理内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)）。

## 4最大化指令吞吐量
为了最大化指令吞吐量，应用程序应该：

* 尽量减少使用低吞吐量的算术指令； 这包括在不影响最终结果的情况下用精度换取速度，例如使用内部函数而不是常规函数（内部函数在[内部函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions)中列出），单精度而不是双精度，或者将非规范化数字刷新为零；
* 最大限度地减少由控制流指令引起的发散warp，如[控制流指令](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#control-flow-instructions)中所述
* 减少指令的数量，例如，尽可能优化同步点（如[同步指令](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-instruction)中所述）或使用受限指针（如 [__restrict__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict) 中所述）。

在本节中，吞吐量以每个多处理器每个时钟周期的操作数给出。 对于 32 的 warp 大小，一条指令对应于 32 次操作，因此如果 N 是每个时钟周期的操作数，则指令吞吐量为每个时钟周期的 N/32 条指令。

所有吞吐量都是针对一个多处理器的。 它们必须乘以设备中的多处理器数量才能获得整个设备的吞吐量。

### 4.1 算数指令
[如下图所示](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions)
![Throughput.png](Throughput.png)

其他指令和功能是在本机指令之上实现的。不同计算能力的设备实现可能不同，编译后的native指令的数量可能会随着编译器版本的不同而波动。对于复杂的函数，可以有多个代码路径，具体取决于输入。 `cuobjdump` 可用于检查 `cubin` 对象中的特定实现。

一些函数的实现在 CUDA 头文件（`math_functions.h、device_functions.h`、...）上很容易获得。

通常，使用 `-ftz=true` 编译的代码（非规范化数字刷新为零）往往比使用 `-ftz=false` 编译的代码具有更高的性能。类似地，使用 `-prec-div=false`（不太精确的除法）编译的代码往往比使用 `-prec-div=true` 编译的代码具有更高的性能，使用 `-prec-sqrt=false`（不太精确的平方根）编译的代码往往比使用 `-prec-sqrt=true` 编译的代码具有更高的性能。 nvcc 用户手册更详细地描述了这些编译标志。 

**Single-Precision Floating-Point Division**

`__fdividef(x, y)`（参见[内部函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions)）提供比除法运算符更快的单精度浮点除法。

**Single-Precision Floating-Point Reciprocal Square Root**

为了保留 IEEE-754 语义，编译器可以将 1.0/sqrtf() 优化为 `rsqrtf()`，仅当倒数和平方根都是近似值时（即 `-prec-div=false` 和 `-prec-sqrt=false`）。 因此，建议在需要时直接调用 `rsqrtf()`。

**Single-Precision Floating-Point Square Root**

单精度浮点平方根被实现为倒数平方根后跟倒数，而不是倒数平方根后跟乘法，因此它可以为 0 和无穷大提供正确的结果。

**Sine and Cosine**

sinf(x)、cosf(x)、tanf(x)、sincosf(x) 和相应的双精度指令更昂贵，如果参数 x 的量级很大，则更是如此。

更准确地说，参数缩减代码（参见实现的[数学函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions)）包括两个代码路径，分别称为快速路径和慢速路径。

快速路径用于大小足够小的参数，并且基本上由几个乘加运算组成。 慢速路径用于量级较大的参数，并且包含在整个参数范围内获得正确结果所需的冗长计算。

目前，三角函数的参数缩减代码为单精度函数选择幅度小于105615.0f，双精度函数小于2147483648.0的参数选择快速路径。

由于慢速路径比快速路径需要更多的寄存器，因此尝试通过在本地内存中存储一些中间变量来降低慢速路径中的寄存器压力，这可能会因为本地内存的高延迟和带宽而影响性能（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)）。 目前单精度函数使用28字节的本地内存，双精度函数使用44字节。 但是，确切的数量可能会发生变化。

由于在慢路径中需要进行冗长的计算和使用本地内存，当需要进行慢路径缩减时，与快速路径缩减相比，这些三角函数的吞吐量要低一个数量级。

**Integer Arithmetic**

整数除法和模运算的成本很高，因为它们最多可编译为 20 条指令。 在某些情况下，它们可以用按位运算代替：如果 n 是 2 的幂，则 `(i/n)` 等价于 `(i>>log2(n))` 并且 `(i%n)` 等价于` (i&(n- 1))`; 如果 n 是字母，编译器将执行这些转换。

`__brev` 和 `__popc` 映射到一条指令，而 `__brevll` 和 `__popcll` 映射到几条指令。

`__[u]mul24` 是不再有任何理由使用的遗留内部函数。

**Half Precision Arithmetic**

为了实现 16 位精度浮点加法、乘法或乘法加法的良好性能，建议将 half2 数据类型用于半精度，将 `__nv_bfloat162` 用于 `__nv_bfloat16` 精度。 然后可以使用向量内在函数（例如 `__hadd2、__hsub2、__hmul2、__hfma2`）在一条指令中执行两个操作。 使用 `half2` 或 `__nv_bfloat162` 代替使用 `half` 或 `__nv_bfloat16` 的两个调用也可能有助于其他内在函数的性能，例如warp shuffles。

提供了内在的 `__halves2half2` 以将两个半精度值转换为 `half2` 数据类型。

提供了内在的 `__halves2bfloat162` 以将两个 `__nv_bfloat` 精度值转换为 `__nv_bfloat162` 数据类型。

**Type Conversion**

有时，编译器必须插入转换指令，从而引入额外的执行周期。 情况如下：

* 对 char 或 short 类型的变量进行操作的函数，其操作数通常需要转换为 int，
* 双精度浮点常量（即那些没有任何类型后缀定义的常量）用作单精度浮点计算的输入（由 C/C++ 标准规定）。


最后一种情况可以通过使用单精度浮点常量来避免，这些常量使用 f 后缀定义，例如 3.141592653589793f、1.0f、0.5f。

### 4.2 控制流指令
任何流控制指令（`if、switch、do、for、while`）都可以通过导致相同 warp 的线程发散（即遵循不同的执行路径）来显着影响有效指令吞吐量。如果发生这种情况，则必须对不同的执行路径进行序列化，从而增加为此 warp 执行的指令总数。

为了在控制流取决于线程 ID 的情况下获得最佳性能，应编写控制条件以最小化发散warp的数量。这是可能的，因为正如 [SIMT 架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)中提到的那样，整个块的warp分布是确定性的。一个简单的例子是当控制条件仅取决于 (threadIdx / warpSize) 时，warpSize 是warp大小。在这种情况下，由于控制条件与warp完全对齐，因此没有warp发散。

有时，编译器可能会展开循环，或者它可能会通过使用分支预测来优化短 if 或 switch 块，如下所述。在这些情况下，任何warp都不会发散。程序员还可以使用#`pragma unroll` 指令控制循环展开（参见[#pragma unroll](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll)）。

当使用分支预测时，其执行取决于控制条件的任何指令都不会被跳过。相反，它们中的每一个都与基于控制条件设置为真或假的每线程条件代码或预测相关联，尽管这些指令中的每一个都被安排执行，但实际上只有具有真预测的指令被执行。带有错误预测的指令不写入结果，也不评估地址或读取操作数。

### 4.3 同步指令
对于计算能力为 3.x 的设备，`__syncthreads()` 的吞吐量为每个时钟周期 128 次操作，对于计算能力为 6.0 的设备，每个时钟周期为 32 次操作，对于计算能力为 7.x 和 8.x 的设备，每个时钟周期为 16 次操作。 对于计算能力为 5.x、6.1 和 6.2 的设备，每个时钟周期 64 次操作。

请注意，`__syncthreads()` 可以通过强制多处理器空闲来影响性能，如[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)中所述。

## 5最小化内存抖动
经常不断地分配和释放内存的应用程序可能会发现分配调用往往会随着时间的推移而变慢，直至达到极限。这通常是由于将内存释放回操作系统供其自己使用的性质而预期的。为了在这方面获得最佳性能，我们建议如下：
* 尝试根据手头的问题调整分配大小。不要尝试使用 `cudaMalloc / cudaMallocHost / cuMemCreate` 分配所有可用内存，因为这会强制内存立即驻留并阻止其他应用程序能够使用该内存。这会给操作系统调度程序带来更大的压力，或者只是阻止使用相同 GPU 的其他应用程序完全运行。
* 尝试在应用程序的早期以适当大小分配内存，并且仅在应用程序没有任何用途时分配内存。减少应用程序中的 `cudaMalloc`+`cudaFree` 调用次数，尤其是在性能关键区域。
* 如果应用程序无法分配足够的设备内存，请考虑使用其他内存类型，例如 `cudaMallocHost` 或 `cudaMallocManaged`，它们的性能可能不高，但可以使应用程序取得进展。
* 对于支持该功能的平台，`cudaMallocManaged` 允许超额订阅，并且启用正确的 `cudaMemAdvise` 策略，将允许应用程序保留 `cudaMalloc` 的大部分（如果不是全部）性能。 `cudaMallocManaged` 也不会强制分配在需要或预取之前驻留，从而减少操作系统调度程序的整体压力并更好地启用多原则用例。