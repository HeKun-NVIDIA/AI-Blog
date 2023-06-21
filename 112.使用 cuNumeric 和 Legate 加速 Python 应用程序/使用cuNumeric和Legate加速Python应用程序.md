# 使用 cuNumeric 和 Legate 加速 Python 应用程序

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/numpy-legion-cunumeric-graphic-1-1.png)

[cuNumeric](https://developer.nvidia.com/cunumeric) 是一个库，旨在为支持所有 NumPy 功能（例如就地更新、广播和完整索引视图语义）的 NumPy API 提供分布式和加速的直接替换。 这意味着任何使用 NumPy 对大型数据集进行操作的 Python 代码都可以自动并行化，以便在切换到使用 cuNumeric 时利用大型 CPU 和 GPU 集群的功能。

[NumPy](https://numpy.org/) 是科学计算中执行基于数组的数值计算的基本 Python 库。 大多数程序员使用的 NumPy 规范实现在单个 CPU 内核上运行，只有少数操作是跨内核并行的。 这种对单节点 CPU 执行的限制限制了可以处理的数据大小和解决问题的速度。

迄今为止，有几个用于 NumPy 的加速替代库可用（例如 CuPy 和 NumS）。 但是，它们都不能在具有许多 CPU 和 GPU 的多节点机器上提供透明的分布式加速，同时仍然支持 NumPy 的所有重要功能。

在 cuNumeric 之前，必须对 NumPy 代码进行重大更改才能在多个节点/GPU 上执行。 这些修改通常包括手动代码并行化和分发逻辑，这些逻辑通常容易出错且性能不佳，并且可能会损失功能。

cuNumeric 旨在为开发人员提供 NumPy 的生产力以及加速和分布式 GPU 计算的性能，而不妥协。 使用 cuNumeric，计算和数据科学家可以在本地机器上的中等大小的数据集上开发和测试程序。 然后可以使用相同的代码立即扩展到部署在超级计算机上许多节点上的更大数据集。

cuNumeric 是在 GTC 2022 期间首次公布的。有关更多详细信息，请参阅 [NVIDIA 宣布 cuNumeric Public Alpha](https://developer.nvidia.com/blog/nvidia-announces-availability-for-cunumeric-public-alpha/) 的可用性。 此后的更新包括将 API 覆盖率从 NumPy API 的 20% 增加到 60%、对 Jupyter 笔记本的支持以及改进的性能。

cuNumeric 已经证明它可以扩展到数千个 GPU。 例如，从 GitHub 上的 CFDPython 摘录的 CFD 代码在切换到使用 cuNumeric 时显示了良好的缩放结果。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/weak-scaling-CFD-cuNumeric.png)

cuNumeric 中的隐式数据分布和并行化是通过 Legate 实现的。
Legate 是一个生产力层，可以更轻松地在 Legion 运行时之上构建可组合层，以便在异构集群上执行。 cuNumeric 是 Legate 生态系统的一部分，这意味着 cuNumeric 程序可以透明地将对象传入和传出生态系统中的其他库，而不会导致不必要的同步或数据移动，即使在分布式设置中也是如此。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/Legate-software-stack-ecosystem.png)

## 使用 cuNumeric
使用 cuNumeric 只需要在 NumPy 代码中将 import numpy as np 替换为 import cunumeric as np 并使用 Legate 驱动程序脚本来执行程序。

cuNumeric 代码的一个简单示例如下所示：

```python
import cunumeric as np
a = np.arange(10000, dtype =int)
a = a.reshape((100,100,))
b = np.arange(10000, dtype =int)
b = b.reshape((100,100,))
c = np.multiply(a, b)
print(c)
print(type(c))
```
```bash
[[       0        1        4 ...     9409     9604     9801]
 [   10000    10201    10404 ...    38809    39204    39601]
 [   40000    40401    40804 ...    88209    88804    89401]
 ...
 [94090000 94109401 94128804 ... 95981209 96000804 96020401]
 [96040000 96059601 96079204 ... 97950609 97970404 97990201]
 [98010000 98029801 98049604 ... 99940009 99960004 99980001]]
<class 'cunumeric.array.ndarray'>
```
只有第一个导入更改需要从 NumPy 迁移到 cuNumeric。 该代码现在可在多个 GPU 上执行。 数组 a、b 和 c 在 GPU 之间进行分区，以便在 a 的不同分片上异步执行排列、重塑和乘法操作。 有关更多详细信息，请参阅下面有关 cuNumeric 自动数据分区的部分。

## cuNumeric 自动数据分区
cuNumeric 隐式划分其数据对象，同时考虑将访问数据的计算、不同处理器类型使用的理想数据大小以及可用的处理器数量。 子分区的一致性由 Legion 自动管理，无论机器的规模如何。

下图显示了 cuNumeric 2D 数组在四个进程之间的相等分区的可视化。 在执行数据并行操作（例如添加）时，不同颜色的图块将由单独的任务异步处理。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/cuNumeric-implicit-data-partitioning.png)


请注意，不同的 cuNumeric API 可以重用现有分区或请求不同的分区以满足特定需求。 允许多个分区共存，并自动保持同步。 Legion 将仅在需要时复制和重新格式化数据，并将尝试以最有效的方式执行此操作。

## 使用 cuNumeric 异步执行
除了针对每个任务在分区数组的不同部分上异步执行计算外，cuNumeric 还可以在资源可用的情况下执行异步任务和/或操作执行。 底层运行时将创建一个依赖图，然后以分布式乱序方式执行操作，同时保留数据依赖性。

下图可视化了在四个 GPU（单节点）上执行的第一个示例的依赖关系图。 在这里，数组 a 的排列、重塑任务和复制操作可以与数组 b 的那些操作并行执行。 请注意，每个阵列范围的操作也分为四个子操作。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/asynchronous-execution-cuNumeric.jpg)

## 单节点安装和执行
cuNumeric 在 Anaconda 上可用。 要安装 cuNumeric，请执行以下脚本：
```bash
conda install -c nvidia -c conda-forge -c legate cunumeric
```
conda 包与 CUDA >= 11.4（CUDA 驱动程序版本 >= r470）和 NVIDIA Volta 或更高版本的 GPU 架构兼容。

请注意，目前 cuNumeric conda 包中不支持 Mac 操作系统。 如果您在 Mac OS 上安装，请参阅下面有关多节点安装和执行的部分，以获取有关手动安装 cuNumeric 的说明。cuNumeric 程序使用 Legate 文档中描述的 Legate 驱动程序脚本运行：

```bash
legate cunumeric_program.py
```
以下运行时选项可用于控制设备数量：
```bash
  --cpus CPUS                Number of CPUs to use per rank
  --gpus GPUS                Number of GPUs to use per rank
  --omps OPENMP              Number of OpenMP groups to use per rank
  --ompthreads               Number of threads per OpenMP group
```

您可以按照 Legate 文档中的描述或通过将 --help 传递给 Legate 驱动程序脚本来熟悉这些资源标志。 应该在不久的将来取消对基于 Legate 的代码使用 Legate 驱动程序脚本的要求。 （cuNumeric 代码将与标准 python 解释器一起工作。）

## Jupyter notebook和 cuNumeric
当安装在系统上时，cuNumeric 也可以与 Jupyter notebook 一起使用。 具体的Jupyter内核应该配置安装如下：
```bash
legate-jupyter --name legate_cpus_2 --cpus 2
```
可以使用 –help 命令行选项查看内核的其他配置选项。

应使用以下脚本启动 Jupyter 服务器：
```bash
jupyter notebook --port=888 --no-browser
```
然后应使用 Legion wiki 页面上提供的说明在浏览器中打开 Jupyter notebook。

## 多节点安装和执行
要支持多节点执行，必须手动安装 cuNumeric。 手动执行包括以下步骤：

1. 使用以下代码从 GitHub 克隆 Legate：
```bash
git clone git@github.com:nv-legate/legate.core.git
cd legate.core
```
2. 安装 Legate 和 cuNumeric 依赖项。

    检索依赖项的主要方法是通过 conda。 使用来自 Legtate 的 [scripts/generate-conda-envs.py](https://github.com/nv-legate/legate.core/blob/branch-22.12/scripts/generate-conda-envs.py) 脚本创建一个 conda 环境文件，列出构建、运行和测试 Legate Core 以及目标系统上的所有下游库所需的所有包。 例如：
    ```bash
    $ ./scripts/generate-conda-envs.py --python 3.10 --ctk 11.7 --os linux --compilers --openmpi 
    --- generating: environment-test-linux-py310-cuda-11.7-compilers-openmpi.yaml
    ```

    生成环境文件后，通过使用以下脚本创建新的 conda 环境来安装所需的包：
    ```bash
    conda env create -n legate -f <env-file>.yaml
    ```
3. 安装Legate

    Legate Core 存储库在顶层目录中附带一个帮助程序 install.py 脚本，它将构建库的 C++ 部分并在当前活动的 Python 环境下安装 C++ 和 Python 组件。

    要添加 GPU 支持，请使用 --cuda 标志：
    ```bash
    ./install.py --cuda
    ```
    如果在安装期间未找到 CUDA，请将 CUDA_PATH 变量设置为正确的位置。 例如：
    ```bash
    CUDA_PATH=/usr/local/cuda-11.6/lib64/stubs ./install.py --cuda
    ```
    对于多节点执行，Legate 使用 GASNet，可以使用 --network 标志请求。 使用 GASNet 时，您还需要使用 --conduit 标志指定目标机器的互连网络。 例如，以下代码将是 DGX SuperPOD 的安装：
    ```bash
    ./install.py --network gasnet1 --conduit ibv --cuda
    ```

4. 使用以下调用克隆并安装 cuNumeric：
    ```bash
    git clone git@github.com:nv-legate/cunumeric.git
    cd cunumeric
    ./install.py 
    ```
    有关 cuNumeric 安装选项的更多详细信息，包括多节点配置设置，请访问 GitHub 上的 [nv-legate/cunumeri#build](https://github.com/nv-legate/cunumeric/blob/branch-22.12/BUILD.md)。

    如果 Legate 编译时带有允许多节点执行的网络支持，则可以通过使用 --nodes 选项后跟要使用的节点数来并行运行它。 每当使用 --nodes 选项时，Legate 将使用 mpirun 启动，即使 --nodes 为 1。如果没有 --nodes 选项，则不会使用启动器。

    Legate 目前支持 mpirun、srun 和 jsrun 作为启动器，并且可能会添加其他启动器类型。 您可以使用 –launcher 选择目标类型的启动器。 例如，以下命令将在 64 个节点上执行 cuNumeric 代码，每个节点 8 个 GPU：
    ```bash
    legate cunumeric_program.py --nodes 64 --gpus 8
    ```

## cuNumeric例子
cuNumeric 在其存储库中有几个示例代码，可用于熟悉该库。 为简单起见，这篇文章从 Stencil 示例开始。

### 使用 cuNumeric 进行模板计算
Stencil 代码演示了如何以不同的比例编写和执行 cuNumeric 代码。 模板代码可以在许多数值求解器和物理模拟代码的核心找到，因此对科学计算研究特别感兴趣。 本节介绍 cuNumeric 中简单模板代码示例的实现。

首先，使用以下脚本创建和初始化网格：

```python
import cunumeric as np

N = 1000 # number of elements in one dimension
I = 100 # number of iterations

def initialize(N):
	print("Initializing stencil grid...")
	grid = np.zeros((N + 2, N + 2))
	grid[:, 0] = -273.15
	grid[:, -1] = -273.15
	grid[-1, :] = -273.15
	grid[0, :] = 40.0
	return grid
```
这个“初始化”函数将分配 (N+2)x(N+2) 零点的二维矩阵并填充边界元素。

接下来，执行 Stencil 计算：

```python
def run_stencil():
	grid = initialize(N)

	center = grid[1:-1, 1:-1]
	north = grid[0:-2, 1:-1]
	east = grid[1:-1, 2:]
	west = grid[1:-1, 0:-2]
	south = grid[2:, 1:-1]

	for i in range(I):
    	average = center + north + east + west + south
    	average = 0.2 * average
    	center[:] = average

run_stencil()

```
此代码由 cuNumeric 在所有可用资源中完全并行化。 它可以使用以下调用在单个节点上执行：

```bash
legate examples/stencil.py --gpus 8
```
它可以使用以下调用在多个节点上执行：

```bash
legate examples/stencil.py --nodes 128 --gpus 8
```

要查看此示例的原始代码，请访问 GitHub 上的 [nv-legate/cunumeric](https://github.com/nv-legate/cunumeric/blob/branch-22.12/examples/stencil.py)。

## 模板示例性能结果
下图 显示了 Stencil 代码的弱缩放结果。 每个 GPU 的网格点数保持不变（每个 GPU 12264004 个点），增加了问题的总规模。 如图所示，该示例在没有程序员帮助的情况下几乎可以在大型系统上完美扩展。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/weak-scaling-results-Stencil-cuNumeric2.png)

## 分析 cuNumeric 代码
开发功能性 cuNumeric 应用程序后，通常需要分析和调整性能。 本节涵盖用于分析 cuNumeric 代码的不同选项。

## Legate prof
要获得 Legion 级别的分析输出，请在执行 cuNumeric 代码时传递 –profile 标志。 在执行结束时，将创建一个 legate_prof 目录。 该目录包含一个网页，可以在任何显示程序执行时间线的网络浏览器中查看。 请注意，如果您从本地计算机（取决于您的浏览器）查看页面，则可能需要[启用本地 JavaScript](https://legion.stanford.edu/profiling/#using-legion-prof-locally) 执行。

下图 显示了执行 Stencil 示例的分析输出。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/Stencil-profile-output-Legate-profiler.png)

## NVIDIA Nsight 系统
运行时标志用于获取 Stencil cuNumeric 代码的 NVIDIA Nsight Systems 分析器输出：--nsys。 传递此标志时，将生成一个可以加载到 Nsight 系统 UI 中的输出文件。 cuNumeric 生成的 nsys 文件的可视化如下图所示。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/stencil-profile-output-nvidia-nsight-systems2.png)

## 调试 cuNumeric 代码
Legate 提供了在 cuNumeric 应用程序运行期间检查 Legion 构建的数据流和事件图的工具。 构建这些图需要您在计算机上安装可用的 [GraphViz](https://graphviz.org/)。

要为您的程序生成数据流图，请将 --dataflow 标志传递给 legate 脚本。 运行完成后，库将生成一个包含程序数据流图的 dataflow_legate PDF 文件。 要生成相应的事件图，请将 --event 标志传递给 legate.py 脚本以生成 event_graph_legate PDF 文件。

上图和下图 显示了在使用 cuNumeric（上文）部分中的简单示例在四个 GPU 上执行时生成的数据流和事件图。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/cuNumeric-data-flow-graph.png)


![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/01/Profiler-output-cuNumeric-event-graph.png)



## cuNumeric现状和未来计划
cuNumeric 目前正在进行中。 正在逐步添加对未实现的 NumPy 运算符的支持。 在 cuNumeric 的 alpha 版本和最新版本 (v23.01) 之间，API 覆盖率从 25% 增加到 60%。 对于当前不受支持的 API，会提供警告并调用规范的 NumPy。 [API 参考](https://nv-legate.github.io/cunumeric/api/index.html)中提供了可用功能的完整列表。

虽然 cuNumeric v23.01 为许多 cuNumeric 应用程序提供了良好的弱缩放结果，但众所周知，一些改进将导致某些 API/用例的峰值性能。 接下来的几个版本将专注于提高性能，努力在 2023 年实现 API 的全面覆盖。

## 总结
这篇文章介绍了 cuNumeric，它是基于 Legion 编程系统的 NumPy 的替代品。 它透明地加速 NumPy 程序并将其分发到任何规模和功能的机器，通常是通过更改单个模块导入语句。 cuNumeric 通过将 NumPy 应用程序接口转换为 Legate 编程模型并利用 Legion 运行时的性能和可扩展性来实现这一点。





