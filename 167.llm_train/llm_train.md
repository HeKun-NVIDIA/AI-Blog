# 揭秘万亿参数大型语言模型的 AI 推理部署

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/inference-tech-blog-amazon-tensorrt-llm-featured.jpg)


AI 正在改变每个行业，解决人类面临的重大科学挑战，例如精准药物发现和自动驾驶汽车的开发，以及为了解决商业问题，自动创建电子商务产品描述和从法律合同中提取见解。

如今，每家企业都在探索大型语言模型 (LLM) 创造竞争优势的潜力。NVIDIA Cloud 合作伙伴正在介入，支持企业的 AI 之旅。例如，NexGen Cloud 为其客户提供了通过其按需云平台 Hyperstack 运行概念验证 (PoC) 的机会，然后再承诺签订大规模超级云合同。您可以立即测试运行最新一代的 NVIDIA GPU，从而快速采用其他服务层，例如 NVIDIA AI 平台。

在成功实施试点计划后，许多企业现在正在将这些计划投入生产，以增加利润。这提出了一个重要的问题：企业如何在提供出色的用户体验的同时保持强劲的投资回报？

LLM 生成映射到自然语言并发送回用户的令牌。增加 LLM 部署的令牌吞吐量可让您为更多用户提供服务，从而最大化投资回报率。然而，高吞吐量部署可能会导致较低的用户交互性，即可读文字出现在用户面前的速度，从而导致用户体验不佳。

随着 LLM 的发展，在吞吐量和用户交互性之间取得适当的平衡变得越来越具有挑战性，就像大海捞针一样。

在本文中，我们讨论了不同的部署注意事项，例如批处理、并行化和分块。我们分析了这些不同的部署如何影响混合专家 (MoE) 模型的推理。例如，GPT MoE 1.8T 参数模型具有独立执行计算的子网络，然后组合结果以产生最终输出。我们还重点介绍了 [NVIDIA Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) 和 [NVIDIA AI 推理软件](https://developer.nvidia.com/ai-inference-software)（包括 [NVIDIA NIM](https://www.nvidia.com/en-us/ai/#referrer=ai-subdomain)）的独特功能，与上一代 GPU 相比，这些功能提高了性能。

## 在生产中部署 LLM 的平衡行为
在生产中部署 LLM 的企业旨在通过集成类似虚拟助手的功能来创造新的收入来源或增强其产品的吸引力。但是，他们还必须优先考虑投资回报率并确保引人注目的用户体验。

最大化投资回报率需要在不产生额外基础设施成本的情况下满足更多用户请求。要实现这一点，需要批量处理不同的用户请求并同时处理它们。此设置可最大限度地提高 GPU 资源利用率（每秒每 GPU 的令牌数），使组织能够将其 AI 投资分摊到尽可能多的用户身上。

另一方面，用户体验取决于用户等待 LLM 回复的时间。这以每个用户每秒的令牌数来衡量。

为了最大限度地提高用户交互性，将较小批次的用户请求输入到 GPU，从而最大限度地增加分配给每个请求的 GPU 资源量。批次越小，可以分配给每个请求的 GPU 资源就越多。这种方法可以实现计算操作的并行化，从而加快输出 token 的生成速度，但可能会导致 GPU 资源利用不足。

显然，这两个目标需要权衡。最大化 GPU 吞吐量会导致用户交互性降低，反之亦然。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/llm-deployment-tradeoffs.gif)


## LLM 模型的演进使权衡问题更加复杂
最新一代的 LLM 具有更多的参数和更长的上下文窗口，这使得它们能够在更大的知识库中执行更复杂的认知任务，这种权衡变​​得更加困难。

2018 年 10 月推出的第一个 Transformer 模型 (BERT) 具有 3.4 亿个参数、512 个 token 的短上下文窗口和一个前馈网络。这使得它能够适应单个 GPU。

然而，一些最新的模型已经超过 1T 参数，上下文窗口超过 128K token，并且有多个可以独立运行的前馈网络 (专家)。这些模型无法适应单个 GPU，这意味着必须将模型切成更小的块并在多个 GPU 上并行化。


## 探索万亿参数 MoE 模型的推理空间
以具有 16 位专家的 GPT 1.8T MoE 模型为例，假设固定预算为 64 个 GPU，每个 GPU 具有 192 GB 内存。

使用 FP4 量化，您需要半个字节来存储每个参数，仅存储参数就需要至少 5 个 GPU。但是，为了获得更优化的用户体验，您必须将工作分摊到更多 GPU 上，需要超过最低 GPU 来运行工作负载。

以下是并行化大型模型推理的主要方法，这些模型无法放在单个 GPU 上，每种方法对 GPU 吞吐量和用户交互性的影响都不同：

* 数据并行性
* 张量并行性
* 流水线并行性
* 专家并行性
### 数据并行性
数据并行性 (DP) 方法在不同的 GPU 或 GPU 集群上托管 LLM 模型的多个副本，并在模型的每个副本上独立处理用户请求组。

该方法要求在每个 GPU 或 GPU 集群上复制模型，这不会影响 GPU 吞吐量或用户交互性。请求组之间不需要通信，从而导致服务的用户请求数量与分配的 GPU 资源之间存在线性缩放关系。

对于最新一代的 LLM，单独使用 DP 通常不够用，因为它们的模型权重不适合单个 GPU 内存，需要与其他并行方法一起使用。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/data-parallelism-on-dnn.gif)

### 张量并行
使用张量并行 (TP) 方法，模型的每一层都会分布在多个 GPU 上，用户请求会在 GPU 或 GPU 集群之间共享。每个请求的 GPU 计算结果都会通过 GPU 到 GPU 网络分层重新组合。

对于基于 Transformer 的模型（如 GPT），TP 可以提高用户交互性，因为每个请求都会分配更多的 GPU 资源，从而加快处理时间。

但是，如果在没有超高带宽 GPU 到 GPU 网络结构的情况下将 TP 扩展到大量 GPU，则会导致网络瓶颈，从而对用户交互性产生负面影响。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/tensor-parallelism-on-dnn.gif)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/increasing-tensor-parallelism-768x432.jpg)

### 流水线并行
流水线并行 (PP) 方法的工作原理是将模型层组分布在不同的 GPU 上。处理流水线从一个 GPU 开始，然后通过点对点通信继续到下一个 GPU，按顺序处理集群中所有 GPU 上的请求。

这种方法会导致处理效率降低，并且无法显著优化用户交互性。它确实有助于分配无法放在单个 GPU 上的模型权重。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/pipeline-parallelism-on-dnn.gif)

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/increasing-pipeline-parallelism.jpg)


### 专家并行性
专家并行性 (EP) 方法将请求路由到转换器块中的不同专家，从而减少参数交互。每个请求都被路由到一小组不同的专家。

这大大减少了每个请求必须与之交互的参数数量，因为有些专家被跳过了。在专家处理后，请求必须重新组成其原始 GPU，从而通过 GPU 到 GPU 互连结构产生高网络全对全通信。

这种方法比 TP 更有效，因为您不必将操作拆分成更小的块。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/expert-parallelism-on-dnn.gif)

EP 受模型中专家数量的限制。GPT1.8T 有 16 位专家。鉴于我们仅考虑专家和数据并行性，有两种可能的配置：

* EP8DP8：在单个 GPU 上加载两个专家，并使用 DP 复制配置八次。

* EP16DP4：在每个 GPU 上加载单个专家，并使用 DP 复制配置四次。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/increasing-expert-parallelism.jpg)


## 组合并行技术
并行方法也可以组合，这进一步复杂化了权衡问题。

您可以使用 64-GPU 预算构建 73 种并行配置来为模型提供服务，每种配置都有不同的吞吐量和用户交互性权衡。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/73-combination-configurations-645x363.jpg)


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/gpu-throughput-user-interactivity-parallelism-combinations.jpg)

但是，结合不同的并行技术可以在不产生重大影响的情况下大幅提高性能。

例如，与仅使用专家的并行（EP16DP4 或 EP8DP8）相比，使用专家和流水线并行（EP16PP4）对模型进行并行化可将用户交互性提高 2 倍，而 GPU 吞吐量仅损失约 10%。

同样，与仅使用张量的并行（TP64）相比，使用张量、专家和流水线并行（TP4EP4PP4）对模型进行并行化可将 GPU 吞吐量提高 3 倍，而不会对用户交互性造成任何损失。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/parallelism-combos-optimized.jpg)


## 最大化吞吐量：管理预填充和解码阶段要求的策略
当用户向模型提交请求时，它会经历两个不同的操作阶段：预填充和解码。每个阶段对系统资源的使用方式不同。

在预填充期间，系统处理请求的所有输入令牌以计算中间状态，这对于构建对请求的整体上下文理解至关重要。然后使用这些中间状态生成第一个令牌。此阶段具有很高的计算要求，可以并行化，从而实现高资源利用率和吞吐量。

在解码阶段，系统按顺序生成输出令牌，为每个新令牌更新在预填充阶段计算的中间状态。由于在预填充阶段已经完成了中间状态计算的密集计算，因此此阶段仅处理上一阶段新生成的令牌。因此，它的计算密集度较低，内存带宽密集度较高，可能会导致 GPU 计算资源利用不足。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/prefill-decode-phase-gpu-resource-use.gif)

传统的推理方法称为静态批处理，它涉及按顺序完成一批请求的预填充和解码阶段，然后再进行下一批。由于解码阶段 GPU 利用率不足，以及新请求停滞直到所有当前请求完成，导致用户体验不佳，因此这种方法变得效率低下。

可以使用诸如飞行批处理和分块之类的技术来解决这些问题：

飞行批处理：即使当前批次尚未完全完成，也可以动态插入和逐出请求。
分块：将具有长输入序列的请求的预填充阶段分解为较小的块。这有助于防止这些请求的处理成为正在进行的请求的令牌生成率的瓶颈。
飞行批处理和分块提供更好的 GPU 利用率，同时提供良好的用户体验。

块大小会影响用户交互性权衡决策。使用较大的块大小可以降低处理预填充序列所需的迭代次数，从而缩短第一个令牌的时间 (TTFT)。但是，这也会增加完成正在进行的请求的解码阶段所需的时间，从而降低每秒令牌数 (TPS)。

相反，较小的块大小可以更快地弹出令牌，从而提高 TPS，但也增加 TTFT。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/changing-chunk-size.jpg)

回顾之前使用 64 个 GPU 的 GPT 1.8T 示例，您可以分析分块如何影响权衡问题。首先检查最小为 128 个标记的块，然后以 128 或 256 的增量逐步增加，直至 8,192 个标记。这将搜索空间从之前的 73 种配置显著扩展到超过 2.7K 种并行性和块长度组合的可能性。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/scatter-plot-parallelism-combos.jpg)

为了更好地了解各种块大小对 GPT 1.8T MoE 模型的 GPU 吞吐量和用户交互性的影响，我们选择了一些不同的块大小和并行性配置并分别绘制它们。
![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/chunk-size-impacts.gif)

假设平均读取速度约为每秒 5-6 个单词，相当于每秒 15-18 个令牌，您可以清楚地看到，使用 TP2EP16PP2 配置，您可以在 896 个令牌的块大小下最大化 GPU 吞吐量：

* 两个 GPU 上的张量并行性
* 16 个 GPU 上的专家并行性
* 两个 GPU 上的流水线并行性

## NVIDIA Blackwell：为万亿参数 LLM 提供支持的新平台
NVIDIA Blackwell 是一种采用全新变革技术的 GPU 架构。它简化了优化万亿参数 LLM（如 GPT 1.8T MoE）干扰吞吐量和用户交互性的复杂性。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/peak-throughput-nvidia-blackwell.png)

NVIDIA Blackwell 拥有 208B 晶体管和第二代变压器引擎。它支持 NVIDIA 的第五代 NVLink，可将每个 GPU 的双向吞吐量提高 1.8TB/s。NVLink 支持最多 72 个 NVIDIA Blackwell GPU 的域，为多 GPU 部署的万亿参数模型与并行组合中的 GPU 到 GPU 操作提供无与伦比的加速。

与上一代 NVIDIA H100 Hopper GPU 相比，这些功能相结合，使 NVIDIA Blackwell 能够为所有可能的用户交互要求提供高吞吐量增益。更具体地说，NVIDIA Blackwell 可以使用 TP2EP16PP2 和 896 个令牌的块大小，以每位用户每秒 20 个令牌（每秒 5-6 个字）的读取速度提供 30 倍的吞吐量。

[NVIDIA NIM](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/) 是一组易于使用的推理微服务，用于快速生产部署最新的 AI 模型，包括开源社区模型和 NVIDIA AI Foundation 模型。它已获得 [NVIDIA AI Enterprise ](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/)的许可。

[NIM ](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/)基于 NVIDIA 推理软件构建，包括 TensorRT-LLM，可实现高级多 GPU 和多节点原语。 TensorRT-LLM 还提供高级分块和飞行批处理功能。

将 LLM 作为自定义 AI 管道的一部分部署的企业可以使用 NVIDIA NIM 的一部分 [NVIDIA Triton](https://developer.nvidia.com/triton-inference-server) 推理服务器来创建模型集合，将多个 AI 模型和自定义业务逻辑连接到单个管道中。

## 未来
现在，组织只需几行代码，就可以在模型编译阶段使用数据、张量、管道和专家并行技术并行化万亿参数模型。

[NVIDIA Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)、[TensorRT-LLM](https://developer.nvidia.com/tensorrt) 和 [NVIDIA Triton](https://developer.nvidia.com/triton-inference-server) 推理服务器共同为组织提供了自由探索万亿参数 MoE 模型的整个推理搜索空间的能力，并确定理想的吞吐量和用户交互性组合以满足其服务水平协议，无论所需的吞吐量和用户交互性组合如何。




















































