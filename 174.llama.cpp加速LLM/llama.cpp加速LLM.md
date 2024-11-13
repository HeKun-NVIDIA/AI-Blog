#  NVIDIA RTX 系统上使用 llama.cpp 加速 LLM

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/10/image1.jpg)


适用于 Windows PC 的 NVIDIA RTX AI 平台提供了一个蓬勃发展的生态系统，其中包含数千种开源模型，供应用程序开发人员利用并集成到 Windows 应用程序中。值得注意的是，llama.cpp 是一款流行的工具，在撰写本文时拥有超过 65,000 个 GitHub 星标。这个开源存储库最初于 2023 年发布，是一个轻量级、高效的大型语言模型 (LLM) 推理框架，可在包括 RTX PC 在内的一系列硬件平台上运行。

这篇文章解释了 RTX PC 上的 llama.cpp 如何为构建需要 LLM 功能的跨平台或 Windows 原生应用程序提供引人注目的解决方案。

## llama.cpp 概述
虽然 LLM 在解锁令人兴奋的新用例方面表现出了希望，但它们的大内存和计算密集型特性通常使开发人员难以将它们部署到生产应用程序中。为了解决这个问题，llama.cpp 提供了大量功能来优化模型性能并在各种硬件上高效部署。

llama.cpp 的核心是利用 ggml 张量库进行机器学习。这个轻量级软件堆栈支持跨平台使用 llama.cpp，而无需外部依赖项。它具有极高的内存效率，是本地设备推理的理想选择。模型数据以称为 GGUF 的自定义文件格式打包和部署，由 llama.cpp 贡献者专门设计和实施。

在 llama.cpp 上构建项目的开发人员可以从数千个预打包模型中进行选择，涵盖广泛的高质量量化。一个不断壮大的开源社区正在积极开发 llama.cpp 和 ggml 项目。

## llama.cpp 在 NVIDIA RTX 上的加速性能
NVIDIA 继续合作改进和优化 llama.cpp 在 RTX GPU 上运行时的性能以及开发人员体验。一些关键贡献包括：

* [在 llama.cpp 中实现 CUDA 图表](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)，以减少开销和内核执行时间之间的差距以生成令牌。
* 在准备 ggml 图表时减少 CPU 开销。

有关最新贡献的更多信息，请参阅[使用 CUDA 图表优化 llama.cpp AI 推理](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)。

上图显示了 NVIDIA 内部测量结果，展示了使用 llama.cpp 上的 Llama 3 8B 模型在 NVIDIA GeForce RTX GPU 上的吞吐量性能。在 NVIDIA RTX 4090 GPU 上，用户可以预期每秒约 150 个令牌，输入序列长度为 100 个令牌，输出序列长度为 100 个令牌。

要使用带有 CUDA 后端的 NVIDIA GPU 优化构建 llama.cpp 库，请访问 GitHub 上的 [llama.cpp/docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cuda)。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/10/image2.png)


## 使用 llama.cpp 构建的开发人员生态系统
在 llama.cpp 之上构建了一个庞大的开发人员框架和抽象生态系统，以便开发人员进一步加速他们的应用程序开发之旅。流行的开发人员工具（如 [Ollama](https://ollama.com/)、[Homebrew](http://jan.ai/) 和 [LMStudio](https://lmstudio.ai/)）都在底层扩展并利用了 llama.cpp 的功能，以提供抽象的开发人员体验。其中一些工具的主要功能包括配置和依赖项管理、模型权重的捆绑、抽象的 UI 以及本地运行的 LLM API 端点。

此外，还有一个广泛的模型生态系统，这些模型已经预先优化，可供开发人员在 RTX 系统上使用 llama.cpp 利用。值得注意的模型包括 Hugging Face 上提供的最新 [GGUF 量化版本的 Llama 3.2](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)。

此外，llama.cpp 作为 [NVIDIA RTX AI 工具包](https://github.com/NVIDIA/RTX-AI-Toolkit/blob/main/llm-deployment/llama.cpp_deployment.md)的一部分提供推理部署机制。


## 使用 llama.cpp 在 RTX 平台上加速的应用程序

现在有 50 多个工具和应用程序使用 llama.cpp 加速，包括：

* **Backyard.ai**：使用 Backyard.ai，用户可以在私人环境中完全拥有和控制自己喜欢的角色，通过 AI 释放创造力。该平台利用 llama.cpp 加速 RTX 系统上的 LLM 模型。
* **Brave** 已将智能 AI 助手 Leo 直接内置到 Brave 浏览器中。借助隐私保护的 Leo，用户现在可以提问、总结页面和 PDF、编写代码和创建新文本。借助 Leo，用户可以利用 Ollama（利用 llama.cpp 在 RTX 系统上加速）与设备上的本地 LLM 进行交互。
* **Opera**：Opera 现在已集成本地 AI 模型来增强用户的浏览需求，作为 Opera One 开发者版本的一部分。 Opera 使用 Ollama 集成了这些功能，利用完全在 NVIDIA RTX 系统本地运行的 llama.cpp 后端。在 Opera 的浏览器 AI Aria 中，用户还可以向引擎询问网页摘要和翻译，通过其他搜索获取更多信息，生成文本和图像，并大声朗读响应，支持 50 多种语言。
* **Sourcegraph**：Sourcegraph Cody 是一款 AI 编码助手，支持最新的 LLM，并使用最佳开发人员环境来提供准确的代码建议。Cody 还可以处理在本地机器和隔离环境中运行的模型。它利用使用 llama.cpp 的 Ollama 来支持在 NVIDIA RTX GPU 上加速的本地推理。

## 开始使用
在 [RTX AI PC 上使用 llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cuda) 为开发人员提供了一种引人注目的解决方案，可以加速 GPU 上的 AI 工作负载。借助 llama.cpp，开发人员可以利用轻量级安装包的 C++ 实现进行 LLM 推理。了解更多信息并开始使用 [RTX AI 工具包上的 llama.cpp](https://github.com/NVIDIA/RTX-AI-Toolkit/blob/main/llm-deployment/llama.cpp_deployment.md)。

NVIDIA 致力于为 [RTX AI 平台上](https://www.nvidia.com/en-us/ai-on-rtx/)的开源软件做出贡献并加速其发展。




































