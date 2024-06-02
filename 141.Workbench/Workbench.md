# 加速您的 AI 开发：NVIDIA AI Workbench 正式发布


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/ai-workbench.png)


NVIDIA AI Workbench 是一款面向 AI 和 ML 开发人员的工具包，现已普遍提供[免费下载](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/)。 它具有自动化功能，可以消除新手开发人员的障碍并提高专家的工作效率。

无论技能水平如何，开发人员都可以体验快速可靠的 GPU 环境设置以及跨异构平台工作、管理和协作的自由。 购买 NVIDIA AI Enterprise 许可证的客户也可以获得企业支持。

AI Workbench 的主要功能包括：

* 基于 GPU 的开发环境的快速安装、设置和配置。
* 基于最新模型的预构建、随时可用的生成式 AI 和 ML 示例项目。
* 使用 [NVIDIA API 目录](https://build.nvidia.com/explore/discover)中的云端点或使用 [NVIDIA NIM](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/) 微服务在本地部署生成式 AI 模型。
* 直观的 UX 加命令行界面 (CLI)。
* 跨开发环境的轻松再现性和可移植性。
* Git 和基于容器的开发环境的自动化。
* 容器和 Git 存储库的版本控制和管理。
* 与 GitHub、GitLab 和 [NVIDIA NGC 目录集成](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=scoreDESC&query=workbench)。
* 透明地处理凭证、机密和文件系统更改。

自 Beta 版本发布以来，AI Workbench 还具有几个新的关键功能：

* Visual Studio (VS) Code 支持：直接与 VS Code 集成，在 GPU 环境上编排容器化项目。
* 基础镜像的选择：用户在创建项目时可以选择自己的容器镜像作为项目基础镜像。 容器镜像必须使用符合基础镜像规范的镜像标签。
* 改进的包管理：用户可以通过 Workbench 用户界面管理包并将其直接添加到容器中。
* 安装改进：用户在 Windows 和 MacOS 上有更简单的安装路径。 还改进了对 Docker 容器运行时的支持。


## 将生成式 AI 引入您的 NVIDIA RTX
生成式人工智能已经爆炸式增长。 AI Workbench 可以通过数亿台现代 NVIDIA RTX 支持的工作站和 PC 上或跨数据中心和云的统一界面，将生成式 AI 开发引入任何支持 GPU 的环境。 Mac 用户可以安装 AI Workbench，并将项目迁移到 NVIDIA 支持的系统，以实现协作和更强的计算能力。

## 更快上手
除了快速的 GPU 工作站设置之外，AI Workbench 还提供示例项目作为现成的起点，帮助开发人员更快地开始处理他们的数据和用例。 工作台项目汇集了简化跨各种基础设施的工作流程管理所需的所有资源和元数据，同时促进在任何地方的无缝可移植性和可重复性。

NVIDIA 提供了一系列免费的 Workbench Project 示例来帮助用户入门：

* 使用[混合检索增强生成 (RAG)](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) 与您的[文档聊天](https://github.com/NVIDIA/workbench-example-hybrid-rag)。 在您的系统上运行嵌入模型，将文档存储在私有矢量数据库中。 将推理配置为使用 NVIDIA API 在云中运行，或使用 RTX 系统上的 NIM 推理微服务在本地运行。
* 定制任意规模的LLM。 从本地运行量化模型到全面微调以优化精度。 微调并在任何地方运行 — 在本地 RTX 系统上或横向扩展到数据中心或云。 在 GitHub 上查看 Llama-2 或 Mistral-7B 项目。
* 通过在 RTX PC 上本地或在云中运行 Stable Diffusion XL，根据文本提示生成自定义图像。 在您选择的支持 GPU 的环境中轻松重现，以根据图像微调模型。

访问 [GitHub](https://github.com/nvidia?q=workbench&type=all&language=&sort=) 上的 NVIDIA，参考 NVIDIA AI Workbench 项目，可以帮助您更快地获得结果。 NVIDIA API 目录中的某些模型具有关联的工作台项目，可以在推理之前对其进行自定义和微调。

## 更好的开发者体验
许多因素都会影响开发人员的工作效率。 让我们看一下其中的一些内容，以及 AI Workbench 在提高生产力和开发体验方面所做的工作。

## 设置和配置
AI Workbench 会自动执行设置目标 GPU 系统的任务，同时为您选择的开发环境配置支持 GPU 的容器。 它确保整个堆栈中兼容组件的正确安装，包括操作系统驱动程序、CUDA 驱动程序和固件。

## 随时随地自由工作和协作
在不同系统和位置之间无缝迁移工作负载，无需担心出现问题。 迁移到协作、速度、规模和成本方面最佳的平台，无论是本地、数据中心还是公共云； Windows、Linux 或 macOS。 在工作台项目中使用公共云 API、本地微服务、容器和流行存储库。 AI Workbench 可以解决可移植性和可重复性挑战的复杂性，因此开发人员无需这样做。

## 托管 AI 和 ML 工作流程
AI Workbench 在后台管理开发工作流程，以执行文件版本控制、位置更改和跟踪项目依赖项等任务。 这使得新手和熟练的开发人员都可以专注于执行，而不必担心配置和管理挑战。

## 获取人工智能工作台
下载适用于 Windows、macOS 和 Ubuntu Linux 的 NVIDIA AI Workbench。 使用 NVIDIA Launchpad，您还可以立即、短期访问以尝试 AI Workbench 示例项目。




















































