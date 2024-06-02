#! https://zhuanlan.zhihu.com/p/687939761
# NVIDIA NIM 提供优化的推理微服务以大规模部署 AI 模型


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/nim-inference-microservices.png)


生成式人工智能的采用率显着上升。 在 2022 年 OpenAI ChatGPT 推出的推动下，这项新技术在几个月内就积累了超过 1 亿用户，并推动了几乎所有行业的开发活动激增。

到 2023 年，开发人员开始使用来自 Meta、Mistral、Stability 等的 API 和开源社区模型进行 POC。

进入 2024 年，组织将重点转向全面生产部署，其中涉及将 AI 模型连接到现有企业基础设施、优化系统延迟和吞吐量、日志记录、监控和安全性等。 这条生产之路既复杂又耗时——它需要专门的技能、平台和流程，尤其是大规模生产。

NVIDIA NIM 是 NVIDIA AI Enterprise 的一部分，为开发 AI 驱动的企业应用程序和在生产中部署 AI 模型提供了简化的路径。

NIM 是一组优化的云原生微服务，旨在缩短上市时间并简化生成式 AI 模型在云、数据中心和 GPU 加速工作站上的部署。 它通过使用行业标准 API 抽象化 AI 模型开发和生产打包的复杂性来扩展开发人员库。

## 用于优化 AI 推理的 NVIDIA NIM
NVIDIA NIM 旨在弥合复杂的 AI 开发世界与企业环境运营需求之间的差距，使企业应用程序开发人员能够为公司的 AI 转型做出 10-100 倍的贡献。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/NVIDIA-NIM.png)

## 部署在任何地方
NIM 专为可移植性和控制而构建，支持跨各种基础设施（从本地工作站到云再到本地数据中心）进行模型部署。 其中包括 NVIDIA DGX、NVIDIA DGX Cloud、NVIDIA 认证系统、NVIDIA RTX 工作站和 PC。

预构建的容器和 Helm 图表与优化模型打包在一起，在不同的 NVIDIA 硬件平台、云服务提供商和 Kubernetes 发行版上进行了严格的验证和基准测试。 这可以为所有 NVIDIA 支持的环境提供支持，并确保组织可以在任何地方部署其生成式 AI 应用程序，从而保持对其应用程序及其处理的数据的完全控制。

## 使用行业标准 API 进行开发
开发者可以通过符合各领域行业标准的API访问AI模型，简化AI应用的开发。 这些 API 与生态系统内的标准部署流程兼容，使开发人员能够快速更新他们的人工智能应用程序——通常只需三行代码。 这种无缝集成和易用性有助于在企业环境中快速部署和扩展人工智能解决方案。

## 利用特定领域的模型
NIM 还通过几个关键功能满足对特定领域解决方案和优化性能的需求。 它打包了特定于领域的 NVIDIA CUDA 库以及针对语言、语音、视频处理、医疗保健等各个领域量身定制的专用代码。 这种方法可确保应用程序准确且与其特定用例相关。

## 在优化的推理引擎上运行
NIM 针对每个模型和硬件设置利用优化的推理引擎，在加速基础设施上提供最佳的延迟和吞吐量。 这降低了推理工作负载扩展时运行的成本，并改善了最终用户体验。 除了支持优化的社区模型之外，开发人员还可以通过将模型与永不离开数据中心边界的专有数据源进行对齐和微调，从而获得更高的准确性和性能。

## 支持企业级人工智能
NIM 是 NVIDIA AI Enterprise 的一部分，采用企业级基础容器构建，通过功能分支、严格验证、服务级别协议的企业支持以及 CVE 的定期安全更新，为企业 AI 软件提供坚实的基础。 全面的支持结构和优化能力强调了 NIM 作为在生产中部署高效、可扩展和定制的 AI 应用程序的关键工具的作用。

## 加速的 AI 模型已准备好部署
NIM 支持多种 AI 模型，例如社区模型、[NVIDIA AI Foundation 模型](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)以及 NVIDIA 合作伙伴提供的自定义 AI 模型，支持跨多个领域的 AI 用例。 这包括大型语言模型 (LLM)、视觉语言模型 (VLM) 以及语音、图像、视频、3D、药物发现、医学成像等模型。

开发人员可以使用 NVIDIA API 目录中的 NVIDIA 托管云 API 来测试最新的生成式 AI 模型。 或者，他们可以通过下载 NIM 自行托管模型，并使用 Kubernetes 在主要云提供商或本地进行快速部署以进行生产，从而缩短开发时间、复杂性和成本。

NIM 微服务通过打包算法、系统和运行时优化以及添加行业标准 API 来简化 AI 模型部署过程。 这使得开发人员能够将 NIM 集成到他们现有的应用程序和基础设施中，而无需进行大量的定制或专业知识。

使用 NIM，企业可以优化其 AI 基础设施，以实现最大效率和成本效益，而无需担心 AI 模型开发复杂性和容器化。 除了加速的 AI 基础设施之外，NIM 还有助于提高性能和可扩展性，同时降低硬件和运营成本。

对于希望为企业应用程序定制模型的企业，NVIDIA 提供了跨不同领域的模型定制微服务。 NVIDIA NeMo 使用法学硕士、语音 AI 和多模式模型的专有数据提供微调功能。 NVIDIA BioNeMo 通过不断增加的生成生物化学和分子预测模型来加速药物发现。 NVIDIA Picasso 通过 Edify 模型实现更快的创意工作流程。 这些模型在视觉内容提供商的许可库上进行训练，从而能够部署用于视觉内容创建的定制生成人工智能模型。


## NVIDIA NIM 入门
NVIDIA NIM 的入门非常简单明了。 在 NVIDIA API 目录中，开发人员可以访问各种 AI 模型，这些模型可用于构建和部署自己的 AI 应用程序。

使用图形用户界面直接在目录中开始原型设计，或直接与免费的 API 交互。 要在您的基础设施上部署微服务，只需注册 NVIDIA AI Enterprise 90 天评估许可证并按照以下步骤操作即可。

1. 从 NVIDIA NGC 下载您要部署的模型。 在此示例中，我们将下载为单个 A100 GPU 构建的 Llama-2 7B 模型版本。

```bash
ngc registry model download-version "ohlfw0olaadg/ea-participants/llama-2-7b:LLAMA-2-7B-4K-FP16-1-A100.24.01"
```

如果您有不同的 GPU，您可以使用 ngc 注册表模型列表“ohlfw0olaadg/ea-participants/llama-2-7b:*”列出模型的可用版本

2. 将下载的工件解压到模型存储库中：

```bash
tar -xzf llama-2-7b_vLLAMA-2-7B-4K-FP16-1-A100.24.01/LLAMA-2-7B-4K-FP16-1-A100.24.01.tar.gz
```

3. 使用您所需的模型启动 NIM 容器：

```bash
docker run --gpus all --shm-size 1G -v $(pwd)/model-store:/model-store --net=host nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01 nemollm_inference_ms --model llama-2-7b --num_gpus=1
```
4. 部署 NIM 后，您可以开始使用标准 REST API 发出请求：

```bash
import requests
 
endpoint = 'http://localhost:9999/v1/completions'
 
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
 
data = {
    'model': 'llama-2-7b',
    'prompt': "The capital of France is called",
    'max_tokens': 100,
    'temperature': 0.7,
    'n': 1,
    'stream': False,
    'stop': 'string',
    'frequency_penalty': 0.0
}
 
response = requests.post(endpoint, headers=headers, json=data)
print(response.json())
```
NVIDIA NIM 是一款强大的工具，可帮助组织加速生产 AI 之旅。 立即开始您的人工智能之旅。
















