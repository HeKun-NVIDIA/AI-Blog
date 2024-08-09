# 使用 NVIDIA NIM 部署生成式 AI 的简单指南

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/nim-gif.gif)

无论您是在本地还是在云端工作，NVIDIA NIM 推理微服务都可以为企业开发人员提供来自社区、合作伙伴和 NVIDIA 的易于部署的优化 AI 模型。作为 NVIDIA AI Enterprise 的一部分，NIM 提供了一条安全、简化的前进道路，可快速迭代并为世界一流的生成式 AI 解决方案构建创新。

使用单个优化容器，您可以在 5 分钟内轻松在云端或数据中心的加速 NVIDIA GPU 系统上，或在工作站和 PC 上部署 NIM。或者，如果您想避免部署容器，您可以开始使用 NVIDIA API 目录中的 [NIM API](https://build.nvidia.com/explore/discover?nvid=nv-int-tblg-805513) 为您的应用程序制作原型。

* 使用预构建的容器，只需一个命令即可在任何地方的 NVIDIA 加速基础设施上进行部署。
* 保持数据的安全性和控制力，这是您最宝贵的企业资源。
* 通过支持使用 LoRA 等技术进行微调的模型来实现最佳准确性。
* 利用一致的行业标准 API 集成加速 AI 推理端点。
* 使用最流行的生成式 AI 应用程序框架，如 LangChain、LlamaIndex 和 Haystack。

本文介绍了 NVIDIA NIM 的简单 Docker 部署。您将能够在最流行的生成式 AI 应用程序框架（如 Haystack、LangChain 和 LlamaIndex）中使用 NIM 微服务 API。有关部署 NIM 的完整指南，请参阅 [NIM 文档](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html?nvid=nv-int-tblg-432774)。

## 如何在 5 分钟内部署 NIM
在开始之前，请确保您已满足所有先决条件。遵循 NIM 文档中的要求。请注意，下载和使用 NIM 需要 NVIDIA AI Enterprise 许可证。

设置好一切后，运行以下脚本：

```bash
# Choose a container name for bookkeeping
export CONTAINER_NAME=meta-llama3-8b-instruct
 
# Choose a LLM NIM Image from NGC
export IMG_NAME="nvcr.io/nim/meta/llama3-8b-instruct:24.05"
 
# Choose a path on your system to cache the downloaded models
export LOCAL_NIM_CACHE="~/.cache/nim"
mkdir -p "$LOCAL_NIM_CACHE"
 
# Start the LLM NIM
docker run -it --rm --name=$CONTAINER_NAME \
  --runtime=nvidia \
  --gpus all \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME
```

接下来测试一个推理请求：

```bash
curl -X 'POST' \
    'http://0.0.0.0:8000/v1/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "meta-llama3-8b-instruct",
      "prompt": "Once upon a time",
      "max_tokens": 64
    }'
```

现在，您拥有一个受控、优化的生产部署，可以安全地构建生成式 AI 应用程序。

NVIDIA API 目录中还提供了 NVIDIA 托管的 NIM 示例部署。

## 如何将 NIM 与您的应用程序集成
虽然应该先完成之前的设置，但如果您急于测试 NIM 而不自行部署，则可以使用 NVIDIA API 目录中 NVIDIA 托管的 API 端点进行测试。请按照以下步骤操作。

### 集成 NIM 端点
您可以从遵循 OpenAI 规范的完成 curl 请求开始。请注意，要流式传输输出，您应该将 stream 设置为 True。

要在带有 OpenAI 库的 Python 代码中使用 NIM：

如果您使用的是 NIM，则无需提供 API 密钥。

确保将 base_url 更新为 NIM 的运行位置。

```python
from openai import OpenAI
 
client = OpenAI(
  base_url = "http://nim-address:8000/v1,
)
 
completion = client.chat.completions.create(
  model="meta/llama3-70b-instruct",
  messages=[{"role":"user","content":""}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)
 
for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

```



NIM 还集成到 Haystack、LangChain 和 LlamaIndex 等应用程序框架中，为已经使用这些流行工具构建出色的生成式 AI 应用程序的开发人员带来安全、可靠、加速的模型推理。

查看每个框架的笔记本以了解如何使用 NIM：

* 带有自部署 AI 模型和 NVIDIA NIM 的 Haystack RAG 管道
* 带有 NVIDIA NIM 的 LangChain RAG 代理
* 带有 NVIDIA NIM 的 LlamaIndex RAG 管道
## 从 NIM 获得更多
通过使用 NVIDIA NIM 进行快速、可靠和简单的模型部署，您可以专注于构建高性能和创新的生成式 AI 工作流程和应用程序。要从 NIM 获得更多，请了解[如何将微服务与使用 LoRA 适配器定制的 LLM 一起使用](https://docs.nvidia.com/nim/large-language-models/latest/peft.html)。

NIM 定期发布和改进。经常访问 [API 目录](https://build.nvidia.com/meta/llama3-8b?nvid=nv-int-tblg-491613)以查看用于视觉、检索、3D、数字生物学等的最新 NVIDIA NIM 微服务。









































































