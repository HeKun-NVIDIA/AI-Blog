# 利用 NVIDIA TensorRT-LLM 和 NVIDIA Triton 推理服务器提升 Meta Llama 3 性能


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/dev-llama3-blog-1920x1080-1.png)


我们很高兴地宣布 NVIDIA [TensorRT-LLM](https://developer.nvidia.com/tensorrt#inference) 支持 Meta Llama 3 系列模型，从而加速和优化您的 LLM 推理性能。 您可以通过浏览器用户界面立即试用 [Llama 3 8B](https://build.nvidia.com/meta/llama3-8b) 和 [Llama 3 70B](https://build.nvidia.com/meta/llama3-70b)（该系列中的首款型号）。 或者，通过在 [NVIDIA API 目录](http://build.nvidia.com/)中完全加速的 NVIDIA 堆栈上运行的 API 端点，其中 Llama 3 被打包为 [NVIDIA NIM](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/)，具有可部署在任何地方的标准 API。

大型语言模型是计算密集型的。 它们的尺寸使得它们昂贵且运行缓慢，尤其是在没有正确的技术的情况下。 许多优化技术都可用，例如内核融合和量化到运行时优化（如 C++ 实现、KV 缓存、连续运行中批处理和分页注意力）。 开发人员必须决定哪种组合有助于他们的用例。 TensorRT-LLM 简化了这项工作。

TensorRT-LLM 是一个开源库，可加速 NVIDIA GPU 上最新 LLM 的推理性能。 NeMo 是一个用于构建、定制和部署生成式 AI 应用程序的端到端框架，它使用 TensorRT-LLM 和 [NVIDIA Triton 推理服务器](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/)进行生成式 AI 部署。

TensorRT-LLM 使用 [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) 深度学习编译器。 它包括用于 FlashAttention 尖端实现的最新优化内核以及用于 LLM 模型执行的屏蔽多头注意力 (MHA)。 它还由简单的开源 Python API 中的预处理和后处理步骤以及多 GPU/多节点通信原语组成，可在 GPU 上实现突破性的 LLM 推理性能。

为了了解该库以及如何使用它，让我们看一下如何通过 TensorRT-LLM 和 Triton 推理服务器使用和部署 Llama 3 8B 的示例。

如需更深入的了解（包括不同的模型、不同的优化和多 GPU 执行），请查看 TensorRT-LLM 示例的完整列表。


## 开始安装
我们将首先使用 pip 命令按照操作系统特定的安装说明克隆和构建 TensorRT-LLM 库。 这是构建 TensorRT-LLM 的更简单方法之一。 或者，可以使用 dockerfile 检索依赖项来安装该库。

以下命令拉取开源库并安装在容器内安装 TensorRT-LLM 所需的依赖项。

```bash
git clone -b v0.8.0 https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
```

## 检索模型权重
TensorRT-LLM 是一个用于 LLM 推理的库。 要使用它，您必须提供一组经过训练的权重。 可以从 Hugging Face Hub 或 NVIDIA NGC 等存储库中提取一组权重。 另一种选择是使用在 NeMo 等框架中训练的您自己的模型权重。

本文中的命令会自动从 Hugging Face Hub 中提取 80 亿参数 Llama 3 模型的指令调整变体的权重（和分词器文件）。 您还可以使用以下命令下载权重以供离线使用，并更新后面命令中的路径以指向此目录：
```bash
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```
请注意，使用此模型需要特定的许可证。 同意条款并通过 HuggingFace 进行身份验证以下载必要的文件。

## 运行 TensorRT-LLM 容器
我们将启动一个基础 docker 容器并安装 TensorRT-LLM 所需的依赖项。

```bash
# Obtain and start the basic docker image environment.
docker run --rm --runtime=nvidia --gpus all --volume ${PWD}:/TensorRT-LLM --entrypoint /bin/bash -it --workdir /TensorRT-LLM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Install the stable version (corresponding to the cloned branch) of TensorRT-LLM.
pip3 install tensorrt_llm==0.8.0 -U --extra-index-url https://pypi.nvidia.com


```

## 编译模型
该过程的下一步是将模型编译到 TensorRT 引擎中，并使用 TensorRT-LLM Python API 编写模型权重和模型定义。

TensorRT-LLM 存储库包含多个模型架构，我们使用 Llama 模型定义。 有关更多详细信息以及更强大的插件和可用量化，请参阅此 [Llama 示例](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0/examples/llama)和[精度文档](https://nvidia.github.io/TensorRT-LLM/precision.html)。


```bash
# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****

# Build the Llama 8B model using a single GPU and BF16.
python3 examples/llama/convert_checkpoint.py --model_dir ./Meta-Llama-3-8B-Instruct \
            --output_dir ./tllm_checkpoint_1gpu_bf16 \
            --dtype bfloat16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/llama/8B/trt_engines/bf16/1-gpu \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16

```

当我们使用 TensorRT-LLM API 创建模型定义时，我们会根据构成神经网络各层的 TensorRT 原语构建操作图。 这些操作映射到特定的内核，这些内核是为 GPU 预先编写的程序。

TensorRT 编译器可以扫描图表，为每个操作和每个可用 GPU 选择最佳内核。 它还可以识别图中的模式，其中多个操作适合合并到单个融合内核中，从而减少所需的内存移动量和启动多个 GPU 内核的开销。

此外，TensorRT 将操作图构建到可以同时启动的 NVIDIA CUDA Graph 中。 这进一步减少了启动内核的开销。

TensorRT编译器在融合层和提高执行速度方面非常高效，但是，有一些复杂的层融合，例如FlashAttention，涉及将许多操作交错在一起并且无法自动发现。 对于这些，我们可以在编译时用插件显式替换部分图。 在我们的示例中，我们包含 gpt_attention 插件（它实现了类似 FlashAttention 的融合注意力内核）和 gemm 插件（它通过 FP32 累加执行矩阵乘法）。 我们还将完整模型所需的精度称为 FP16，与我们从 HuggingFace 下载的权重的默认精度相匹配。

当我们完成运行构建脚本时，我们应该会在 `/tmp/llama/8B/trt_engines/bf16/1-gpu` 文件夹中看到以下三个文件：

* rank0.engine 是我们构建脚本的主要输出，包含嵌入模型权重的可执行操作图。
* config.json 包含有关模型的详细信息，例如其一般结构和精度，以及有关引擎中合并了哪些插件的信息。

## 运行模型
那么，现在我们已经有了模型引擎，我们可以用它做什么呢？

引擎文件包含执行模型的信息。 TensorRT-LLM 包括高度优化的 C++ 运行时，用于执行引擎文件和管理流程，例如从模型输出中采样令牌、管理 KV 缓存以及一起批处理请求。

我们可以直接使用运行时在本地执行模型，也可以在生产环境中使用Triton Inference Server进行部署，以便与多个用户共享模型。

要在本地运行模型，我们可以执行以下命令：

```bash
python3 examples/run.py --engine_dir=./tmp/llama/8B/trt_engines/bf16/1-gpu --max_output_len 100 --tokenizer_dir ./Meta-Llama-3-8B-Instruct --input_text "How do I count to nine in French?"


```

## 使用 Triton 推理服务器进行部署
除了本地执行之外，我们还可以使用 Triton Inference Server 来创建 LLM 的生产就绪部署。 TensorRT-LLM 的 Triton 推理服务器后端使用 TensorRT-LLM C++ 运行时来实现高性能推理执行。 它包括动态批处理和分页 KV 缓存等技术，可在低延迟的情况下提供高吞吐量。 TensorRT-LLM 后端已与 Triton 推理服务器捆绑在一起，并可作为 [NGC 上](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)的预构建容器使用。

首先，我们必须创建一个模型存储库，以便 Triton 推理服务器可以读取模型和任何关联的元数据。

tensorrtllm_backend 存储库包括我们可以复制的 all_models/inflight_batcher_llm/ 下所需模型存储库的设置。

该目录中有四个子文件夹，其中保存模型执行过程不同部分的工件。 preprocessing/ 和 postprocessing/ 文件夹包含 Triton Inference Server python 后端的脚本。 这些脚本用于对文本输入进行标记，并对模型输出进行去标记，以在字符串和模型运行的标记 ID 之间进行转换。

tensorrt_llm 文件夹是我们放置之前编译的模型引擎的位置。 最后，ensemble 文件夹定义了一个模型集成，它将前面的三个组件链接在一起，并告诉 Triton 推理服务器如何通过它们流动数据。

拉下示例模型存储库并将您在上一步中编译的模型复制到其中。
```bash
# After exiting the TensorRT-LLM docker container
cd ..
git clone -b v0.8.0 https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
cp ../TensorRT-LLM/tmp/llama/8B/trt_engines/bf16/1-gpu/* all_models/inflight_batcher_llm/tensorrt_llm/1/

```
接下来，我们必须使用已编译模型引擎的位置修改存储库骨架中的配置文件。 我们还必须更新配置参数（例如分词器），以便在批处理推理请求时使用和处理 KV 缓存的内存分配。

```bash
#Set the tokenizer_dir and engine_dir paths
HF_LLAMA_MODEL=TensorRT-LLM/Meta-Llama-3-8B-Instruct
ENGINE_PATH=tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,preprocessing_instance_count:1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,postprocessing_instance_count:1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:64

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0

```
现在，我们可以启动 docker 容器并启动 Triton 服务器。 我们必须指定世界大小（模型构建的 GPU 数量）并指向我们刚刚设置的 model_repo。

```bash
#Change to base working directory
cd..
docker run -it --rm --gpus all --network host --shm-size=1g \
-v $(pwd):/workspace \
--workdir /workspace \
nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3

# Log in to huggingface-cli to get tokenizer
huggingface-cli login --token *****

# Install python dependencies
pip install sentencepiece protobuf

# Launch Server

python3 tensorrtllm_backend/scripts/launch_triton_server.py --model_repo tensorrtllm_backend/all_models/inflight_batcher_llm --world_size 1


```


## 发送请求
要发送推理请求并从正在运行的服务器接收完成，您可以使用 Triton 推理服务器客户端库之一或将 HTTP 请求发送到生成的端点。
以下curl命令演示了对正在运行的服务器请求完成的快速测试，并且可以查看功能更齐全的客户端脚本以与服务器进行通信。

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
"text_input": "How do I count to nine in French?",
"parameters": {
"max_tokens": 100,
"bad_words":[""],
"stop_words":[""]
}
}'


```







































































