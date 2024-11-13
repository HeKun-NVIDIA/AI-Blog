# 利用NVIDIA 工具创建基于 RAG 的问答式 LLM 工作流程

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/10/llamaindex-workflow-chat-app-featured-1.gif)

使用检索增强生成 (RAG) 进行问答 LLM 工作流的解决方案的快速发展导致了新型系统架构的出现。我们在 NVIDIA 使用 AI 进行内部运营的工作取得了一些重要发现，有助于发现系统功能与用户期望之间的一致性。

我们发现，无论预期范围或用例如何，用户通常都希望能够执行非 RAG 任务，例如执行文档翻译、编辑电子邮件甚至编写代码。原始 RAG 应用程序可能会被实现为对每条消息执行检索管道，从而导致过度使用令牌和不必要的延迟，因为其中包含了不相关的结果。

我们还发现，即使应用程序是为访问内部私有数据而设计的，用户也非常欣赏能够访问网络搜索和摘要功能。例如，我们使用 Perplexity 的搜索 API 来满足这一需求。

在这篇文章中，我们分享了一个解决这些问题的基本架构，使用路由和多源 RAG 来生成一个能够回答广泛问题的聊天应用程序。这是一个精简版的应用程序，有很多方法可以构建基于 RAG 的应用程序，但这可以帮助您入门。有关更多信息，请参阅 /NVIDIA/GenerativeAIExamples GitHub 存储库。

特别是，我们重点介绍了如何使用 LlamaIndex、NVIDIA NIM 微服务和 Chainlit 快速部署此应用程序。您可以将此项目作为 NVIDIA 和 LlamaIndex 开发者大赛的灵感来源，展示这些技术在实际应用中的创新用途，并有机会赢得令人兴奋的奖品。

我们发现这些技术之间存在巨大的协同作用。NVIDIA NIM 微服务及其 LlamaIndex 连接器使使用自管理或托管 LLM 开发 LLM 应用程序变得毫不费力。Chainlit 和 LlamaIndex Workflow 事件由于它们共享的事件驱动架构而完美地结合在一起，这使得为用户界面提供有关 LLM 响应完整跟踪的详细信息变得容易。我们在本文中概述了更多系统细节。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/10/system-architecture-rag-workflow.png)



## 用于 LLM 部署的 NIM 推理微服务
我们的项目是围绕 NVIDIA NIM 微服务构建的，适用于多种模型，包括：

* [Meta 的 llama-3.1-70b-instruct](https://build.nvidia.com/meta/llama-3_1-70b-instruct)
* [NVIDIA 的 nv-embed-v1 用于文本嵌入](https://build.nvidia.com/nvidia/nv-embed-v1/modelcard)
* [Mistral 的 nv-rerankqa-mistral-4b-v3 用于重新排名](https://build.nvidia.com/nvidia/nv-rerankqa-mistral-4b-v3)

尽管我们的团队中没有任何机器学习工程师或 LLM 推理专家，但我们在短短几个小时内就使用在配备 NVIDIA A100 的节点（8 个 GPU）上运行的 NIM 容器请求并部署了我们自己的 llama-3.1-70b-instruct 实例。这帮助我们规避了我们在某些企业 LLM API 中发现的可用性和延迟问题。

要试用 NIM API，请在 [build.nvidia.com](http://build.nvidia.com/) 注册一个帐户并获取 API 密钥。要在此项目中使用 API 密钥，请确保它在项目目录中的 .env 文件中可用。适用于 NVIDIA 模型和 API 的 LlamaIndex 连接器可在 Python 包 [llama-index-llms-nvidia](https://docs.llamaindex.ai/en/stable/examples/llm/nvidia/) 中找到。有关基于 NIM 的 LLM 部署的性能优势的更多信息，请参阅[使用 NVIDIA NIM 微服务优化大规模 LLM 的推理效率](https://developer.nvidia.com/blog/optimizing-inference-efficiency-for-llms-at-scale-with-nvidia-nim-microservices/)。



### LlamaIndex 工作流事件
我们的第一个版本的应用程序是围绕 LlamaIndex 的 [ChatEngine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/) 类构建的，它提供了一个交钥匙解决方案，用于部署由矢量数据库支持的对话式 AI 助手。虽然这很有效，但我们发现我们想要注入额外的步骤来增强上下文并切换功能，而这需要更多的可扩展性。

幸运的是，LlamaIndex 工作流事件通过其事件驱动、基于步骤的方法来控制应用程序的执行流程，提供了我们所需的解决方案。我们发现将我们的应用程序扩展为工作流事件要容易得多，也快得多，同时在必要时仍保留关键的 LlamaIndex 功能，例如矢量存储和检索器。

下图显示了我们的工作流事件，我们将在本文后面对此进行更详细的解释。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/10/workflow_graph.png)

### 通过 Chainlit 的用户界面
Chainlit 包含多项有助于加快我们开发和部署的功能。它使用 chainlit.Step 装饰器支持进度指示器和步骤摘要，而 LlamaIndexCallbackHandler 可实现自动跟踪。我们为每个 LlamaIndex Workflow 事件使用了一个 Step 装饰器，以公开应用程序的内部工作方式，而不会让用户感到不知所措。

Chainlit 对企业身份验证和 PostgreSQL 数据层的支持对于生产也至关重要。

## 设置项目环境、依赖项和安装
要部署此项目，请克隆位于 [/NVIDIA/GenerativeAIExamples 的存储库](https://github.com/NVIDIA/GenerativeAIExamples/tree/main/community/routing-multisource-rag)并创建虚拟 Python 环境，在安装依赖项之前运行以下命令来创建和激活环境：


```bash
mkdir .venv
pip -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 配置
安装依赖项后，请确保在项目的顶级目录中有一个 .env 文件，其中包含以下值：

* NVIDIA_API_KEY：必需。您可以从 [build.nvidia.com](http://build.nvidia.com/) 获取 NVIDIA 服务的 API 密钥。
* PERPLEXITY_API_KEY。可选。如果未提供，则应用程序运行时不会使用 Perplexity 的搜索 API。要获取 Perplexity 的 API 密钥，请按照[说明](https://docs.perplexity.ai/home)进行操作。


### 项目结构
我们将项目代码组织成单独的文件：

* LlamaIndex 工作流 (workflow.py)：路由查询并聚合来自多个来源的响应。
* 文档摄取 (ingest.py)：将文档加载到 Milvus Lite 数据库中，这是开始使用 Milvus 而不使用容器的简单方法。Milvus Lite 的主要限制是低效的向量查找，因此当文档集合增长时，请考虑切换到专用集群。摄取模块使用 LlamaIndex 的 SimpleDirectoryReader 来解析和加载 PDF。
* Chainlit 应用程序 (chainlit_app.py)：Chainlit 应用程序包含由事件触发的函数，主要函数 (on_message) 在用户消息上激活。
* 配置 (config.py)：要尝试不同的模型类型，请编辑默认值。在这里，您可以选择不同的模型进行路由和聊天完成，以及每次完成时从聊天历史记录中使用的过去消息数量，以及 Perplexity 用于网络搜索和摘要的模型类型。

您还可以调整 prompts.py 中列出的提示以适合您的用例。

## 构建核心功能
此应用程序通过 Chainlit 集成了 LlamaIndex 和 NIM 微服务。为了展示如何实现此逻辑，我们将完成以下步骤：

* 创建用户界面
* 实现工作流事件
* 集成 NIM 微服务


### 创建用户接口
以下是该项目的实现方式，从 chainlit_app.py 中的 Chainlit 应用程序开始。在 set_starter 函数中创建一个 Starter 对象列表，以将初始问题预填充为可点击按钮。这些有助于指导用户采取可能的操作或提出问题，并可以将他们引导到特定功能。

在主函数中管理的主要聊天功能使用 cl.user_session 变量处理消息历史记录。Chainlit 显示对话历史记录不需要这样做，但使我们能够将状态保留在客户端而不是 LlamaIndex 对象中。

这种方法使原型设计更加直接，并有助于过渡到传统的用户前端后端应用程序，而有状态的 LlamaIndex ChatEngine 会使 REST API 部署复杂化。

当使用 Workflow.run 调用 Workflow 时，将通过 Workflow 触发一系列异步函数调用，这只需要当前用户查询和过去的聊天消息作为输入。生成流式响应后，使用 Chainlit 的 Message 类上的 stream_token 方法在用户界面中显示它。我们还添加了少量 HTML 样式来显示令牌计数和已用时间。

### 实现 Workflow 事件
RAG 逻辑包含在 Workflow.py 中的 QueryFlow 类中，由定义为 QueryFlow 方法的多个步骤组成。当其签名中的事件发生时，每个方法都会被触发。使用 nodes 属性在步骤之间传递节点列表是一种构建 Workflow 的简单方法。节点代表 LlamaIndex 内离散的信息单元。

以下是 Workflow 步骤：

* workflow_start：将用户查询和聊天记录添加到工作流的上下文 (ctx.data)，并使用 LLMTextCompletionProgram 在 RAG 和非 RAG 查询之间路由。根据结果，它会生成 RawQueryEvent（触发 RAG 逻辑）或 ShortcutEvent（触发即时响应合成）。
* rewrite_query：通过删除可能妨碍文档查找的指令关键字（如“电子邮件”和“表格”）来转换用户的查询以获得更好的搜索结果。它会触发 Milvus 检索和 Perplexity 搜索步骤的 TransformedQueryEvent。
* embed_query：为转换后的查询生成向量嵌入。
* milvus_retrieve：使用向量嵌入进行向量搜索。
* pplx_retrieve：使用 Perplexity 搜索 API 的 LlamaIndex 连接器获取 Web 搜索结果，并汇总为单个节点。
* collect_nodes：结合 Milvus 和 Perplexity 检索的结果。此步骤在两个检索事件完成后触发。在此处添加重新排序器可以优先考虑高价值节点。

    ```python
    ready = ctx.collect_events(
            qe,
            expected=[
                MilvusQueryEvent,
                PerplexityQueryEvent,
            ],
        )
 
        if ready is None:
            logger.info("Still waiting for all input events!")
            return None
    ```

* response_synthesis：使用过去的聊天历史上下文和检索到的文档构建提示字符串。我们手动形成此字符串，但也可以使用 LlamaIndex 模板。此步骤触发 StopEvent，结束 Workflow 事件并通过为 LLM 生成的每个标记生成 CompletionResponse 对象将响应返回给 Chainlit 应用程序。
总而言之，用户的查询首先经过路由步骤，其中 LLM 决定是否值得使用检索来查找文档以回答查询。如果不值得，则使用后续完成调用来生成答案。

当用户想要使用 LLM 执行不需要检索的任务（例如编辑电子邮件或总结一段现有文本）时，会触发此分支。如果选择检索，则用户的查询将转换为更适合搜索的形式。然后将其用于使用 NVIDIA 嵌入模型以及 Milvus 矢量存储对摄取的文档进行矢量查找。

然后，这些步骤返回的文本将使用 Perplexity 的 API 进行搜索，从而从 Web 中查找数据以形成答案。最后，这些结果用于响应合成。图 2 显示了使用 llama-index-utils-workflow 生成的图表。

### 集成 NIM 微服务

由于 llama-index-llms-nvidia 和 llama-index-embeddings-nvidia 软件包中提供的连接器，因此使用 NVIDIA NIM 微服务实现 LLM 和嵌入功能非常快捷。

由于 build.nvidia.com 提供了多种模型，我们可以选择小型、快速执行的模型 Meta 的 meta/llama-3.1-8b-instruct 来路由查询，同时还可以使用更大的模型 Mistral 的 mistralai/mistral-large-2-instruct，该模型具有出色的推理能力，可以生成最终响应。

对于高性能、大型模型，另一个不错的选择是 Meta 的 meta/llama-3.1-405b-instruct。

使用 NIM 微服务的一大优势是，如果您想要迁移到本地或自我管理的 LLM 推理部署，除了为 LLM 创建设置 base_url 参数之外，无需更改任何代码。否则，它是相同的！

您可以在 build.nvidia.com 上记录的面向公众的 NVIDIA 推理 API 或自我管理的 Llama 3.1 部署之间切换。这为在决定要自行管理和部署哪种类型的 NIM 微服务之前尝试几种模型进行原型设计提供了极大的灵活性。


## 额外功能
虽然其中一些功能超出了本文的范围，但这里还有一些易于添加以增强价值的功能：

* 通过使用视觉语言模型 (VLM) 读取表格、执行光学字符识别和为图像添加标题，实现多模式摄取。您可以在 Vision Language Models 中找到其中的许多功能。
* 使用 Chainlit 的 Postgres 连接器的用户聊天历史记录。要保留用户对话，您可以使用 chainlit.data_layer 的功能向 Chainlit 提供 PostgreSQL 连接详细信息。
* 使用基于 NVIDIA Mistral 的重新排名器进行 RAG 重新排名。
* 通过提示 LLM 使用 HTML 样式显示带有答案的超链接引文来添加引文。
* 错误处理和超时管理以提高可靠性。虽然像 Perplexity 这样的 API 可以有效地回答广泛的查询，但由于涉及的底层组件的复杂性，它们的执行时间可能会有很大的变化。设置合理的超时并在无法快速获得此类答案时从容恢复是迈向生产就绪应用程序的重要一步。































