# 什么是LLM智能体

![](https://v.png.pub/imgs/2024/07/25/1337d02b56c0a267.png)


考虑一个大型语言模型 (LLM) 应用程序，该应用程序旨在帮助财务分析师回答有关公司业绩的问题。借助精心设计的检索增强生成 (RAG) 管道，分析师可以回答诸如“X 公司 2022 财年的总收入是多少？”之类的问题。经验丰富的分析师可以轻松地从财务报表中提取这些信息。


现在考虑一个问题，例如“23 财年第二季度财报电话会议的三个要点是什么？关注公司正在构建的技术护城河”。这是财务分析师希望在他们的报告中得到答案但需要投入时间来回答的问题类型。

我们如何开发解决方案来回答上述问题？很明显，这些信息需要的不仅仅是从财报电话会议中进行简单的查找。这种探究需要规划、量身定制的重点、记忆、使用不同的工具以及将复杂的问题分解为更简单的子部分。这些概念组合在一起，本质上就是我们所说的 LLM 智能体。

在这篇文章中，我介绍了 LLM 驱动的智能体，并讨论了什么是智能体以及企业应用程序的一些用例。有关更多信息，请参阅[构建您的第一个智能体应用程序](https://developer.nvidia.com/blog/building-your-first-llm-agent-application/)。在那篇文章中，我提供了一个生态系统演练，涵盖了构建 AI 智能体的可用框架以及任何尝试使用问答 (Q&A) 智能体的人的入门指南。


## 什么是 AI 智能体？
虽然对于 LLM 驱动的智能体没有一个被广泛接受的定义，但它们可以被描述为一个可以使用 LLM 推理问题、制定解决问题的计划并在一系列工具的帮助下执行计划的系统。

简而言之，智能体是一个具有复杂推理能力、记忆和执行任务手段的系统。

这种能力首先出现在 AutoGPT 或 BabyAGI 等项目中，这些项目无需太多干预即可解决复杂问题。为了更详细地描述智能体，下面是一个 LLM 驱动的智能体应用程序的一般架构（如下图）。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/agent-components-645x391.png)


智能体由以下关键组件组成（稍后将详细介绍）：

* 智能体核心
* 记忆模块
* 工具
* 规划模块
### 智能体核心
智能体核心是管理智能体核心逻辑和行为特征的中央协调模块。可以将其视为智能体的“关键决策模块”。我们还在其中定义：

* 智能体的总体目标：包含智能体的总体目标和目的。
* 执行工具：本质上是智能体可以访问的所有工具的简短列表或“用户手册”
* 如何使用不同规划模块的说明：有关不同规划模块的实用性以及在什么情况下使用哪个模块的详细信息。
* 相关记忆：这是一个动态部分，它在推理时填充过去与用户的对话中最相关的记忆项。使用用户提出的问题来确定“相关性”。
* 智能体人的角色（可选）：此角色描述通常用于使模型偏向于使用某些类型的工具或在智能体人的最终反应中注入典型的特质。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/Figure-2-Basic-template-of-how-the-different-modules-of-an-agent-are-assembled-in-its-core.png)



### 记忆模块
记忆模块在 AI 智能体中起着关键作用。记忆模块本质上可以被认为是智能体内部日志以及与用户交互的存储。

记忆模块有两种类型：

* 短期记忆：智能体尝试回答用户提出的单个问题时所经历的行为和想法的记录：智能体的“思路”。
* 长期记忆：用户和智能体之间发生的事件的行为和想法的记录。它是一本日志，包含跨越数周或数月的对话历史。

记忆需要的不仅仅是基于语义相似性的检索。通常，综合得分由语义相似性、重要性、新近度和其他特定于应用程序的指标组成。它用于检索特定信息。

### 工具
工具是智能体可以用来执行任务的明确定义的可执行工作流。通常，它们可以被视为专门的第三方 API。

例如，智能体可以使用 RAG 管道来生成上下文感知答案、使用代码解释器来解决复杂的编程任务、使用 API 在互联网上搜索信息，甚至使用任何简单的 API 服务（如天气 API 或即时通讯应用程序的 API）。

### 规划模块
复杂问题，例如分析一组财务报告以回答分层业务问题，通常需要细致入微的方法。借助 LLM 驱动的智能体，可以通过结合两种技术来处理这种复杂性：

* 任务和问题分解
* 反思或批评
### 任务和问题分解

复合问题或推断信息需要某种形式的分解。例如，这个问题是“NVIDIA 上次财报电话会议的三个要点是什么？”

回答这个问题所需的信息无法直接从一小时会议的记录中提取。但是，问题可以分解为多个问题主题：

* “讨论最多的技术转变是什么？”
* “是否存在任何业务阻力？”
* “财务结果如何？”

这些问题中的每一个都可以进一步分解为子部分。也就是说，专门的 AI 智能体必须指导这种分解。

### 反思或批评
ReAct、反思、思维链和思维图等技术已作为批评或基于证据的提示框架。它们已被广泛用于提高 LLM 的推理能力和反应能力。这些技术还可用于改进智能体生成的执行计划。


##  企业应用的智能体
虽然智能体的应用实际上是无限的，但以下是一些有趣的案例，可能会对许多企业产生巨大影响：

* “与数据对话”智能体
* 智能体群
* 推荐和体验设计智能体
* 定制的 AI 作者智能体
* 多模式智能体
### “与数据对话”智能体
“与数据对话”不是一个简单的问题。有很多挑战是简单的 RAG 管道无法解决的：

* 源文档的语义相似性
* 复杂的数据结构，如表格
* 缺乏明显的上下文（并非每个块都包含其来源的标记）
* 用户提出的问题的复杂性
* ……还有更多

例如，回到之前的收益电话会议记录示例（2023 年第三季度 | 2024 年第一季度）。您如何回答这个问题，“2023 年第三季度和 2024 年第一季度之间数据中心的收入增长了多少？”要回答这个问题，你基本上必须分别回答三个问题（即，我们需要一个规划模块）：

* 2023 年第三季度的数据中心收入是多少？

* 2024 年第一季度的数据中心收入是多少？

* 两者有什么区别？

在这种情况下，你需要一个可以访问规划模块的智能体，该模块可以进行问题分解（生成子问题并搜索答案，直到更大的问题得到解决）、RAG 管道（用作工具）来检索特定信息，以及内存模块来准确处理子问题。在 LLM 驱动的智能体：构建您的第一个智能体应用程序帖子中，我详细介绍了这种情况。

## 智能体群
智能体群可以理解为一组智能体，它们共同努力在单一环境中共存，彼此协作解决问题。分散的智能体生态系统非常类似于多个“智能”微服务，它们协同使用来解决问题。

像 Generative Agents 和 ChatDev 这样的多智能体环境在社区中非常受欢迎（下图）。为什么？像 ChatDev 这样的框架使您能够组建一个由工程师、设计师、产品管理、CEO 和智能体组成的团队，以低成本构建基本软件。像 Brick Breaker 或 Flappy Bird 这样的热门游戏的原型制作成本低至 50 美分！

有了智能体群，您可以为数字公司、社区甚至整个城镇填充应用程序，例如经济研究的行为模拟、企业营销活动、物理基础设施的 UX 元素等

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/multiple-agents-chatdev.png)


目前，没有 LLM，这些应用程序无法模拟，而且在现实世界中运行成本极高。

### 用于推荐和体验设计的智能体
互联网依靠推荐工作。由智能体驱动的对话式推荐系统可用于打造个性化体验。

例如，考虑电子商务网站上的 AI 智能体，它可以帮助您比较产品并根据您的一般要求和选择提供建议。还可以构建完整的礼宾式体验，多个智能体协助最终用户浏览数字商店。选择观看哪部电影或预订哪个酒店房间等体验可以以对话的形式打造，而不仅仅是一系列决策树式对话！

### 定制的 AI 作者智能体
另一个强大的工具是拥有一个个人 AI 作者，可以帮助您完成诸如共同撰写电子邮件或为您准备时间敏感的会议和演示等任务。常规创作工具的问题在于，不同类型的材料必须根据不同的受众进行量身定制。例如，投资者推介的措辞必须与团队演示不同。

智能体可以利用您以前的工作。然后，让智能体根据您的个人风格塑造智能体生成的宣传，并根据您的特定用例和需求定制工作。这个过程通常太过细致，不适合一般的 LLM 微调。

### 多模式智能体
如果仅使用文本作为输入，您无法真正“与数据对话”。所有提到的用例都可以通过构建可以消化各种输入（例如图像和音频文件）的多模式智能体来增强。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/schrodinger-non-text-response.png)



























