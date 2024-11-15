# 什么是Vector Database(向量数据库)？
向量数据库是向量嵌入的有组织的集合，可以随时创建、读取、更新和删除。向量嵌入将文本或图像等数据块表示为数值。

## 什么是嵌入模型(Embedding Model)？
嵌入模型将各种数据（例如文本、图像、图表和视频）转换为数字向量，从而在多维向量空间中捕捉其含义和细微差别。嵌入技术的选择取决于应用需求，平衡语义深度、计算效率、要编码的数据类型和维数等因素。
![](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vector-database/_jcr_content/root/responsivegrid/nv_container_1795650/nv_image_copy.coreimg.100.630.jpeg/1710829331227/vector-database-embedding-1920x1080.jpeg)
将向量映射到多维空间可以对向量的语义相似性进行细致入微的分析，从而显著提高搜索和数据分类的准确性。嵌入模型在使用 AI 聊天机器人、大型语言模型 (LLM) 和带有向量数据库的检索增强生成 (RAG) 的 AI 应用中起着至关重要的作用，以及搜索引擎和许多其他用例。


## 嵌入模型如何与向量数据库一起使用？
当私有企业数据被提取时，它会被分块，创建一个向量来表示它，并且数据块及其对应的向量与可选元数据一起存储在向量数据库中以供以后检索。

![](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vector-database/_jcr_content/root/responsivegrid/nv_container_1795650_1499237187/nv_image_copy.coreimg.100.630.jpeg/1710829331582/vector-database-embedding-vector-1920x1080.jpeg)


在收到来自用户、聊天机器人或 AI 应用程序的查询后，系统会对其进行解析并使用嵌入模型来获取表示提示部分内容的向量嵌入。然后使用提示的向量在向量数据库中进行语义搜索，以找到完全匹配或前 K 个最相似的向量及其相应的数据块，这些数据块在发送到 LLM 之前会放入提示的上下文中。LangChain 或 LlamaIndex 是流行的开源框架，用于支持创建 AI 聊天机器人和 LLM 解决方案。流行的 LLM 包括 OpenAI GPT 和 Meta LlaMA。流行的向量数据库包括 Pinecone 和 Milvus 等。两种最流行的编程语言是 Python 和 TypeScript。

## 什么是向量数据库中的相似性搜索？
相似性搜索，也称为向量搜索、向量相似性或语义搜索，是指 AI 应用程序根据指定的相似性度量从数据库中有效检索与给定查询的向量嵌入在语义上相似的向量的过程，例如：

* 欧几里得距离(Euclidean distance)：测量点之间的直接距离。适用于聚类或对整体差异很重要的密集特征集进行分类。
* 余弦相似性(Cosine similarity)：关注向量之间的角度。非常适合文本处理和信息检索，根据方向而不是传统距离捕获语义相似性。
* 曼哈顿距离(Manhattan distance)：计算笛卡尔坐标中绝对差异的总和。适用于网格结构中的寻路和优化问题。适用于稀疏数据。

相似性测量指标可以高效检索 AI 聊天机器人、推荐系统和文档检索中的相关项目，通过利用数据中的语义关系来通知生成 AI 过程并执行自然语言处理 (NLP)，从而增强用户体验。

## 向量搜索中的聚类算法是什么？
聚类算法根据共同特征将向量组织成有凝聚力的组，从而促进向量数据库中的模式识别和异常检测。

![](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vector-database/_jcr_content/root/responsivegrid/nv_container_1795650_1610178080/nv_image_copy.coreimg.100.630.jpeg/1710829332102/vector-database-cluster-1920x1080.jpeg)

此过程不仅有助于通过减小数据集大小来压缩数据，而且还揭示了潜在的模式，为各个领域提供了宝贵的见解。

* K 均值：根据质心接近度将数据拆分为 K 个簇。适用于大型数据集。需要预定义簇数。
* DBSCAN 和 HDBSCAN：根据密度形成簇，区分异常值。适应复杂形状而无需指定簇数。
* 层次聚类：通过聚集合并或分割数据点来创建簇树。适用于层次数据可视化。
* 谱聚类：利用相似矩阵特征值进行降维。适用于非线性可分数据。
* 均值漂移：通过查找密度函数最大值来识别簇。可灵活处理簇形状和大小。无需预定义簇数。

算法方法的多样性适用于不同的数据类型和聚类目标，强调了聚类在从 RAG 架构中的向量数据中提取有意义信息方面的多功能性和关键重要性。

## 索引在向量数据库中的作用是什么？
向量数据库中的索引在提高高维数据空间内搜索操作的效率和速度方面起着至关重要的作用。鉴于向量数据库中存储的数据的复杂性和数量，索引机制对于快速定位和检索与查询最相关的向量至关重要。以下是向量数据库中索引的主要功能和优势的细分：

* 高效的搜索操作：索引结构（例如 K-D 树、VP 树或倒排索引）通过以减少在整个数据集中执行详尽搜索的需要的方式组织数据，从而实现更快的搜索操作。
* 可扩展性：随着数据量的增长，索引有助于保持性能水平，确保搜索操作可以随着数据库的大小而有效地扩展。
* 减少延迟：通过促进更快的搜索，索引显著减少了查询与其相应结果之间的延迟，这对于需要实时或近实时响应的应用程序至关重要。
* 支持复杂查询：高级索引技术通过高效导航高维空间来支持更复杂的查询，包括最近邻搜索、范围查询和相似性搜索。
* 优化资源使用：有效的索引可最大限度地减少搜索所需的计算资源，从而节省成本并提高系统可持续性，尤其是在基于云或分布式的环境中。

总之，索引是向量数据库性能和功能的基础，使它们能够快速有效地管理和搜索大量复杂的高维数据。这种能力对于从推荐系统和个性化引擎到人工智能驱动的分析和内容检索系统等各种应用都至关重要。RAPIDS cuVS 提供 GPU 加速，可将索引构建时间从几天缩短到几小时。

## 什么是向量数据库中的查询处理？
向量数据库的查询处理器与传统关系数据库中使用的架构截然不同。向量数据库中查询处理的效率和精度取决于复杂的步骤，包括解析、优化和执行查询。

![](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vector-database/_jcr_content/root/responsivegrid/nv_container_1795650_359986625/nv_image_copy.coreimg.100.630.jpeg/1710829332623/vector-database-query-1920x1080.jpeg)

处理诸如最近邻识别和相似性搜索之类的复杂操作需要使用高级索引结构以及并行处理算法（例如 cuVS 中的 CAGRA），以进一步增强系统有效管理大规模数据的能力。

这种综合方法可确保向量数据库能够及时准确地响应用户查询，从而保持快速的响应时间和高水平的信息检索准确性。处理用户查询以收集其嵌入，然后使用嵌入有效地查询向量数据库以获得语义相似的嵌入（向量）。

## 什么影响向量数据库的可扩展性？
向量数据库中的 GPU 加速（例如通过 RAPIDS cuVS 等库）对于处理不断增加的数据量和计算需求至关重要，而不会影响性能。它确保这些数据库能够适应 AI 和大数据分析中日益增长的复杂性，采用两种主要策略：API 背后的垂直和水平扩展。

垂直扩展通过升级计算资源来增强容量，从而允许在同一台机器内处理更大的数据集和更复杂的操作。水平扩展将数据和工作负载分布在多个服务器上，使系统能够管理更大的请求量并确保高可用性以满足不断变化的需求。

优化的算法和并行处理（尤其是通过 GPU 进行）是实现高效可扩展性的关键。这些方法通过简化数据处理和检索任务来最大限度地减少系统负载。GPU 具有并行处理能力，尤其有价值，可以加速数据密集型计算，并使数据库在跨节点扩展时保持高性能水平。


## 什么是向量数据库中的数据规范化？
向量数据库中的数据规范化涉及将向量调整为统一的比例，这是确保基于距离的操作（例如聚类或最近邻搜索）的一致性能的关键步骤。为了实现标准化，人们使用常用技术，例如最小-最大缩放，将数据值调整为指定范围（通常为 0 到 1 或 -1 到 1）和 Z 分数规范化，将数据集中在平均值附近，标准差为 1。

这些方法对于使来自不同来源或维度的数据具有可比性至关重要，从而提高了对数据执行的分析的准确性和可靠性。这种规范化过程在机器学习应用中尤其重要，它有助于消除由特征尺度变化引起的偏差，从而显著提高模型的预测性能。

通过确保所有数据点都以一致的尺度进行评估，数据规范化有助于提高向量数据库中存储数据的质量，从而有助于获得更有效、更有洞察力的机器学习结果。


## 哈希在向量数据库中是如何使用的？
哈希是向量数据库运行的基础概念。它将高维数据转换为简化的固定大小格式，优化向量数据库中的向量索引和检索过程。局部敏感哈希 (LSH) 等技术对于有效的近似最近邻搜索、降低计算复杂性和提高查询处理速度特别有价值。哈希在管理大规模高维空间、确保高效的数据访问以及支持广泛的机器学习和相似性检测任务方面起着至关重要的作用。

## 什么是向量数据库中的降噪？
降低向量数据库中的噪声对于提高各种应用（包括相似性搜索和机器学习任务）中的查询准确性和性能至关重要。有效的降噪不仅可以提高存储在这些数据库中的数据的质量，还可以促进更准确、更有效地检索信息。为了实现这一点，可以采用一系列技术，每种技术都针对噪声和数据复杂性的不同方面进行量身定制。

这些方法侧重于简化、规范化和细化数据，同时采用旨在学习和过滤噪音的模型。选择正确的技术组合取决于数据的性质和数据库应用程序的特定目标。

* 降维和规范化：PCA 和向量规范化等技术有助于去除不相关的特征和缩放向量，减少噪音并提高查询性能。

* 特征选择和数据清理：识别关键特征并预处理数据以删除重复项和错误，从而简化数据集，专注于相关信息。

* 去噪模型：利用去噪自动编码器从嘈杂的数据中重建输入，教会模型忽略噪音，从而提高数据质量。

* 向量量化和聚类：这些方法将向量组织成具有相似特征的组，从而减轻数据中异常值和方差的影响。

* 嵌入细化：对于特定领域的应用程序，使用额外的训练或改造等技术细化嵌入可以提高向量相关性并降低噪音。

## 查询扩展如何在向量数据库中发挥作用？
向量数据库中的查询扩展通过将其他相关术语合并到查询中来提高搜索查询的有效性，从而扩大搜索范围以实现更全面的数据检索。此技术调整查询向量以捕获更广泛的语义相似性，更紧密地与用户意图保持一致并实现更彻底的文档检索。通过这样做，查询扩展显著提高了搜索结果的精度和范围，使其成为在向量数据库中更高效、更有效地发现信息的关键策略。

## 如何对向量数据库进行数据可视化？
在向量数据库中，数据可视化对于将高维数据转换为易于理解的视觉效果、帮助分析和决策至关重要。主成分分析 (PCA)、t 分布随机邻域嵌入 (t-SNE) 和均匀流形近似和投影 (UMAP) 等技术对于降低维度和揭示复杂数据中隐藏的模式至关重要。这一过程对于发现原始数据中不明显的宝贵见解、更清晰地传达复杂的数据模式以及促进战略性的数据驱动决策至关重要。

## 如何在向量数据库中处理数据稀疏性？
稀疏矩阵表示和专门的处理技术提高了深度学习应用中的存储效率和计算性能，确保向量数据库能够有效地管理和分析稀疏数据。

![](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vector-database/_jcr_content/root/responsivegrid/nv_container_1795650_452904768/nv_image_copy.coreimg.100.630.jpeg/1710829334025/vector-database-sparsity-1920x1080.jpeg)


解决数据稀疏性问题需要有效处理主要由零值组成的向量，这种情况在高维数据集中很常见。压缩稀疏行 (CSR) 和压缩稀疏列 (CSC) 等稀疏矩阵格式旨在通过仅存储非零元素来有效存储和操作主要为零的数据。
目标技术包括针对稀疏矩阵优化的算法，这些算法可显着减少计算开销和内存使用量，从而实现更快的处理和分析。这些方法对于机器学习和数据科学处理高维数据至关重要，它们可以提高效率并在数据处理和分析任务中实现低延迟。

## 如何确保向量数据库中的数据完整性？
确保向量数据库中的数据完整性至关重要，重点是通过错误检测、强大加密、数据管理和定期审核等复杂措施来保障准确性、一致性和安全性。NVIDIA NeMo™ 放大了这一过程，提供了专门的 AI 工具来增强数据的管理和完整性。该框架的功能扩展到创建和管理 AI 模型，以增强数据库可靠性，这是进行详细数据分析和推进机器学习应用程序的基石。通过 NeMo，NVIDIA 倡导在向量数据库中导航和分析复杂数据集所必需的基础信任和可靠性。
























































