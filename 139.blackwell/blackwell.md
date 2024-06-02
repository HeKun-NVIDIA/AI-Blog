# NVIDIA Blackwell架构技术

在AI和大型语言模型（LLMs）迅速发展的领域中，追求实时性能和可扩展性至关重要。从医疗保健到汽车行业，组织正深入探索生成性AI和加速计算解决方案的领域。对生成性AI解决方案的需求激增，促使企业需要适应不断增长的模型规模和复杂性。

请进入NVIDIA Blackwell GPU架构，这是世界上最大GPU，专为处理数据中心规模的生成性AI工作流程而设计，其能效是前一代NVIDIA Hopper GPU的25倍。

本技术简报详细介绍了NVIDIA Blackwell的优势，包括下一代超级芯片Grace Blackwell GB200，以及下一代高性能HGX系统，NVIDIA HGX B200和HGX B100。

## NVIDIA Blackwell GPU和超级芯片概览
大型语言模型（LLMs）需要巨大的计算能力才能实现实时性能。LLMs的计算需求也意味着更高的能源消耗，因为需要越来越多的内存、加速器和服务器来适应、训练和从这些模型中推断。旨在实时推理的组织必须应对这些挑战。

NVIDIA Blackwell架构和产品系列旨在满足不断增长的AI模型规模和参数的需求，提供了一长串新创新，包括新的第二代Transformer引擎。

NVIDIA Blackwell架构以David H. Blackwell的名字命名，他是一位了不起且鼓舞人心的美国数学家和统计学家，以Rao-Blackwell定理而闻名，并在概率论、博弈论、统计学和动态规划方面做出了许多贡献和进步。

有了NVIDIA Blackwell产品，每个企业都可以使用和部署最先进的LLMs，具有可负担的经济性，通过生成性AI的优势优化他们的业务。同时，NVIDIA Blackwell产品也使得生成性AI模型的下一个时代成为可能，支持具有实时性能的多万亿参数模型，这在没有Blackwell创新的情况下是无法实现的。

## NVIDIA Blackwell架构创新
Blackwell架构为生成性AI和加速计算引入了突破性的进展。新的第二代Transformer引擎，以及更快更宽的NVIDIA NVLink互连，将数据中心推向了一个新的时代，与上一代架构相比，性能提高了数个数量级。

NVIDIA Confidential Computing技术的进一步进步提高了大规模实时生成性AI推理的安全性，而不会影响性能。NVIDIA Blackwell的新型压缩引擎结合Spark RAPIDS™库提供了无与伦比的数据库性能，推动数据分析应用。NVIDIA Blackwell的多项进步建立在加速计算技术的几代基础上，定义了生成性AI的下一个篇章，具有无与伦比的性能、效率和规模。

## 新型AI超级芯片
Blackwell架构使用了2080亿个晶体管，比NVIDIA Hopper GPU多2.5倍以上，并使用了专为NVIDIA定制的TSMC 4NP工艺，Blackwell是迄今为止建造的最大的GPU。NVIDIA Blackwell在单个芯片上实现了最高的计算能力，达到了20 petaFLOPS。

## 第二代Transformer引擎
Blackwell引入了新的第二代Transformer引擎。第二代Transformer引擎使用定制的Blackwell Tensor Core技术，结合TensorRT-LLM和Nemo Framework的创新，加速了LLMs和Mixture-of-Experts（MoE）模型的推理和训练。

## 高性能的保密计算和安全AI
生成性AI为企业带来了巨大的潜力。优化收入、提供商业洞察和帮助生成内容只是其中的一些好处。但是，对于需要在私人数据上训练它们、可能受到隐私法规约束或包含专有信息的企业来说，采用生成性AI可能会很困难。

NVIDIA Confidential Computing能力将可信执行环境（TEE）从CPU扩展到GPU。NVIDIA Blackwell上的保密计算旨在为LLMs和其他敏感数据提供最快、最安全、可验证（基于证据）的保护。

## 第五代NVLink和NVLink交换机
解锁E级计算和万亿参数AI模型的全部潜力取决于服务器集群中每个GPU之间迅速、无缝的通信需求。

## 压缩引擎
数据分析和数据库工作流程传统上依赖于CPU进行计算，速度慢且繁琐。加速的数据科学可以显著提高端到端分析的性能，加快价值生成和洞察力的生成时间，同时降低成本。数据库，包括Apache Spark，在处理、处理和分析大量数据以进行数据分析中发挥着关键作用。Blackwell的新型专用压缩引擎可以以高达800GB/s的速率解压缩数据，结合GB200中使用的一个GPU的8TB/s的HBM3e（高带宽内存）以及Grace CPU的高速NVLink-C2C（芯片到芯片）互连，加速了数据库查询的完整流程，为数据分析和数据科学提供了最高性能。支持最新的压缩格式，如Lz4、Snappy和Deflate，NVIDIA Blackwell的性能比CPU快18倍，比NVIDIA H100 Tensor Core GPU快6倍。

## RAS引擎
Blackwell架构通过专用的可靠性、可用性和可维护性（RAs）引擎增加了智能弹性，以识别可能早期发生的故障，以最小化停机时间。NVIDIA的AI驱动的预测管理能力不断监控硬件和软件中的数千个数据点，以预测和拦截停机和效率低下的来源。这建立了智能弹性，节省了时间、能源和计算成本。

## NVIDIA GB200超级芯片和GB200 NVL72
NVIDIA GB200 Grace Blackwell超级芯片通过NVIDIA NVLink@-C2C互连连接两个高性能的NVIDIA Blackwell Tensor Core GPU和一个NVIDIA Grace CPU，该互连为两个GPU提供了900 GB/s的双向带宽。

## NVIDIA GB200 NVL72
NVIDIA GB200 NVL72集群在机架规模设计中连接了36个GB200超级芯片（36个Grace CPU和72个Blackwell GPU）。GB200 NVL72是一个液冷的、机架规模的72-GPU NVLink域，可以作为一个巨大的GPU来提供比前一代快30倍的实时万亿参数LLM推理。

## 下一代大型语言模型的实时推理
GB200 NVL72引入了尖端能力和第二代Transformer引擎，显著加速了LLM推理工作负载，使得资源密集型应用（如多万亿参数语言模型）的实时性能成为可能。GB200 NVL72与H100相比提供了30倍的速度提升，Tc0降低了25倍，能源使用量也降低了25倍，对于像GPT-MoE-1.8T这样的大型模型，使用相同数量的GPU（见图5）。这一进步是通过新一代Tensor Core实现的，它们引入了包括FP4在内的新精度。此外，GB200利用NVLink和液冷创建了一个单一的巨大的72-GPU机架，可以克服通信瓶颈。

## AI训练性能的新水平
GB200包括一个更快的Transformer引擎，具有FP8精度，并与NVIDIA Hopper GPU相比，为像GPT-MoE-1.8T这样的大型语言模型提供了4倍更快的训练性能。这一性能提升提供了9倍的机架空间减少和3.5倍的Tc0和能源使用量减少。这一突破得到了第五代NVLink（它实现了1.8 TB/s的GPU到GPU互连和更大的72-GPU NVLink域）、InfiniBand网络和NVIDIA Magnum I/O软件的补充。这些共同确保了企业和广泛的GPU计算集群的有效可扩展性。

## 加速数据处理和基于物理的模拟
GB200以其紧密耦合的CPU和GPU，在数据处理和工程设计模拟的加速计算中带来了新的机会。

## 可持续计算
计算密度和计算能力正在推动从空气冷却向液冷的转变。在数据中心内外使用液体而不是空气有许多积极影响，包括每个机架更高的性能、减少冷却的水消耗，以及允许数据中心在更高的环境空气温度下运行，这进一步降低了能源消耗。

## 加速网络平台用于生成性AI
GB200 NVL72作为一个单一的、极其强大的计算单元，需要强大的网络来实现最佳应用性能。与NVIDIA Quantum-X800 InfiniBand、Spectrum-X800以太网和BlueField-3 DPU配合使用，GB200在大规模AI数据中心提供了前所未有的性能、效率和安全性。

## NVIDIA Blackwell HGX
NVIDIA Blackwell HGX B200和HGX B100包括用于生成性AI、数据分析和高性能计算的同样突破性的进展，并扩展了HGX以包括Blackwell GPU。


Blackwell架构相比Hopper架构在多个方面实现了显著的技术提升，以下是一些关键的改进点：

1. 晶体管数量和计算能力：Blackwell架构使用了2080亿个晶体管，这是Hopper GPU晶体管数量的2.5倍以上。Blackwell架构在单个芯片上实现了20 petaFLOPS的计算能力，这是迄今为止最高的。

2. 第二代Transformer引擎：Blackwell引入了新的第二代Transformer引擎，使用定制的Blackwell Tensor Core技术和TensorRT-LLM以及Nemo Framework的创新，以加速大型语言模型（LLMs）和专家混合模型（MoE）的推理和训练。

3. 新的精度格式：Blackwell Tensor Cores引入了新的精度格式，包括社区定义的微缩放格式，提供了高准确性和更大的吞吐量。Blackwell Transformer引擎利用先进的动态范围管理算法和称为微张量缩放的细粒度缩放技术，优化性能和准确性，并启用了FP4 AI。

4. 压缩引擎：Blackwell架构包括一个专用的压缩引擎，可以以高达800GB/s的速率解压缩数据，与GB200中的8TB/s的HBM3e内存和Grace CPU的高速NVLink-C2C互连相结合，显著加速数据库查询。

5. 第五代NVLink和NVLink交换机：Blackwell架构的NVLink性能是Hopper架构中第四代NVLink的两倍，每个方向的有效带宽达到50 GB/s。NVLink交换机ASIC和基于它的交换机使得可以扩展到576个GPU，以加速万亿参数和多万亿参数AI模型的性能。

6. RAS引擎：Blackwell架构增加了一个专用的可靠性、可用性和可维护性（RAs）引擎，用于识别可能早期发生的故障，以最小化停机时间。NVIDIA的AI驱动的预测管理能力持续监控硬件和软件中的数千个数据点，预测并拦截停机和效率低下的来源。

7. 能效：Blackwell架构在保持高性能的同时，提供了比Hopper架构更高的能效，为数据中心规模的生成性AI工作流程提供了高达25倍的能效提升。

8. 保密计算：Blackwell架构引入了首个支持TEE-I/O的GPU，提供了最高性能的保密计算解决方案，同时保护AI知识产权，并安全地启用保密AI训练、推理和联邦学习。

这些技术提升使得Blackwell架构能够更好地处理和支持日益增长的AI模型规模和复杂性，特别是在实时性能和可扩展性方面，满足了当前和未来AI应用的需求。

Blackwell架构的第二代Transformer引擎通过以下方式提高了AI模型训练效率：

1. 定制的Tensor Core技术：Blackwell架构的Tensor Core使用了专为大型语言模型（LLMs）和Mixture-of-Experts（MoE）模型设计的第二代技术。这些Tensor Core结合了TensorRT-LLM和Nemo Framework的创新，优化了模型的推理和训练过程。

2. 新的精度格式：Blackwell的Tensor Core引入了新的精度格式，包括社区定义的微缩放格式，这些新精度提供了高准确性和更大的吞吐量。这种精度的引入使得模型可以在保持性能的同时，使用更少的计算资源进行训练。

3. 微张量缩放技术：Blackwell Transformer引擎利用了先进的动态范围管理算法和微张量缩放技术，这是一种细粒度的性能和准确性优化方法。这种技术使得FP4 AI的性能翻倍，同时将参数带宽翻倍至HBM内存，并使得每个GPU能够处理的下一代模型大小翻倍。

4. 专家并行技术：第二代Transformer引擎与Nemo Framework和Megatron-Core结合，使用了新的专家并行技术。这些技术与其他并行技术相结合，并利用第五代NVLink，为前所未有的模型性能提供了支持。降低精度格式为大规模训练打开了进一步加速的可能性。

5. 量化和自定义内核：TensorRT-LLM中的创新，包括量化到4位精度，以及具有专家并行映射的自定义内核，使得当今的MoE模型能够实现实时推理，使用更少的硬件和能源，同时降低了成本。

通过这些技术的提升，Blackwell架构的第二代Transformer引擎使得企业能够使用和部署最先进的MoE模型，优化他们的业务，并利用生成性AI的好处。同时，它也为训练和实时推理超过10万亿参数的模型提供了支持，这在没有Blackwell架构的创新之前是无法实现的。



## 结论
生成性AI已经将计算提升到了一个新的时代，这个时代的特点是拥有10万亿或更多参数的AI模型的惊人能力。当AlexNet在2012年开启了AI热潮时，它使用了6000万个参数。短短十多年后，今天的复杂性已经超过了160,000倍。

这些新模型现在可以找到治疗癌症的方法，预测极端天气事件，自动化机器人进行工业检查，并在每个行业中解锁新的经济机会。然而，充分发挥它们全部潜力的旅程面临着挑战，尤其是模型训练所需的大量计算资源和时间。

新的极大规模LLMs结合实时推理的需求揭示了规模、部署和运营方面的更多挑战和复杂性。

NVIDIA Blackwell是一个千载难逢的平台，拥有有效训练和推断这些模型所需的力量和能效，并将成为生成性AI时代的基础。Blackwell架构将被部署到万亿美元市场中，并将实时使用这些新的巨型模型民主化。训练这些模型需要NVIDIA Blackwell的exaFLOPs计算能力。部署它们需要数十个Blackwell GPU作为一个单一的统一GPU工作。