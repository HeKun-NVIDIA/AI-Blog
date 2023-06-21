# RAN-in-the-Cloud：为 5G RAN 提供云经济性

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/02/connected-cityscape.jpg)


5G 部署在全球范围内一直在加速。 许多电信运营商已经推出了5G服务并正在快速扩张。 除了电信运营商之外，企业也对使用 5G 建立私有网络产生了浓厚的兴趣，这些私有网络利用了更高的带宽、更低的延迟、网络切片、毫米波和 CBRS 频谱。


5G 的出现恰逢其时。 在过去的二十年里，云计算已经成熟，成为开发人员构建应用程序的首选平台。 云提供了许多优势，包括成熟的软件工具、自动化和编排、业务敏捷性和较低的总拥有成本 (TCO)。

此外，每个细分市场（工业机器人、云游戏、智能城市、安全、零售、自动驾驶、智能农业）的应用都越来越多地使用人工智能 (AI) 来实现变革性体验。 5G、云计算和人工智能的融合将在未来十年推动许多创新。

NVIDIA Aerial SDK 是构建虚拟化无线电接入网络 (vRAN) 的关键技术基础。 它是一种软件定义的完整 5G Layer1 (L1) 卸载，在 NVIDIA GPU 中实现为内联加速。 它还实现了所有 3GPP 和 O-RAN 兼容接口。 包含复杂信号处理算法的 L1 软件在 CUDA C/C++ 中实现，可以轻松优化 L1 算法，实现新功能，并为 5G 演进和 6G 的 RAN 应用提供面向未来的验证。 NVIDIA Aerial SDK 作为具有 E2E 云原生架构的模块化微服务实施，并由 Kubernetes 使用标准的 ORAN SMO 兼容接口进行管理。



## 从 CloudRAN 到 RAN-in-the-Cloud
最近有很多关于 CloudRAN 的讨论。 作为加速计算平台和云计算领域的行业领导者，NVIDIA 一直走在 CloudRAN 创新的前沿。 许多行业领导者使用术语 CloudRAN 来表示无线电接入网络 (RAN) 的云原生实现。

虽然云原生技术是筹码，但重要的问题是 CloudRAN 是否等同于使用云原生技术。 我们认为事实并非如此。 我们相信真正的云 RAN 将所有计算元素（vDU、vCU 和 dUPF）都部署在云端。 因此，术语 RAN-in-the-Cloud：一个 5G 无线电接入网络作为服务完全托管在多租户云基础设施中。

为什么这种区别很重要，RAN-in-the-Cloud 的动机是什么？ 首先，RAN 构成了电信运营商最大的资本支出和运营支出。 它也是最未充分利用的资源，大多数无线电基站的使用率通常低于 50%。 将 RAN 计算迁移到云中可以带来云计算的所有优势：共享云基础设施中的池化和更高利用率，从而最大程度地减少电信运营商的资本支出和运营支出。

具有 GPU 加速器的 COTS 平台不仅可以加速 5G RAN； 它还可以加速边缘人工智能应用程序。 电信运营商和企业越来越多地使用 NVIDIA GPU 服务器来加速边缘 AI 应用程序。 这提供了一种简单的途径，可以使用相同的 GPU 资源来加速 AI 应用程序之外的 5G RAN 连接。 这反过来又降低了 TCO，并为建立企业 5G 网络提供了最佳途径。

多年来，云软件、工具和技术已经成熟，除了可靠性、可观察性和服务保证之外，还带来了大规模自动化、降低能耗、弹性计算和按需自动缩放等优势。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/02/Unified-accelerated-data-center.png)


值得注意的是，一些供应商正在为 RAN L1 卸载设计基于专用集成电路 (ASIC) 的固定功能加速器卡。 基于这些基于 ASIC 的加速器构建的 RAN 类似于固定功能设备。 它只能进行 RAN L1 处理，在不使用时是浪费资源。

具有通用 GPU 加速服务器的 NVIDIA Aerial SDK 提供了一个真正的多服务、多租户平台，可用于 5G RAN、企业 AI 和部署在云中的其他边缘应用程序，具有上述所有优势。


## 云原生作为 RAN-in-the-Cloud 的基础
随着行业加速 5G 部署，实现 5G 的全部商业价值需要可扩展且灵活的解决方案。 将 RAN 软件与硬件分离并使软件在云中可用和可部署有可能推动更快的创新和新的增值服务。

云原生 vDU/vCU RAN 软件套件旨在完全开放和自动化部署和整合操作，支持私有、公共或混合云基础设施上的 3GPP 和 O-RAN 接口。 它利用了云原生架构的优势，包括水平和垂直扩展、自动修复和冗余。 它还针对移动网络演进进行了优化设计，包括 6G 等下一代无线电技术。

NVIDIA Aerial SDK 云原生架构有助于将 RAN 功能实现为由 Kubernetes 编排和管理的容器中的微服务。 模块化软件支持：

* 改进了软件升级、发布和修补的粒度并提高了速度
* 遵循 DevOps 原则的独立生命周期管理，具有持续集成和持续交付 (CI/CD)
* 独立扩展不同的 RAN 微服务元素
* 应用程序级可靠性、可观察性和服务保证
* 通过网络自动化简化操作和维护

为了真正的云原生 RAN 体验，云、边缘平台和网络都需要发展。 在我们看来，许多要求对于云原生容器化 RAN 软件堆栈的商业部署至关重要，包括：

* 时间同步
* CPU 亲和性和隔离
* 拓扑管理和特征发现
* 多个网络接口
* 高性能数据平面和加速硬件
* 低延迟、QoS保证、高吞吐量
* 远程分布式部署
* 零接触配置
* 用于加速器设备的 Kubernetes 运算符框架和生产就绪运算符


[NVIDIA GPU Operator](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/gpu-operator) 使用 Kubernetes 中的运算符框架来自动管理配置 GPU 所需的所有 NVIDIA 软件组件。 这些组件包括设备驱动程序（以启用 CUDA）、GPU 的 Kubernetes 设备插件、NVIDIA 容器运行时、自动节点标签、基于数据中心 GPU 管理器 (DCGM) 的监控等。

GPU Operator 使 Kubernetes 集群的管理员能够像管理集群中的 CPU 节点一样管理 GPU 节点。 管理员可以依赖 CPU 和 GPU 节点的标准操作系统映像，然后依靠 GPU 操作员为 GPU 提供所需的软件组件，而不是为 GPU 节点提供特殊的操作系统映像。

它利用 Kubernetes CRD 和操作员 SDK，管理与网络相关的组件，以实现与 RDMA 和 NVIDIA GPUDirect 的快速网络连接，以处理 Kubernetes 集群中的工作负载。 网络运营商与 GPU 运营商合作，在兼容系统上启用 GPU 直接 RDMA。 网络运营商的目标是管理网络相关组件，同时在 Kubernetes 集群中执行 RDMA 和 GPU 直接 RDMA 工作负载。

NVIDIA Aerial SDK 基于微服务和云原生架构构建，为构建和部署 5G RAN-in-the-Cloud 奠定了坚实的基础。

## 在云中构建、部署和管理
O-RAN 联盟计划将传统无线电基站分解为 RRU、vDU 和 vCU 实例，它们之间具有定义明确的接口，从而形成了一个更大的生态系统，提供了供应商选择。 此外，云原生容器化软件支持由 Kubernetes 和 SMO 管理的可组合和自动化 RAN。 云化和托管完整的 RAN 作为云中的服务需要什么？

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/02/RAIN-in-the-cloud-vision.png)

部署 5G 的经济性一直具有挑战性。 与前几代无线技术相比，5G 正在推动 RAN 资本支出大幅增长。 预计在未来 5 年内，蜂窝基站的数量将增加近一倍。 因此，RAN 资本支出占总体 TCO 的份额从 45-50% 增加到 65%。 更多详情，请参见无线回传演进和5G时代移动网络成本演进。

此外，众所周知，RAN 传统上是针对峰值容量进行配置的，这导致宝贵的计算资源严重未得到充分利用。 突发和时间相关的流量意味着许多传统 RAN 站点的平均运行容量使用率低于 25%。 如果 RAN 可以托管在云中，池化的好处可以减少与节能相关的运营支出并提高使用率。 此外，可以为其他应用程序和工作负载以真正的类似云的方式重新配置未使用的资源。

仅在美国，将 420,000 个基站总数中的 50% 迁移到 GPU 加速云可能会为电信运营商带来重要的新收入机会。 当 RAN 利用率低且 GPU 闲置时，它们可用于多租户云环境中的企业 AI、视频服务和其他边缘应用程序。 这可能会在全球范围内带来数十亿美元的新收入机会。

上图显示了通过使用 NVIDIA GPU 的加速计算基础架构构建的数据中心如何加速许多应用程序，从而提供云经济性和最佳 TCO。

带有 NVIDIA Base Command Platform 和 NVIDIA Fleet Command 软件的 NVIDIA AI Enterprise 使企业能够在 NVIDIA GPU 云中运行 AI 应用程序，利用适用于各个垂直领域的所有预构建和强化软件。 5G 连接作为容器化解决方案与使用相同基础设施的其他 AI 应用程序一起运行对于企业来说将非常强大。 这将改变世界对无线连接的看法。 5G 将成为完全基于云的服务，可以按需部署。 这就是 RAN-in-the-Cloud 的本质。


## 使用 NVIDIA 构建您的 5G RAN-in-the-Cloud

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/02/Five-key-technologies-enable-RAN-in-the-Cloud-1.png)

全新 NVIDIA Spectrum SN3750-SX 开放式以太网交换机是 RAN-in-the-Cloud 解决方案的关键组件。 它基于 NVIDIA Spectrum-2 以太网 ASIC，是有史以来第一款软件定义的 xHaul 交换机，能够提供电信数据中心所需的前传、中传和回传网络。

该交换机的一个关键功能是它可以动态编程以将流量路由到部署在数据中心任何服务器上的任何 vDU，支持自动扩展和按需 RAN 部署。 它是第一个将在同一基础设施上运行电信和人工智能所需的所有功能结合在一起的交换机。 SN3750-SX 支持先进的定时协议，例如电信级精确时间协议 (PTP)、同步以太网 (SyncE) 和 PPS（每秒数据包数），以及动态 RU/DU 映射。

为了实现 AI 训练，该交换机支持低延迟 200G 带宽以实现最高吞吐量。 Spectrum ASIC 带来了创新功能，例如 RoCE（融合以太网上的 RDMA）和自适应路由，所有这些都处于最高网络规模。 需要注意的是，许多应用程序（例如虚拟世界和 AR/VR）都需要支持 PTP 的数据中心。 这将为 RAN-in-the-Cloud 用例铺平道路。 一些网络规模公司已经在他们的数据中心支持 PTP。

配备 NVIDIA A100 Tensor Core GPU 和 NVIDIA BlueField DPU 的 NVIDIA A100X 融合加速器支持完整的内联 5G RAN 卸载。 这为从 4T4R 到大规模 MIMO 32T32R 和 64T64R 的一系列配置提供了市场领先的性能（以每瓦单元密度和每瓦 MHz 层数衡量）。

NVIDIA 正在与各种生态系统合作伙伴合作，以确保其他 O-RAN 软件组件，如 SMO（服务管理和编排）、RIC（RAN 智能控制器）、xApps 和 rApps 针对 NVIDIA Aerial SDK 进行了优化，并为 RAN-in- 云部署。 这些组件仍处于早期开发阶段，但将成为关键的差异化因素，因为它们将 AI 用于 RAN 自动化和可编程性。 虽然 RAN-in-the-Cloud 需要一些时间才能成熟，但我们相信 NVIDIA 将以 NVIDIA GPU 加速平台为基础站在这一创新的前沿。

## 总结
RAN-in-the-Cloud 是未来。 这是无线市场的自然演变和下一步。 使用云原生技术构建的 vRAN 是必要的第一步。 实现 5G RAN 的云经济并推动 5G 与边缘 AI 应用的共同创新需要拥抱 RAN-in-the-Cloud。 NVIDIA Aerial SDK 提供可扩展的云原生软件架构，作为 RAN-in-the-Cloud 的基础技术。

最后，需要注意的是，RAN 转型才刚刚开始。 使用 AI 来优化复杂的信号处理算法将在未来几年释放出一系列全新的创新。 GPU 加速平台是让您的投资永不过时的最佳方法。 如果您想与我们合作构建创新的 RAN-in-the-Cloud 解决方案，请联系我们。 有关详细信息，请参阅 NVIDIA AI-on-5G 平台。















































