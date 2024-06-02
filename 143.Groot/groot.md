#! https://zhuanlan.zhihu.com/p/689063998
# NVIDIA 宣布推出适用于人形机器人的 GR00T 项目基础模型和主要 Isaac 机器人平台更新


Isaac 机器人平台现为开发人员提供新的机器人训练模拟器、Jetson Thor 机器人计算机、生成式 AI 基础模型以及 CUDA 加速感知和操作库

![](https://s3.amazonaws.com/cms.ipressroom.com/219/files/20242/project-gr00t-humanoid.jpg)

GTC — NVIDIA 今天宣布推出 GR00T 项目，这是一个用于人形机器人的通用基础模型，旨在进一步推动机器人技术和具体人工智能领域的突破。

作为该计划的一部分，该公司还推出了一款用于基于 NVIDIA Thor 片上系统 (SoC) 的人形机器人的新型计算机 Jetson Thor，以及对 NVIDIA Isaac™ 机器人平台的重大升级，包括 用于模拟和人工智能工作流程基础设施的生成式人工智能基础模型和工具。

NVIDIA 创始人兼首席执行官黄仁勋表示：“为通用人形机器人构建基础模型是当今人工智能领域最令人兴奋的问题之一。” “这些使能技术正在汇聚在一起，让世界各地领先的机器人专家在人工通用机器人领域取得巨大飞跃。”

GR00T代表通用机器人00技术，由GR00T驱动的机器人将被设计为通过观察人类行为来理解自然语言和模仿动作——快速学习协调性、灵活性和其他技能，以便导航、适应现实世界并与现实世界互动。 在他的 GTC 主题演讲中，展示了几个这样的机器人来完成各种任务。

## 专为人形机器人打造
Jetson Thor 是一个新的计算平台，能够执行复杂的任务并与人和机器安全、自然地交互。 它具有针对性能、功耗和尺寸进行优化的模块化架构。

该 SoC 包括基于 NVIDIA Blackwell 架构的下一代 GPU，其变压器引擎可提供 800 teraflops 的 8 位浮点 AI 性能，以运行 GR00T 等多模式生成 AI 模型。 凭借集成的功能安全处理器、高性能 CPU 集群和 100GB 以太网带宽，它显着简化了设计和集成工作。

NVIDIA 正在为 1X Technologies、Agility Robotics、Apptronik、Boston Dynamics、Figure AI、Fourier Intelligence、Sanctuary AI、Unitree Robotics 和 XPENG Robotics 等领先的人形机器人公司构建全面的 AI 平台。

“我们正处于历史的转折点，像 Digit 这样以人为中心的机器人将永远改变劳动力。 现代人工智能将加速发展，为像 Digit 这样的机器人在日常生活的各个方面为人们提供帮助铺平道路。”Agility Robotics 联合创始人兼首席机器人官 Jonathan Hurst 说道。 “我们很高兴与 NVIDIA 合作，投资计算、模拟工具、机器学习环境和其他必要的基础设施，以实现机器人成为日常生活一部分的梦想。”

Sanctuary AI 联合创始人兼首席执行官 Geordie Rose 表示：“嵌入式人工智能不仅有助于解决人类面临的一些最大挑战，还将创造目前超出我们能力或想象的创新。” “如此重要的技术不应孤立构建，这就是我们优先考虑 NVIDIA 等长期合作伙伴的原因。”

## Isaac 平台的主要更新
GR00T 使用的 Isaac 工具能够为任何环境中的任何机器人实例创建新的基础模型。 这些工具包括用于强化学习的 [Isaac Lab ](https://developer.nvidia.com/isaac-sim#isaac-lab)和计算编排服务 [OSMO](https://developer.nvidia.com/blog/scale-ai-enabled-robotics-development-workloads-with-nvidia-osmo/)。

具体的人工智能模型需要大量的真实和合成数据。 新的 Isaac Lab 是一款基于 Isaac Sim 构建的 GPU 加速、轻量级、性能优化的应用程序，专门用于运行数千个机器人学习并行模拟。

为了跨异构计算扩展机器人开发工作负载，OSMO 协调分布式环境中的数据生成、模型训练和软件/硬件在环工作流程。

NVIDIA [还发布了 Isaac Manipulator 和 Isaac Perceptor](https://blogs.nvidia.com/blog/isaac-generative-ai-manufacturing-logistics/)——一系列机器人预训练模型、库和参考硬件。

[Isaac Manipulator](https://developer.nvidia.com/isaac/manipulator) 为机械臂提供最先进的灵活性和模块化 AI 功能，并拥有强大的基础模型和 GPU 加速库集合。 它在路径规划方面提供高达 80 倍的加速，零样本感知提高了效率和吞吐量，使开发人员能够自动执行更多新的机器人任务。 早期的生态系统合作伙伴包括安川、优傲机器人、泰瑞达旗下公司、PickNik Robotics、Solomon、READY Robotics 和 Franka Robotics。

[Isaac Perceptor](https://developer.nvidia.com/isaac/perceptor) 提供多摄像头、3D 环绕视觉功能，这些功能越来越多地用于制造和履行操作中采用的自主移动机器人，以提高效率和工人安全，并降低错误率和成本。 早期采用者包括 ArcBest、比亚迪和凯傲集团，因为他们的目标是在物料搬运操作等方面实现新的自主水平。

新的 Isaac 平台功能预计将在下个季度推出。 了解有关 [GR00T 项目](https://developer.nvidia.com/project-gr00t)的更多信息。











































