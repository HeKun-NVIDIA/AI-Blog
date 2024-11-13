# 使用新的 NVIDIA Isaac Foundation 模型和工作流程创建、设计和部署机器人应用程序

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/Isaac-robotics.gif)


机器人技术的应用正在智能制造设施、商业厨房、医院、仓库物流和农业领域等各种环境中迅速扩展。该行业正在转向智能自动化，这需要增强机器人功能，以执行感知、绘图、导航、负载处理、物体抓取和复杂的装配任务等功能。

人工智能在这一演变中发挥着关键作用，提高了机器人的性能。通过集成 NVIDIA AI 加速，机器人可以更精确、更高效地处理复杂任务，在各种应用中充分发挥其潜力。

在 COMPUTEX 上，我们宣布了多项新功能，以帮助机器人专家和工程师构建智能机器人。这些包括：

* [NVIDIA Isaac Perceptor](https://developer.nvidia.com/isaac/perceptor)，一种用于自主移动机器人 (AMR) 和自动导引车 (AGV) 的新参考工作流程。
* [NVIDIA Isaac Manipulator](https://developer.nvidia.com/isaac/manipulator) 为工业机械臂提供了新的基础模型和参考工作流程。
* [NVIDIA Jetson for Robotics](https://developer.nvidia.com/embedded/jetpack)，在 NVIDIA JetPack 6.0 中进行了新的更新。
* [NVIDIA Isaac Sim 4.0](https://developer.nvidia.com/isaac/sim) 带来了 NVIDIA Isaac Lab，这是一款用于机器人学习的轻量级应用程序。


## NVIDIA Isaac Perceptor
AMR 和 AGV 对于装配线效率、物料搬运和医疗保健物流至关重要。当这些机器人在复杂且非结构化的环境中导航时，感知和响应周围环境的能力变得至关重要。

Isaac Perceptor 建立在 NVIDIA Isaac 机器人操作系统 (ROS) 之上，使原始设备制造商 (OEM)、货运服务提供商、软件供应商和 AMR 生态系统能够加速机器人技术的发展。团队可以为移动机器人配备感知功能，以便在非结构化环境中成功导航和避障。

Isaac Perceptor 的早期合作者包括汽车制造商、工业机器人制造公司和机器人解决方案提供商的仓储/内部物流行业领导者，例如 ArcBest、比亚迪电子、Gideon、KION、Kudan、idealworks、RGo 和 Teradyne Robotics。


## Isaac Perceptor 的主要功能
Isaac Perceptor 提供多种功能，为基于 AI 的自主移动机器人提供多摄像头、3D 环视功能。

### 基于 AI 的多摄像头深度感知
Isaac Perceptor 以 30 Hz 的频率每秒处理每台摄像头 16.5M 个深度点。立体视差是根据来自立体摄像头的时间同步图像对计算得出的，用于为场景生成深度图像或点云。高效的半监督深度神经网络 (ESS DNN) 为基于 DNN 的立体视差提供了 GPU 加速包。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/isaac-forklift.png)



### 多摄像头视觉惯性里程计
Isaac ROS Visual SLAM 提供 ROS 2 软件包，用于视觉同步定位和地图绘制 (VSLAM) 和视觉里程计 (VO)。它基于 NVIDIA CUDA Visual SLAM (cuVSLAM) 库，可在无特征环境中导航时提供强大的导航功能，翻译误差小于 1%。

在具有稀疏视觉特征或重复模式的环境中导航是 VSLAM 解决方案面临的一个众所周知的挑战。这可以通过融合来自多个视点的输入来缓解。在最新更新中，cuVSLAM 结合了来自多个立体摄像头的并发视觉里程计估计。

我们的测试表明有显著的改进。机器人使用多个摄像头始终能够实现其导航目标，而使用单个摄像头时，实现目标的几率不到 25%。


|VO method|	Runtime|
|----|----|
|cuVSLAM|	5 ms|
|FRVO, S-PTAM	|30 ms|
|ORB-SLAM2|	60 ms|
![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/06/Isaac-Perception.gif)

### 实时多摄像头体素网格映射
Isaac Perceptor 的核心是 nvblox，这是一个 CUDA 加速的 3D 重建库，可以识别最远五米外的障碍物，以提供 2D 成本图并在 300 毫秒内更新它们。

Isaac ROS nvblox 提供 ROS 2 软件包，用于 3D 场景重建和导航的本地障碍物成本图生成。此软件包可用于静止环境以及有人和移动物体的场景。

此版本中的新功能是多摄像头支持，可使用最多三个 HAWK 摄像头扩大覆盖范围，提供约 270° 的视野。

有关更多信息，请访问 [Isaac ROS nvblox](https://isaac_ros.gitlab-master-pages.nvidia.com/isaac_ros_docs/repositories_and_packages/isaac_ros_nvblox/index.html) 文档。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/Nvblox.gif)


### NVIDIA Nova Orin 开发套件
此开发套件采用 NVIDIA Jetson AGX Orin，支持多达六个摄像头，包括多达三个立体摄像头和三个鱼眼摄像头，摄像头内延迟低于 100 微秒。

立体摄像头的分辨率为每台 2MP，视野为 110X70，适用于 3D 占用网格映射、深度感知、视觉里程计和人员检测。从 Segway 或 Leopard Imaging 购买 Nova Orin 开发套件即可使用 Isaac Perceptor。

Isaac Perceptor 有一个参考图，支持此开发套件上多达三个立体摄像头。通过与 ROS 2 软件包的增强模块化，此版本还具有与 Nova Carter 参考机器人上的 Nav2 的参考集成。

### 增强了与摄像头和传感器的兼容性
Isaac Perceptor 为与摄像头和传感器合作伙伴的集成提供了增强的支持。Orbbec 成功将其 Gemini 335L 摄像头与 NVIDIA Isaac Perceptor 组件集成。这种集成在 NVIDIA Jetson AGX Orin 上使用 Isaac ROS Visual SLAM 和 Nvblox 进行了演示。

LIPS 还成功将其 AE450 摄像头与 Isaac Perceptor 组件 Nvblox 集成。


## NVIDIA Isaac Manipulator
Isaac Manipulator 是 NVIDIA 加速库和 AI 模型的工作流程。它使开发人员能够将 AI 加速带入机械臂或操纵器，从而无缝感知、理解和与环境交互。

其基础模型和加速库可以作为独立模块或作为解决方案开发中的整个工作流程进行集成。除了独立的模块化组件外，还为开发人员提供了示例工作流程（ROS 2 启动脚本），这些工作流程结合了 Isaac Manipulator 组件，以实现完整的端到端参考集成。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/Isaac-Manipulator-workflow.png)


Isaac Manipulator 的早期合作者包括机器人开发平台公司、OEM 和 ISV/SI，包括 Intrinsic（Alphabet 旗下公司）、西门子、Solomon、Techman Robot、Teradyne Robotics、Vention 和 Yaskawa。

## Isaac Manipulator 的主要功能

Isaac Manipulator 提供 AI 功能，以加速机械臂的开发。

### cuMotion 可实现更快的路径规划

这款 GPU 加速的运动规划器有助于缩短周期时间。cuMotion 可作为 MoveIt 2 运动规划框架的插件使用，该框架是由国际社区开发并由 PickNik Robotics 领导的开源项目。

cuMotion 可同时在多个种子上运行轨迹优化并返回最佳解决方案。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/cuMotion.gif)

Solomon 是先进视觉和机器人解决方案领域的领导者，也是 Isaac Manipulator 的早期合作者。与传统算法相比，他们的箱体拾取系统通过 Isaac Manipulator cuMotion 增强，使路径规划速度提高了 8 倍，路径奇异性发生率降低了 50%。

|指标|改进率 (%)|
|----|----|
|成功率改进 |346.43|
|移动时间减少| 55.50|
|轨迹长度减少 |42.27|
|轨迹规划时间减少 |816.66|

### FoundationPose
[FoundationPose](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/foundationpose/index.html) 是一种新的统一基础模型，用于单次 6D 姿势估计和新物体跟踪。该模型旨在在遇到以前未见过的物体的应用中以高精度工作，而无需进行微调。

[FoundationPose](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/foundationpose/index.html) 目前在 2023 年 BOP 排行榜上名列前茅，用于对未见过的物体进行 6D 定位。它对遮挡、快速运动和纹理和比例等各种物体属性具有很强的鲁棒性，可确保在各种场景中提供可靠的性能。开发人员可以从任何角度生成物体的真实视图。从 [GitHub 获取 Foundation Pose 模型](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/index.html)。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/FoundationPose.gif)

### SyntheticaDETR
SyntheticaDETR 是一组基于实时检测变换器 (DETR) 的模型，用于使用 NVIDIA Omniverse 生成的合成数据进行单次图像空间物体检测。它通过使用变换器编码器-解码器架构一次性预测所有物体，从而实现了一种比传统物体检测器更有效的方法。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/SyntheticaDETR.gif)


SyntheticaDETR 经过合成数据和真实数据训练，在 YCB-Video 数据集上可见物体的 2D 检测 BOP 排行榜上名列前茅（平均精度为 0.885，平均召回率为 0.903）。

这些模型还可以将物体检测为姿势估计器（如 NVIDIA FoundationPose）的 2D 边界框感兴趣区域。下载 SyntheticaDETR 模型并下载 Isaac Manipulator。

### NVIDIA JetPack 6.0
NVIDIA Isaac ROS 3.0 与 JetPack 6.0 兼容，并受所有 NVIDIA Jetson Orin 模块和开发套件支持。

NVIDIA Jetson Platform Services 即将推出模块化、API 驱动的服务，以更快、更轻松地构建生成式 AI 和机器人应用程序。这些预构建和可定制的服务旨在加速 NVIDIA Jetson Orin 系统模块上的 AI 应用程序开发。

## NVIDIA Isaac Sim 4.0
使用 Isaac Sim，开发人员可以使用业界领先的传感器和机器人类型测试生成合成数据和多样化的虚拟复杂测试环境。这可以实现高度逼真的模拟，以实时同时测试数千个机器人。

### NVIDIA Isaac Lab
Isaac Lab 是一款基于 Isaac Sim 平台构建的轻量级参考应用程序，在机器人基础模型训练中发挥着关键作用。它支持强化学习、模仿学习和迁移学习。它可以训练各种机器人实例，供开发人员探索设计和功能。

新版本还提供了易于使用的功能，包括与兼容性检查器的 VSCode 集成、对强化学习的多 GPU 支持、通过 RTX 传感器平铺渲染实现的性能改进、优化的缓存和着色器管理。

Isaac Sim 中的其他新功能包括：

* 易于使用的 PIP 安装和用于导入机器人等的向导。

* 性能提高，合成数据生成 (SDG) 速度提高 80%
支持 COCO 格式和用于姿势估计的自定义编写器的新 SDG 格式。
* ROS 2 推出支持端到端工作流程和更好的图像发布性能。
* 更多内置机器人支持：包括 Universal Robots UR20 和 UR30 以及 Boston Dynamics Spot。还有许多人形机器人，包括 1X Neo、Unitree H1、Agility Digit、Fourier Intelligence GR1、Sanctuary A1 Phoenix 和 XiaoPeng PX5。

















































