# NVIDIA 全面转向开源 GPU 内核模块

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/hpc-featured.jpg)

借助 R515 驱动程序，NVIDIA 于 2022 年 5 月发布了一组 Linux GPU 内核模块，作为具有双重 GPL 和 MIT 许可的开源模块。初始版本针对数据中心计算 GPU，GeForce 和 Workstation GPU 处于 alpha 状态。

当时，NVIDIA宣布，后续版本将提供更强大、功能更全面的 GeForce 和 Workstation Linux 支持，NVIDIA 开放内核模块最终将取代闭源驱动程序。

NVIDIA GPU 共享通用的驱动程序架构和功能集。您的台式机或笔记本电脑的同一驱动程序在云中运行世界上最先进的 AI 工作负载。对NVIDIA来说，做到恰到好处非常重要。

两年过去了，NVIDIA利用开源 GPU 内核模块实现了同等甚至更好的应用程序性能，并增加了大量新功能：

* 异构内存管理 (HMM) 支持
* 机密计算
* NVIDIA Grace 平台的一致内存架构
* 还有更多

现在，NVIDIA正处于完全过渡到开源 GPU 内核模块的正确时机，NVIDIA将在即将发布的 R560 驱动程序版本中进行这一改变。

## 支持的 GPU
并非所有 GPU 都与开源 GPU 内核模块兼容。

对于 NVIDIA Grace Hopper 或 NVIDIA Blackwell 等尖端平台，您必须使用开源 GPU 内核模块。这些平台不支持专有驱动程序。

对于 Turing、Ampere、Ada Lovelace 或 Hopper 架构的较新 GPU，NVIDIA 建议切换到开源 GPU 内核模块。

对于 Maxwell、Pascal 或 Volta 架构的较旧 GPU，开源 GPU 内核模块与您的平台不兼容。继续使用 NVIDIA 专有驱动程序。

对于在同一系统中混合部署较旧和较新 GPU，请继续使用专有驱动程序。

如果您不确定，NVIDIA 提供了一个新的检测帮助脚本来帮助指导您选择哪个驱动程序。有关更多信息，请参阅本文后面的使用安装帮助脚本部分。

## 安装程序更改
一般来说，所有安装方法安装的驱动程序的默认版本都是从专有驱动程序切换到开源驱动程序。有几个特定场景值得特别注意：

* 带有 CUDA 元包的包管理器
* 运行文件
* 安装帮助脚本
* 包管理器详细信息
* 适用于 Linux 的 Windows 子系统
* CUDA 工具包

### 使用带有 CUDA 元包的包管理器
当您使用包管理器（而不是 .run 文件）安装 CUDA Toolkit 时，安装元包存在并且很常用。通过安装顶级 cuda 包，您可以安装 CUDA Toolkit 和相关驱动程序版本的组合。例如，通过在 CUDA 12.5 发布时间范围内安装 cuda，您将获得专有的 NVIDIA 驱动程序 555 以及 CUDA Toolkit 12.5。

下图显示了此包结构。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/07/cuda-package-installation-before-12-6.png)

以前，使用开源 GPU 内核模块意味着您无法使用顶级元包。您必须安装特定于发行版的 NVIDIA 驱动程序开放包以及您选择的 cuda-toolkit-X-Y 包。

从 CUDA 12.6 版本开始，流程有效地切换了位置（下图）。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/07/cuda-package-installation-after-12-6.png)


## 使用运行文件
如果您使用 .run 文件安装 CUDA 或 NVIDIA 驱动程序，安装程序会查询您的硬件并自动安装最适合您系统的驱动程序。您还可以使用 UI 切换按钮在专有驱动程序和开源驱动程序之间进行选择，任您选择。

如果您通过 CUDA .run 文件进行安装并使用 ncurses 用户界面，您现在会看到类似以下菜单：

```bash
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Driver                                                                  │
│   [ ] Do not install any of the OpenGL-related driver files                  │
│   [ ] Do not install the nvidia-drm kernel module                            │
│   [ ] Update the system X config file to use the NVIDIA X driver             │
│ - [X] Override kernel module type                                            │
│      [X] proprietary                                                         │
│      [ ] open                                                                │
│   Change directory containing the kernel source files                        │
│   Change kernel object output directory                                      │
│   Done                                                                       │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘
```
如果您通过驱动程序 .run 文件进行安装，您会看到类似的选择（下图）。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/07/runfile-interactive-selection.png)

您还可以使用命令行传递覆盖，以便在没有用户界面的情况下进行安装，或者如果您使用 Ansible 等自动化工具。

```bash
# sh ./cuda_12.6.0_560.22_linux.run --override --kernel-module-type=proprietary
 
# sh ./NVIDIA-Linux-x86_64-560.run --kernel-module-type=proprietary
```

### 使用安装帮助脚本
如前所述，如果您不确定为系统中的 GPU 选择哪种驱动程序，NVIDIA 创建了一个帮助脚本来指导您完成选择过程。

要使用它，首先使用包管理器安装 nvidia-driver-assistant 包，然后运行脚本：
```bash
$ nvidia-driver-assistant
```

### 包管理器详细信息
为了获得一致的体验，NVIDIA 建议您使用包管理器来安装 CUDA Toolkit 和驱动程序。但是，不同发行版使用哪些包管理系统或包的结构的具体细节可能会因您的特定发行版而异。

本节概述了各种平台所需的具体细节、注意事项或迁移步骤。

apt：基于 Ubuntu 和 Debian 的发行版
运行以下命令：
```bash
$ sudo apt-get install nvidia-open
```

要在 Ubuntu 20.04 上使用 cuda 元包进行升级，请先切换到打开内核模块：
```bash
$ sudo apt-get install -V nvidia-kernel-source-open

$ sudo apt-get install nvidia-open
```

### dnf：Red Hat Enterprise Linux、Fedora、Kylin、Amazon Linux 或 Rocky Linux
运行以下命令：
```bash
$ sudo dnf module install nvidia-driver:open-dkms
```

要在基于 dnf 的发行版上使用 cuda 元包进行升级，必须禁用模块流：
```bash
$ echo "module_hotfixes=1" | tee -a /etc/yum.repos.d/cuda*.repo
$ sudo dnf install --allowerasing nvidia-open
$ sudo dnf module reset nvidia-driver

```
### zypper：SUSE Linux Enterprise Server 或 OpenSUSE
运行以下命令之一：
```bash
# default kernel flavor
$ sudo zypper install nvidia-open
```
```bash
# azure kernel flavor (sles15/x86_64)
$ sudo zypper install nvidia-open-azure
```
```bash
# 64kb kernel flavor (sles15/sbsa) required for Grace-Hopper
$ sudo zypper install nvidia-open-64k
```

### 软件包管理器摘要
为简化起见，我们以表格形式压缩了软件包管理器建议。驱动程序版本 560 和 CUDA Toolkit 12.6 之后的所有版本都将使用这些打包约定。

|Distro	|Install the latest 	|Install a specific release |
|----|----|----|
|Fedora/RHEL/Kylin|	dnf module install nvidia-driver:open-dkms	|dnf module install nvidia-driver:560-open|
|openSUSE/SLES	|zypper install nvidia-open{-azure,-64k}	|zypper install nvidia-open-560{-azure,-64k}|
|Debian	|apt-get install nvidia-open	|apt-get install nvidia-open-560|
|Ubuntu	|apt-get install nvidia-open	|apt-get install nvidia-open-560|

### 适用于 Linux 的 Windows 子系统
适用于 Linux 的 Windows 子系统 (WSL) 使用主机 Windows 操作系统中的 NVIDIA 内核驱动程序。您不应专门在此平台中安装任何驱动程序。如果您使用的是 WSL，则无需进行任何更改或操作。

### CUDA 工具包
CUDA 工具包的安装通过包管理器保持不变。运行以下命令：
```bash
$ sudo apt-get/dnf/zypper install cuda-toolkit
```

### 更多信息
有关如何安装 NVIDIA 驱动程序或 CUDA 工具包的更多信息，包括如果您目前无法迁移到开源 GPU 内核模块，如何确保安装专有驱动程序，请参阅[ CUDA 安装指南中的驱动程序安装](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#driver-installation)。

![](1.png)

































































