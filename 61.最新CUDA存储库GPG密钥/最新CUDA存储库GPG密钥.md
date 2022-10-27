# 最新CUDA GPG Repository Key

![](cuda-image-16x9-1.jpg)

为了最好地确保 RPM 和 Debian 软件包存储库的安全性和可靠性，NVIDIA 将从 2022 年 4 月 27 日开始更新和轮换 apt、dnf/yum 和 zypper 软件包管理器使用的签名密钥。

如果您不更新存储库签名密钥，则在尝试从 CUDA 存储库访问或安装包时会出现包管理错误。

为确保继续访问最新的 NVIDIA 软件，请完成以下步骤。

## 删除过时的签名密钥
**Debian、Ubuntu、WSL**

```Bash
$ sudo apt-key del 7fa2af80
```

**Fedora, RHEL, openSUSE, SLES**


```Bash
$ sudo rpm --erase gpg-pubkey-7fa2af80*
```

## 安装新密钥
对于基于 Debian 的发行版，包括 Ubuntu，您还必须安装新软件包或手动安装新的签名密钥。

### 安装新的 cuda-keyring 包
为避免手动安装密钥步骤的需要，NVIDIA 提供了一个新的帮助程序包来自动安装 NVIDIA 存储库的新签名密钥。

将以下命令中的 `$distro/$arch` 替换为适合您的操作系统的值； 例如：

* debian10/x86_64
* debian11/x86_64
* ubuntu1604/x86_64
* ubuntu1804/cross-linux-sbsa
* ubuntu1804/ppc64el
* ubuntu1804/sbsa
* ubuntu1804/x86_64
* ubuntu2004/cross-linux-sbsa
* ubuntu2004/sbsa
* ubuntu2004/x86_64
* ubuntu2204/sbsa
* ubuntu2204/x86_64
* wsl-ubuntu/x86_64

**Debian, Ubuntu, WSL**

```Bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
$ sudo dpkg -i cuda-keyring_1.0-1_all.deb
```


## RPM 发行版
在全新安装中，作为 `dnf/yum/zypper` 的 Fedora、RHEL、openSUSE 或 SLES 会在存储库签名密钥更改时提示您接受新密钥。 出现提示时接受更改。

将以下命令中的 `$distro/$arch` 替换为适合您的操作系统的值； 例如：


* fedora32/x86_64
* fedora33/x86_64
* fedora34/x86_64
* fedora35/x86_64
* opensuse15/x86_64
* rhel7/ppc64le
* rhel7/x86_64
* rhel8/cross-linux-sbsa
* rhel8/ppc64le
* rhel8/sbsa
* rhel8/x86_64
* sles15/cross-linux-sbsa
* sles15/sbsa
* sles15/x86_64

对于基于 RPM 的发行版（包括 Fedora、RHEL 和 SUSE）的升级，您还必须运行以下命令。

**Fedora 和 RHEL 8**

```Bash
$ sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-$distro.repo
```

**RHEL 7**

```Bash
$ sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/$arch/cuda-rhel7.repo
```

**openSUSE and SLES**


```Bash
$ sudo zypper removerepo cuda-$distro-$arch
$ sudo zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-$distro.repo
```

## 使用容器
使用旧 NGC 基础容器构建的 CUDA 应用程序可能包含过时的存储库密钥。 如果您使用这些映像作为基础构建 Docker 容器并更新包管理器或安装其他 NVIDIA 包作为 Dockerfile 的一部分，则这些命令可能会像在非容器系统上一样失败。 要解决此问题，请将较早的命令集成到用于构建容器的 Dockerfile 中。

不使用包管理器安装更新的现有容器不受此密钥轮换的影响。

## 使用 NVIDIA GPU 运算符
如果您是 Ubuntu 发行版上 GPU Operator 的当前用户，您可能会受到 CUDA GPG 密钥轮换的影响，其中一些由 GPU Operator 管理的容器可能无法启动并出现以下错误：


```Bash
Stopping NVIDIA persistence daemon... Unloading 
NVIDIA driver kernel modules... Unmounting NVIDIA 
driver rootfs... Checking NVIDIA driver packages...
Updating the package cache... W: GPG error:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ InRelease: The 
following signatures couldn't be verified because
the public key is not available: NO_PUBKEY 
A4B469963BF863CC E: The repository 'https://
developer.download.nvidia.com/compute/cuda/repos/
ubuntu2004/x86_64 InRelease' is no longer signed.
```

NVIDIA 正在通过覆盖现有图像标签为驱动程序容器发布新图像。 您可以通过更新现有的 clusterPolicy 来拉取新图像来解决此错误：

```Bash
$ kubectl edit clusterpolicy
...
set  driver.imagePullPolicy=Always
```
此步骤导致 GPU Operator 拉取更新的图像。

GPU Operator 的新安装应该不受此更改的影响，并且不需要任何 clusterPolicy 更新。 如果您在 RHEL 或 OpenShift 上使用 GPU Operator，您也不会受到此更改的影响。

## 基于 Debian 的发行版的常见问题和解决方案
以下是我帮助人们解决的一些常见错误。 如果您看到此处未列出的错误，请在下面发表评论。

### 重复的 .list 
```Bash
{{E: Conflicting values set for option Signed-By regarding source
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /: 
/usr/share/keyrings/cuda-archive-keyring.gpg !=
E: The list of sources could not be read.}}
```
* 解决方案：如果您之前使用 add-apt-repository 启用 CUDA 存储库，则删除重复条目。

### 未注册新的 GPG 密钥

```Bash
{{Reading package lists...
W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64
InRelease: The following signatures couldn't be verified because the public key is not available:
NO_PUBKEY A4B46996 3BF863CC
E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64
InRelease' is no longer signed.}}
```
* 解决方案：请参阅“复制 .list ”安装 cuda-keyring 包或 3bf863cc 公钥的手动注册方法之一。

### 机器学习存储库

```Bash
{W: An error occurred during the signature verification.
The repository is not updated and the previous index files will be used.
GPG error: https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64
Release: The following signatures couldn't be verified because the public key is not available:
NO_PUBKEY F60F4B3D 7FA2AF80}}
```

* 解决方案：删除 NVIDIA 机器学习存储库条目，因为它不再更新。 CUDA 存储库中提供了较新版本的 cuDNN、NCCL 和 TensorRT。

### 文件大小

```C++
{{Packages.gz File has unexpected size (631054 != 481481). Mirror sync in progress? [IP: XXX.XXX.XXX.XXX 443]
Hashes of expected file:

* Filesize:481481 [weak]
* SHA256:8556d67c6d380c957f05057f448d994584a135d7ed75e5ae6bb25c3fc1070b0b
* SHA1:c5ea9556407a3b5daec4aac530cd038e9b490441 [weak]
* MD5Sum:a5513131dbd2d4e50f185422ebb43ac9 [weak]
* Release file created at: Mon, 25 Apr 2022 23:27:19 +0000
* E: Some index files failed to download. They have been ignored, or old ones used instead.}}
```
* 解决方案：向 NVIDIA 报告 CDN 问题。









