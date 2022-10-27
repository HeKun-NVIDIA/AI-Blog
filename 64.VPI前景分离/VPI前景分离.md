# (含源码)利用NVIDIA VPI之前景分离

![](fgmask_cpu%2000_00_00-00_00_30.gif)


NVIDIA 视觉编程接口 (`VPI: Vision Programming Interface`) 是 NVIDIA 的计算机视觉和图像处理软件库，使您能够实现在 NVIDIA Jetson 嵌入式设备和独立的GPU 上可用的不同硬件后端上加速的算法。

库中的一些算法包括过滤方法、透视扭曲、时间降噪、直方图均衡、立体视差和镜头失真校正。 VPI 提供易于使用的 Python 绑定以及 C++ API。

除了与 OpenCV 接口外，VPI 还能够与 PyTorch 和其他基于 Python 的库进行互操作。 在这篇文章中，我们将通过基于 PyTorch 的目标检测和跟踪示例向您展示这种互操作性如何工作。 有关详细信息，请参阅[视觉编程接口 (VPI) 页面](https://developer.nvidia.com/embedded/vpi)和[视觉编程接口](https://docs.nvidia.com/vpi/)文档。


下面的示例从输入视频源中获取帧，在当前图像上运行算法，然后计算前景部分。输出前景蒙版将保存到视频文件中。

```Python
import cv2
 import sys
 import vpi
 import numpy as np
 from argparse import ArgumentParser
  
 # ----------------------------
 # Parse command line arguments
  
 parser = ArgumentParser()
 parser.add_argument('backend', choices=['cpu','cuda'],
                     help='Backend to be used for processing')
  
 parser.add_argument('input',
                     help='Input video to be denoised')
  
 args = parser.parse_args();
  
 if args.backend == 'cuda':
     backend = vpi.Backend.CUDA
 else:
     assert args.backend == 'cpu'
     backend = vpi.Backend.CPU
  
 # -----------------------------
 # Open input and output videos
  
 inVideo = cv2.VideoCapture(args.input)
  
 fourcc = cv2.VideoWriter_fourcc(*'MPEG')
 inSize = (int(inVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
 fps = inVideo.get(cv2.CAP_PROP_FPS)
  
 outVideoFGMask = cv2.VideoWriter('fgmask_python'+str(sys.version_info[0])+'_'+args.backend+'.mp4',
                                  fourcc, fps, inSize)
  
 outVideoBGImage = cv2.VideoWriter('bgimage_python'+str(sys.version_info[0])+'_'+args.backend+'.mp4',
                                   fourcc, fps, inSize)
  
 #--------------------------------------------------------------
 # Create the Background Subtractor object using the backend specified by the user
 with backend:
     bgsub = vpi.BackgroundSubtractor(inSize, vpi.Format.BGR8)
  
 #--------------------------------------------------------------
 # Main processing loop
 idxFrame = 0
 while True:
     print("Processing frame {}".format(idxFrame))
     idxFrame+=1
  
     # Read one input frame
     ret, cvFrame = inVideo.read()
     if not ret:
         break
  
     # Get the foreground mask and background image estimates
     fgmask, bgimage = bgsub(vpi.asimage(cvFrame, vpi.Format.BGR8), learnrate=0.01)
  
     # Mask needs to be converted to BGR8 for output
     fgmask = fgmask.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA);
  
     # Write images to output videos
     with fgmask.rlock_cpu(), bgimage.rlock_cpu():
         outVideoFGMask.write(fgmask.cpu())
         outVideoBGImage.write(bgimage.cpu())
```

输入视频:
![](pedestrians%2000_00_00-00_00_30.gif)


输出结果:

前景:
![](fgmask_cpu%2000_00_00-00_00_30.gif)


背景:
![](bgimage_cpu%2000_00_00-00_00_30.gif)





























