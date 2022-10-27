# (含源码)利用NVIDIA VPI之透视变换

![](perspwarp_cuda%2000_00_00-00_00_30.gif)

NVIDIA 视觉编程接口 (`VPI: Vision Programming Interface`) 是 NVIDIA 的计算机视觉和图像处理软件库，使您能够实现在 NVIDIA Jetson 嵌入式设备和独立的GPU 上可用的不同硬件后端上加速的算法。

库中的一些算法包括过滤方法、透视扭曲、时间降噪、直方图均衡、立体视差和镜头失真校正。 VPI 提供易于使用的 Python 绑定以及 C++ API。

除了与 OpenCV 接口外，VPI 还能够与 PyTorch 和其他基于 Python 的库进行互操作。 在这篇文章中，我们将通过基于 PyTorch 的目标检测和跟踪示例向您展示这种互操作性如何工作。 有关详细信息，请参阅[视觉编程接口 (VPI) 页面](https://developer.nvidia.com/embedded/vpi)和[视觉编程接口](https://docs.nvidia.com/vpi/)文档。


下面的示例获取输入视频并输出视频，其中对每一帧应用不同的透视扭曲。 结果是透视弹跳效果。 可以修改示例应用程序以从相机获取输入并实时应用效果。

```Python
import cv2
 import sys
 import vpi
 import numpy as np
 from math import sin, cos, pi
 from argparse import ArgumentParser
  
 # ----------------------------
 # Parse command line arguments
  
 parser = ArgumentParser()
 parser.add_argument('backend', choices=['cpu', 'cuda','vic'],
                     help='Backend to be used for processing')
  
 parser.add_argument('input',
                     help='Input video to be denoised')
  
 args = parser.parse_args();
  
 if args.backend == 'cuda':
     backend = vpi.Backend.CUDA
 elif args.backend == 'cpu':
     backend = vpi.Backend.CPU
 else:
     assert args.backend == 'vic'
     backend = vpi.Backend.VIC
  
 # -----------------------------
 # Open input and output videos
  
 inVideo = cv2.VideoCapture(args.input)
  
 fourcc = cv2.VideoWriter_fourcc(*'MPEG')
 inSize = (int(inVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
 fps = inVideo.get(cv2.CAP_PROP_FPS)
  
 outVideo = cv2.VideoWriter('perspwarp_python'+str(sys.version_info[0])+'_'+args.backend+'.mp4',
                             fourcc, fps, inSize)
  
 #--------------------------------------------------------------
 # Main processing loop
 curFrame = 1
 while True:
     print("Frame: {}".format(curFrame))
     curFrame+=1
  
     # Read one input frame
     ret, cvFrame = inVideo.read()
     if not ret:
         break
  
     # Convert it to NV12_ER format to be used by VPI
     with vpi.Backend.CUDA:
         frame = vpi.asimage(cvFrame).convert(vpi.Format.NV12_ER)
  
     # Calculate the transformation to be applied ------------
  
     # Move image's center to origin of coordinate system
     T1 = np.array([[1, 0, -frame.width/2.0],
                    [0, 1, -frame.height/2.0],
                    [0, 0, 1]])
  
     # Apply some time-dependent perspective transform
     v1 = sin(curFrame/30.0*2*pi/2)*0.0005
     v2 = cos(curFrame/30.0*2*pi/3)*0.0005
     P = np.array([[0.66, 0, 0],
                   [0, 0.66, 0],
                   [v1, v2, 1]])
  
     # Move image's center back to where it was
     T2 = np.array([[1, 0, frame.width/2.0],
                    [0, 1, frame.height/2.0],
                    [0, 0, 1]])
  
     # Do perspective warp using the backend passed in the command line.
     with backend:
         frame = frame.perspwarp(np.matmul(T2, np.matmul(P, T1)))
  
     # Convert it to RGB8 for output using the CUDA backend
     with vpi.Backend.CUDA:
         frame = frame.convert(vpi.Format.RGB8)
  
     # Write the denoised frame to the output video
     with frame.rlock_cpu() as data:
         outVideo.write(data)
```

原视频:

![](noisy%2000_00_00-00_00_30~2.gif)

处理结果:

![](perspwarp_cuda%2000_00_00-00_00_30.gif)































