# Locating Veins on Infrared(Gray) Images for IV Insertion with Two Approaches - Segementation(U-Net) | Detection(YOLO)

This project is an self-implemented submodule of Robotic IV Insertion Project affiliated with a Duke Medical Lab, which has the similar idea as a paper Visual Vein-Finding for Robotics IV Insertion by R. D. Brewer and J. K. Salisbury. Genereally speaking, a fast and automated algorithm to find the precise vein edges on the captured infrared images can help suggest insertion points(region) for a human or robotic practitioner.

The images look like the following one because the deoxygenated haemoglobin in veins absorb infrared light more than surrounding tissues, which makes the veins appear as dark on a lighter background.

<p align="center"> <img src="./assets/sample.jpg" alt="drawing" height="40%" width="40%"/> </p>


# Progress & Notes
### Last Meet
- 从downsample image resolution的角度去看能否提升U-Net的速度
- 确定了一个统一的metric来衡量算法的accuracy: 比较中心点差距
- 在真实数据集上训练，并对于performance做统计对比分析。
- 搞懂yolo代码，对model部分进行修改。 

[Mid-term Report](https://docs.google.com/presentation/d/1ixdv6zaCGtkkfqN84M2ob_LZV4D6BFkfnUIeEUIfd7A/edit?usp=sharing)

修改意见：
- 介绍segmentation的网络结构时，放一个sample input和sample output
- segmentation results visualization可以用如下形式：
  ```
              human_label  
            /              \
  input_img                   overlapped image (two color layers showing direct differernce)
            \              / 
              predicte_img
  ```
- add mathematical functions 添加数学公式
- yolo的regression思想最好也通过简单的visualization来展现


## Segmentation - UNet
### Jan 30
- add 20 pairs of img-mask into 'data' directory
- modify the original n_channels = 3 to n_channels = 1 in train.py and predict.py when instantiating an U-Net
- modify glob() in __getitem__ of BasicDataset so that one img to one mask
- pip3 install future, tb-nightly (for torch.utils.tensorboard.SummaryWriter) - 'past module not installed'
- batchsize目前看来只能是1，待研究其中的原因。
- 解决了不能打开tensorboard的问题：运行diagnose_tensorboard.py文件，根据suggestions进行操作。- 实际上就是因为前面装了tb-nightly和pytorch内置的tensorboard冲突了。
- 解决了working with remote jupyter notebook ipynb on local vscode:（推荐方法二）
    - 方法一：
        - 在remote server terminal上运行 jupyter notebook --no-browser --port=8889
        - 在本地terminal运行 ssh -N -f -L localhost:8888:localhost:8889 joey@0.tcp.ngrok.io
        - 在本地浏览器打开localhost:8888
    - 方法二：
        - 下载ms-python extension。安装之后，选择python运行环境。

### Feb 5
- 确定了training用的是binary crossentropy, eval用的是dice loss
- torch.size() return [bs, channels, depth, height, width], 而pil_image.size return [width, height]。
- 猜测原本predict.py文件中多写了一步resize，使得output mask和input image的size不匹配。
- 将train.py中向tensorboard写images或者scalars的部分改成了每隔一个epoch写一次。
- 在ipyng中实现了：
    - 从一个model set中找到score最大的model.pth
    - draw training loss and validation score
    - 做一些当前performance的测试
- went through basic pipeline

### 初步结果
<p align="center"> <img src="./assets/output_unet.jpg" alt="drawing" height="90%" width="90%"/> </p>

### Next:
- 标Img_Invivo的200张。尽量给一个比较平滑的外边界，不要太在意边角。label的区域可以比实际的多一圈。
- 在新标的数据上，训练，完了统计evaluation performance。
  
- 加augmentation
- 尝试调整loss function
- 加weight matrix以给边界上的点更高权重
- 给prediction加上后处理 - 调用CV包去拟合一个elliptical shape


## Detection - YOLO
### YOLO三大特点
- 快。base版本：45frames/s; fast版本：155frames/s;
- 准。mAP接近其他的SOTA methods，且less likely to get false positives.
- generalize well from natural images to other domains like artwork.


### Mar 5
- 标了20张bounding box的数据，并在eda.ipynb中，将RectLabel生成的存有objects attributes的xml文件，读取出bounding box的位置信息，按照label_idx, xmin, xmax, ymin, ymax的format写到txt中
- 根据原项目的customization instructions，初步完成yolo在本数据集上的训练和test。需要注意的是：cpu数量改为1，batch size改为1，epochs改为适量（20或30）。
- datasets.ImageFolder中的image preprocessing默认接收的是3xWxH的图。暂时解决：用torch.tensor.expand增加了2个channel（其实就是channel-wise copy）

### Mar 15
- 看论文，check网络结构。看能否将网络结构改成适用于gray image的，即channel size是1的。而不是将图片转换成3 channel。
  - 应该是要把yolov3-custom.cfg的channels从3换成1。
  - 改过来后，如果training没问题，再把tensorboard改成pytorch自身的tensorboard utils。

### Mar 16
- 重新create yolo custom config,再运行training，过一遍代码修改，看gray image的效果。
  - 出现了一个object有多个detection box的情况：通过提高conf_thres和降低nms_thres来解决的。
- 解决输出图片的size和original image shape不一致的问题。
  - check过，发现是因为plot show image and add bbox之后存下来的图确实可能会和原图大小不一，所以就在detect.py中加了一个后处理。
  - 即，check存下来的图的size，如果和input_img的size不一致，就resize并且保存覆盖掉之前的图。
  - 目前保存下来的图，其实会多了一些白色边框，是因为有些bbox超出了image本身的边长限制，从而在add生成的patches到plot上的时候自适应地加了白色边框。
    - 目前已经考虑了patch的坐标超出plot边界的情况，白色边框的效果稍微没有那么明显了。
- 写一个对single image prediction的block
  
### 初步结果
<p align="center"> <img src="./assets/output_yolo.jpg" alt="drawing" height="80%" width="80%"/> </p>

### Next
- label 200张的数据集，分别用于seg和detection，训练，测试性能。
- 测试performance - efficiency vs. accuracy。搭好性能对比的框架。

## Post-Processing - Active Contour
基于unet和yolo做的结果，分别调active contour的包，去得到最终refine的vein edges。

## Idea of Later Further Modification
- 有没有可能设计loss让网络直接学target edge，从而实现end-to-end。