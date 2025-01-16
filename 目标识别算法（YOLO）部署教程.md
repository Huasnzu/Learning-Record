---
title: 目标识别算法（YOLO）部署教程
created: '2024-11-01T11:09:10.036Z'
modified: '2024-11-01T11:12:16.390Z'
---

# **目标识别算法（YOLO）部署教程**

*By huasnzu*

# *Anaconda* 部分

![](D:\Notable\images\v2-4f5f96196712ce30fcb89ed801f9703a_1440w.png)

## 安装部分

官网下载地址：

>https://www.anaconda.com/download#downloads

安装时修改默认安装地址，不装到 C 盘，更换到自己常用的盘里

后续勾选

>✔️Add Anaconda3 to my PATH environment variable

检验是否安装成功：

```
conda –V
```

输出结果为 `conda +版本号` 即安装成功

## *Anaconda* 换源

### 创建`.condarc`文件

```
conda config --set show_channel_urls yes
```

文件目录位于

>C:用户/用户名/.condarc

打开文件，用以下命令覆盖

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

清理索引缓存

```
conda clean -i
```

## 操作部分

### 激活虚拟环境

```
conda activate [name of env]
```

### 退出虚拟环境

```
deactivate
```

### 查看已经创建的虚拟环境

```
conda env list
```

### 删除已经创建的虚拟环境

```
conda remove -n xxx --all
```

# *CUDA* 部分

![](D:\Notable\images\nvidia-logo-horz.png)

## 安装 *NVIDIA CUDA*

### 查看显卡型号

打开设备管理器，再展开显示适配器

若型号样式为 `NVIDIA GeForce XXX XXXX`

则为 *NVIDIA* 显卡，支持 `CUDA`

>ps：若没有独立显卡则不能安装 `CUDA` 和 `cuDNN`

### 更新显卡驱动

官网地址

>https://www.nvidia.cn/Download/index.aspx?lang=cn

根据版本和型号选择分类，笔记本产品系列为 `Notebooks`

默认路径安装即可

### 查看显卡支持 *CUDA* 版本

```
nvidia-smi
```

`Driver Version` 为当前显卡驱动版本，`CUDA Version` 为当前显卡支持最新CUDA版本

### 安装 *CUDA*

先选好 `CUDA` 和 `pytorch` 的版本搭配（笔者选择的是 `CUDA11.8` + `torch2.1.0`）

下载 `CUDA` 

>https://developer.nvidia.com/cuda-toolkit-archive

`Installer Type` 选择 `Local` 

安装说明

* 第一个弹窗的地址为临时路径，安装成功后会被删除，默认即可。在自定义处修改安装地址

* 由于文件较大，不安装在C盘，安装时选择自定义路径，但是要知道安装在哪，因为要配置环境变量，路径最好不要有中文

* 安装选项选择自定义（高级）

* 自定义安装选项勾选所有组件

安装成功之后系统会自动配置两个环境变量

```
CUDA_PATH                       E:\CUDA
CUDA_PATH_V11_8                 E:\CUDA
```

还需额外配置五个环境变量

>*%CUDA_PATH%*代表*CUDA*安装路径，等价于*E:\CUDA*，实际目录以自己安装为准

```
CUDA_SDK_PATH                   #变量名
E:\CUDA                         #变量值 
 
CUDA_LIB_PATH                   #变量名
%CUDA_PATH%\lib\x64             #变量值 
 
CUDA_BIN_PATH                   #变量名
%CUDA_PATH%\bin                 #变量值 
 
CUDA_SDK_BIN_PATH               #变量名
%CUDA_SDK_PATH%\bin\win64       #变量值 
 
CUDA_SDK_LIB_PATH               #变量名
%CUDA_SDK_PATH%\common\lib\x64  #变量值
```

安装完成之后，查看 *CUDA* 版本

```
nvcc –V
```

输出 `CUDA 版本` 即说明安装成功

### 检查 *CUDA* 环境变量

```
set cuda
```

应该输出 7 个环境变量

```
CUDA_BIN_PATH=E:\CUDA\bin
CUDA_LIB_PATH=E:\CUDA\lib\x64
CUDA_PATH=E:\CUDA
CUDA_PATH_V11_8=E:\CUDA
CUDA_SDK_LIB_PATH=E:\CUDA\common\lib\x64
CUDA_SDK_PATH=E:\CUDA
CUDA_SKD_BIN_PATH=E:\CUDA\bin\win64
```

## 安装 *cuDNN*

> CUDA Deep Neural Network

### 官网下载

官网地址：

>https://developer.nvidia.com/rdp/cudnn-archive

找到并下载适配自己 `CUDA` 版本的 `cuDNN`

### 移动文件

将下载完的 `cuDNN` 文件里的三个文件夹中的内容添加到 `CUDA` 对应文件夹中

![](D:\Notable\images\1706959776168.jpg)

*By huasnzu*

## 环境要求

`Anaconda`
`CUDA`
`python`


## 创建 *Anaconda* 虚拟环境

### 创建虚拟环境

```
conda create -n yolov5 python=3.10 -y
```

>python=自己需要的python版本

创建成功之后显示

```
Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment， use
# 
#     $ conda activate yolov5
# 
# To deactivate an active environment， use
#     $ conda deactivate
```

### 激活虚拟环境

```
conda activate yolov5
```

在终端地址前出现 `(yolov5)` 字样，即激活成功

## 配置 *Yolov5*

### 下载 *Yolov5* 源码

***github***

>https://github.com/ultralytics/yolov5

### 安装依赖

将终端目录切换至 `yolov5` 源码目录下

```
cd [path_to_yolov5]
```

若 *cmd* 路径没有改变，则输入对应盘符加引号，则可切换到对应目录，如：

>D:

配置 `requirements` 环境

```
pip install -r requirements.txt
```

### 安装 *GPU* 版本 *pytorch*

>ps：若没有独立显卡则跳过该步骤

```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

具体依赖版本根据自己情况更改

检测是否安装成功

```
pip list
```

输出列表中如下显示则安装成功

```
torch                        2.0.0+cu118
torchaudio                   2.0.1+cu118
torchvision                  0.15.1+cu118
```

## 制作数据集

### 安装 *labelimg*

```
pip install labelimg
```

安装之后打开软件

```
labelimg
```

#### 操作指南

* 点击 `Open`，打开单张图片

* 点击 `Open Dir`，打开图片文件夹

* 点击 `Change Save Dir`，选择标签保存文件夹

* 点击 `view`，选择 `Auto Save Mode` 打开自动保存模式后，当切换图片时即可保存前一张图片的标签信息

#### *labelimg* 快捷键
* w：制作标签

* a/d：切换上/下一张

### *Yolo* 数据集格式

```
dataset #(数据集名字：例如fire) 
├── images      
       ├── train          
              ├── xx.jpg     
       ├── val         
              ├── xx.jpg 
├── labels      
       ├── train          
              ├── xx.txt     
       ├── val         
              ├── xx.txt
```

`Yolo` 格式的标签是与图片一一对应的同名 `.txt` 文档

### 数据集内容

```
1 0.529167 0.801042 0.497222 0.181250
0 0.436111 0.365625 0.438889 0.339583
1 0.827778 0.095833 0.250000 0.187500
1 0.543056 0.038542 0.186111 0.072917
1 0.884722 0.939583 0.169444 0.062500
```

* 这串文字记录了五个标签，第一位数字代表标签的类别，如：

>0 代表第一类标签，1 代表第二类等等

* 后面的四个小数则为标签的参数信息。

* 在标签制作完毕之后，文档会生成一个 `classes.txt` 文件，文件里有每一类标签的名字

### 图片批量改名

下载批量命名工具 `Advanced Renamer`

>https://www.advancedrenamer.com/

* 点击 `Folders` 打开文件夹

* 选择 重构文件名 下方的规则

* 点击 `Start Batch`

## 配置 *Pycharm*

### 安装 *Pycharm*

官网安装

>https://www.jetbrains.com/pycharm/download/?section=windows

选择免费的 `community` 版本即可

* 安装时要修改安装路径，因为 `Pycharm` 文件较大

* 安装选项全部勾选即可

### 更改 *pycharm* 终端

选项路径

>设置>工具>终端>*shell* 路径

将 `shell` 路径切换成 `cmd` 路径

### 配置 *yolov5* 环境

* 添加解释器

* 选择 `conda` 环境

* 添加 `conda` 可执行文件

目录位于

>[path_to_Anaconda]/Scripts/conda.exe

* 点击加载环境，选择使用现有环境，选择刚才创建的 `yolov5` 即可

## 准备编译 *Yolov5*

### 下载预训练模型

`github`页面下方的 `Pretrained Checkpoints` 列表

>https://github.com/ultralytics/yolov5

点击对应蓝字下载 `yolov5s.pt` 即可

各模型具体配置参数如下

| Model    | Size (pixels) | mAPval 50-95 | mAPval 50 | Speed CPU b1 (ms) | Speed V100 b1 (ms) | Speed V100 b32 (ms) | Params (M) | FLOPs @640 (B) |
|:--------:|:-------------:|:------------:|:---------:|:------------------:|:------------------:|:-------------------:|:----------:|:--------------:|
| YOLOv5n   |      640      |     28.0     |    45.7   |        45          |         6.3        |          0.6          |     1.9    |      4.5       |
| YOLOv5s   |      640      |     37.4     |    56.8   |        98          |         6.4        |          0.9          |     7.2    |      16.5      |
| YOLOv5m   |      640      |     45.4     |    64.1   |       224          |         8.2        |          1.7          |    21.2    |      49.0      |
| YOLOv5l   |      640      |     49.0     |    67.3   |       430          |        10.1        |          2.7          |    46.5    |     109.1      |
| YOLOv5x   |      640      |     50.7     |    68.9   |       766          |        12.1        |          4.8          |    86.7    |     205.7      |
| YOLOv5n6  |     1280      |     36.0     |    54.4   |       153          |         8.1        |          2.1          |     3.2    |      4.6       |
| YOLOv5s6  |     1280      |     44.8     |    63.7   |       385          |         8.2        |          3.6          |    12.6    |      16.8      |
| YOLOv5m6  |     1280      |     51.3     |    69.3   |       887          |        11.1        |          6.8          |    35.7    |      50.0      |
| YOLOv5l6  |     1280      |     53.7     |    71.3   |      1784          |        15.8        |         10.5          |    76.8    |     111.4      |
| YOLOv5x6 + TTA |    1280     |     55.0     |    72.7   |      3136          |         26.2       |           19.4         |    140.7   |     209.8      |

>ps: `yolov5n.pt` 针对 `NVIDIA nano` 平台优化

下载之后的 `.pt` 文件直接放在 `Yolov5` 主目录下即可

### 创建数据配置文件

目录为

>yolov5-master/data/

命名随意，以 `.yaml` 结尾，最好以需要识别的目标为名，这样好做区分。

文件格式

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: E:\Yolo\yolov5-master\dataset\human  # dataset root dir
train: E:\Yolo\yolov5-master\dataset\human\images\train  # train images (relative to 'path')
val: E:\Yolo\yolov5-master\dataset\human\image\val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['human']  # class names
```

>注：
>
>`path` 为数据集根目录。
>
>`train` 为对应的 `train` 子目录，即训练集目录。
>
>`val` 为对应的 `val` 子目录，为验证集目录。
>
>`test` 不用管，暂时空缺。
>
>`nc` 表示标签的种类数量
>
>`names` 集合中定义标签的名字
>
>具体目录根据自己实际情况修改

### 配置 *train.py* 文件

* 展开 *Pycharm* 右上顶部 当前文件

* 选择 编辑配置

* 点击左侧添加新的运行配置

* 选择 `python`

* 名称填 `train.py`

* 运行解释器选择之前创建的 `conda` 虚拟环境

* 限定模块选择 `script`

* 路径选择 

>E:/Yolo/yolov5-master/train.py

* 下方脚本输入

```
--weights yolov5s.pt --data data/human.yaml --workers 6 --batch-size 8 --epochs 100
```

>注：
>
>--weights+预训练模型
>
>--data+数据配置文件路径
>
>--workers+调用线程数（根据电脑运行内存大小酌情选择）
>
>--batch-sizi+单次处理图片数量（根据显卡显存酌情选择）
>
>--epoch+迭代次数

* 工作目录选择

>E:\Yolo\yolov5-master

* 环境变量输入

```
PYTHONUNBUFFERED=1
```

### 开始训练模型

选择配置好的 *train* 文件，点击运行。

>显示这样的字样则说明成功调用 GPU，若 torch 结尾是 cpu，则是 torch 版本不对，如需调用 GPU，则回到上文重新安装 torch

```
YOLOv5  2024-1-17 Python-3.10.13 torch-2.0.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4050 Laptop GPU, 6140MiB)
```

开始训练模型实例

```
hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5  runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs\train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=1
```

```
Transferred 343/349 items from yolov5s.pt
AMP: checks passed 
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
train: Scanning E:\Yolo\yolov5-master\datasets\human\labels\train... 105 images, 0 backgrounds, 0 corrupt: 100%|██████████| 105/105 [00:08<00:00, 12.91it/s]
train: New cache created: E:\Yolo\yolov5-master\datasets\human\labels\train.cache
val: Scanning E:\Yolo\yolov5-master\datasets\human\labels\val... 26 images, 0 backgrounds, 0 corrupt: 100%|██████████| 26/26 [00:06<00:00,  3.86it/s]
val: New cache created: E:\Yolo\yolov5-master\datasets\human\labels\val.cache
```

```
AutoAnchor: 4.06 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset 
Plotting labels to runs\train\exp36\labels.jpg... 
Image sizes 640 train, 640 val
Using 6 dataloader workers
Logging results to runs\train\exp36
Starting training for 100 epochs...
```

这样的结果则表示已经开始训练模型

```
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 2/2 [00:12<00:00,  6.47s/it]
                   all         26         85     0.0414      0.259     0.0264    0.00667

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/99      1.78G    0.08875    0.04473          0          3        640: 100%|██████████| 14/14 [00:01<00:00, 10.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 2/2 [00:00<00:00,  6.21it/s]
                   all         26         85        0.3      0.236      0.219     0.0667

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       2/99      1.78G    0.07692    0.04411          0          3        640: 100%|██████████| 14/14 [00:01<00:00, 11.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 2/2 [00:00<00:00,  6.96it/s]
```

模型训练完成结果

```
5 epochs completed in 0.007 hours.
Optimizer stripped from runs\train\exp40\weights\last.pt, 14.4MB
Optimizer stripped from runs\train\exp40\weights\best.pt, 14.4MB

Validating runs\train\exp40\weights\best.pt...
Fusing layers... 
Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 2/2 [00:01<00:00,  1.53it/s]
                   all         26         85      0.734      0.541       0.62      0.244
Results saved to runs\train\exp40
```

### 开始检测

*pycharm* 终端输入

```
python detect.py --weights ./pt/humans.pt --source ./datasets/humans/images/
```

>注：
>
>`--*weights*` + 所用来检测的训练模型
>
>`-- *source*` + 所用来检测的图像路径，若为 *'0'*，则调用电脑摄像头，调用其他摄像头则用 *'1'*、*'2'*……等

### 查看训练结果

训练好的模型位置

>.\yolov5-master\runs\train\exp\weights\best.pt

训练效果解析图位置

>.\yolov5-master\runs\train\exp\result.png

## *train.py* opt 参数解析

```
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 预训练权重文件
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")  #指定模型配置文件路径
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")  #数据集对应的参数文件
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")  #指定超参数文件的路径
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")  # 迭代次数
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")  #每批次的输入数据量
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")  #训练图片的像素大小
    parser.add_argument("--rect", action="store_true", help="rectangular training")  #是否采用矩阵推理的方式去训练模型
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")  #断点续训
    #即是否在之前训练的一个模型基础上继续训练，default 值默认是 False；如果想采用断点续训的方式，这里我推荐一种写法，即首先将 default=False 改为 default=True。
    #python train.py --resume D:\Pycharm_Projects\yolov5-6.1-4_23\runs\train\exp19\weights\last.pt

    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")  #是否只保存最后一轮的pt文件
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")  #只在最后一轮测试
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")  #是否禁用自动锚框；默认是开启的，自动锚点的好处是可以简化训练过程；
    parser.add_argument("--noplots", action="store_true", help="save no plot files")  #开启这个参数后将不保存绘图文件
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")  #遗传超参数进化；yolov5使用遗传超参数进化，提供的默认参数是通过在COCO数据集上使用超参数进化得来的
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")  #谷歌云盘
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")  #是否提前缓存图片到内存，以加快训练速度，默认False；开启这个参数就会对图片进行缓存，从而更好的训练模型。
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")  #是否启用加权图像策略，默认是不开启的；主要是为了解决样本不平衡问题；开启后会对于上一轮训练效果不好的图片，在下一轮中增加一些权重；
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  #设备选择；这个参数就是指定硬件设备的，系统会自己判断的。
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")  #是否启用多尺度训练，默认是不开启的；多尺度训练是指设置几种不同的图片输入尺度，训练时每隔一定iterations随机选取一种尺度训练，这样训练出来的模型鲁棒性更强。
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")  #设定训练数据集是单类别还是多类别；默认为False多类别。
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")  #选择优化器；默认为SGD,可选SGD,Adam,AdamW.
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")  #是否开启跨卡同步BN；开启参数后即可使用SyncBatchNorm多 GPU 进行分布式训练。
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")  #最大worker数量；这里经常出问题，Windows系统报错时可以设置成 0 0 0。
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")  #指定训练好的模型的保存路径；默认在runs/train。
    parser.add_argument("--name", default="exp", help="save to project/name")  #设定保存的模型文件夹名，默认在exp；
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")  #每次预测模型的结果是否保存在原来的文件夹；如果指定了这个参数的话，那么本次预测的结果还是保存在上一次保存的文件夹里；如果不指定就是每次预测结果保存一个新的文件夹下。
    parser.add_argument("--quad", action="store_true", help="quad dataloader")  #官方发布的开启这个功能后的实际效果：好处是在比默认640大的数据集上训练效果更好。副作用是在640大小的数据集上训练效果可能会差一些。
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")  #是否开启余弦学习率；
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")  #是否对标签进行平滑处理，默认是不启用的；
    parser.add_argument("--patience", type=int, default=40, help="EarlyStopping patience (epochs without improvement)")  #早停；如果模型在default值轮数里没有提升，则停止训练模型。
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")  #指定冻结层数量；可以在yolov5s.yaml中查看主干网络层数。
    #冻结训练是迁移学习常用的方法，当我们在使用数据量不足的情况下，通常我们会选择公共数据集提供权重作为预训练权重，我们知道网络的backbone主要是用来提取特征用的，一般大型数据集训练好的权重主干特征提取能力是比较强的，这个时候我们只需要冻结主干网络，fine-tune后面层就可以了，不需要从头开始训练，大大减少了实践而且还提高了性能。
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")  #用于设置多少个epoch保存一下checkpoint
    parser.add_argument("--seed", type=int, default=1, help="Global training seed")  #如果你使用torch>=1.12.0的单GPU训练， 那你的训练结果完全可再现。
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")  #单机多卡训练，单GPU设备不需要设置

    # Logger arguments
    parser.add_argument("--entity", default=1, help="Entity")  #在线可视化工具，类似于tensorboard。
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')  #是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")  #设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认 -1
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")  #这个功能作者还未实现。

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()
```
### *train* 报错解决

#### 警告1

报错信息显示有 `tensorflow` 不支持的函数格式

将 `tensorflow` 版本降低即可

```
pip install tensorflow==2.13.0
```

#### 警告2

训练时产生如下警告
```
train: WARNING  E:\Yolo\yolov5-master\datasets\mask\images\train\7432.jpg: corrupt JPEG restored and saved
```

造成这个问题的原因就是：图片在前期数据格式转化的时候是直接由 `png` `jped` `bmp` 等格式的图像数据强制转化为了 `jpg` ，这样是存在风险的

删除警告图像及其标签即可

## *detect.py* opt 参数解析

```
def parse_opt():
    parser = argparse.ArgumentParser()

    #weights: 训练的权重路径,可以使用自己训练的权重,也可以使用官网提供的权重
    #默认官网的权重yolov5s.pt(yolov5n.pt/yolov5s.pt/yolov5m.pt/yolov5l.pt/yolov5x.pt/区别在于网络的宽度和深度以此增加)
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")

    # source: 测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流, 默认data/images
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")

    #data: 配置数据文件路径, 包括image/label/classes等信息, 训练自己的文件, 需要作相应更改, 可以不用管
    #如果设置了只显示个别类别即使用了--classes = 0 或二者1, 2, 3等, 则需要设置该文件，数字和类别相对应才能只检测某一个类
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")

    #imgsz: 网络输入图片大小, 默认的大小是640
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")

    #conf-thres: 置信度阈值， 默认为0.25
    parser.add_argument("--conf-thres", type=float, default=0.30, help="confidence threshold")

    #iou-thres:  做nms的iou阈值, 默认为0.45
    parser.add_argument("--iou-thres", type=float, default=0.60, help="NMS IoU threshold")

    #max-det: 保留的最大检测框数量, 每张图片中检测目标的个数最多为1000类
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")

    #device: 设置设备CPU/CUDA, 可以不用设置
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")

    #view-img: 是否展示预测之后的图片/视频, 默认False, --view-img 电脑界面出现图片或者视频检测结果
    parser.add_argument("--view-img", action="store_true", help="show results")

    #save-txt: 是否将预测的框坐标以txt文件形式保存, 默认False, 使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")

    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")

    #save-conf: 是否将置信度conf也保存到txt中, 默认False
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")

    #save-crop: 是否保存裁剪预测框图片, 默认为False, 使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")

    #nosave: 不保存图片、视频, 要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")

    #classes: 设置只保留某一部分类别, 形如0或者0 2 3, 使用--classes = n, 则在路径runs/detect/exp*/下保存的图片为n所对应的类别, 此时需要设置data
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")

    #agnostic-nms: 进行NMS去除不同类别之间的框, 默认False
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")

    #augment: TTA测试时增强/多尺度预测
    parser.add_argument("--augment", action="store_true", help="augmented inference")

    #visualize: 是否可视化网络层输出特征
    parser.add_argument("--visualize", action="store_true", help="visualize features")

    #update: 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
    parser.add_argument("--update", action="store_true", help="update all models")

    #project:保存测试日志的文件夹路径
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")

    #name:保存测试日志文件夹的名字, 所以最终是保存在project/name中
    parser.add_argument("--name", default="exp", help="save results to project/name")

    #exist_ok: 是否重新创建日志文件, False时重新创建文件
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")

    #line-thickness: 画框的线条粗细
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")

    #hide-labels: 可视化时隐藏预测类别
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")

    #hide-conf: 可视化时隐藏置信度
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")

    #half: 是否使用F16精度推理, 半进度提高检测速度
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")

    #dnn: 用OpenCV DNN预测
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")

    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")

    opt = parser.parse_args()  # 扩充维度
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))  # 打印所有参数信息
    return opt
```

