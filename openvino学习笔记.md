# OpenVINO推理方法
OpenVINO的新版API引入了一些改进和变化，以提供更简洁、更易用的接口。新版API主要集中在OpenVINO 2022.1及之后的版本，并通过**openvino.runtime**模块提供功能。

使用OpenVINO新版API进行模型推理主要分为8个步骤，其主要流程如下：

创建推理引擎核心对象——>读取中间格式文件——>将模型加载到设备——>准备输入输出格式——>创建推理请求——>准备输入数据——>运行推理——>处理结果

以下是分步骤的详细解析及示例代码：

1、创建推理引擎核心对象

初始化推理引擎核心对象 ```Core```
```python
from openvino.runtime import Core

#初始化Core
core = Core()
```
2、读取中间格式文件

读取经过Model Optimizer转换后的中间表示（IR）文件。请注意，这里需要通过mo_onnx.py将已有的.onnx文件转换为.xml和.bin文件。其中，.xml文件记录模型的拓扑结构，.bin文件记录模型的权重和偏置。
```python
#读取IR模型
model = core.read_model(model='model.xml')
```
3、将模型加载到设备

将编译后的模型加载到指定的设备上
```python
compiled_model = core.compile_model(model=model, device_name='CPU')
```
如果要用GPU推理，那么在原始代码前加入以下代码
```python
available_devices = core.available_devices
device_name = "GPU" if "GPU" in available_devices else "CPU"
```
4、准备输入输出格式

获取模型的输入和输出信息
```python
input_layer = model.input(0)
output_layer = model.output(0)
```
这里的.input(0)方法用于获取模型的第一个输入层，.output(0)方法用于获取模型的第一个输出层。这里的input_layer和output_layer是表示输入层和输出层的对象，包含形状、数据类型等信息。

5、创建推理请求

在新版API中，通过```compiled_model```对象直接进行推理，因此创建推理请求隐含在```compiled_model```中。

6、处理输入数据

加载并预处理输入数据，使其符合模型输入的要求。
```python
import cv2
import numpy as np

# 准备输入数据
image = cv2.imread('image.jpg')  # 读取图像
image = cv2.resize(image, (input_layer.shape[3], input_layer.shape[2]))  # 调整图像大小至模型输入大小
image = image.transpose((2, 0, 1))  # 将图像格式从HWC转换为CHW
image = image.reshape(1, *image.shape)  # 调整图像形状为批量大小
```
`cv2.resize(image, (input_layer.shape[3], input_layer.shape[2]))`用于将图像调整为模型输入层所需的尺寸。`cv2.resize(image,(width,height))`接受两个参数，第一个参数是要调整的图像（这里是image），第二个参数是一个元组（width,height），指定调整后的宽度和高度。

input_layer.shape是一个描述输入层形状的元组，包含各个维度的大小。例如，对于一个（1，3，224，224）的输入层，input_layer.shape的形状是（1，3，224，224）。其中，`1`表示批量大小（batch_size），`3`表示通道数，这里是RGB三个通道，`224`表示图像的高度（H），`224`表示图像的宽度（W）。因此，给cv2.resize函数输入的参数元组是（input_layer.shape[3], input_layer.shape[2]）。

深度学习模型在推理或训练时通常会处理多个样本（图像）的批量。批量大小（batch size）是指一次处理的样本数量。在 OpenVINO 中，模型推理通常期望输入数据的形状为 (N, C, H, W)，其中：

N 是批量大小，即一次输入到模型中的样本数。
C 是通道数。
H 是高度。
W 是宽度。

对于推理任务，即使只有一张图像，我们也需要将其形状调整为 (1, C, H, W) 形式，这样可以与模型的输入格式一致。因此，`image.reshape(1, *image.shape)`的作用是，将图像新装从`(C,H,W)`调整为`(1,C,H,W)`，其中`1`代表批量大小。

7、运行推理

运行推理请求，使用预处理后的输入数据进行推理
```python
results = compiled_model([image])
```

8、处理结果

处理推理结果，提取并使用推理输出。
```python
# 处理输出
output = results[output_layer]
print(output)
```

补充：在实际应用中，步骤6-8会在一个循环中反复进行，以处理多个输入。

完整代码如下：
```python
# 安装必要的库
!pip install openvino openvino-dev[onnx] notebook numpy opencv-python

# 导入库
import cv2
import numpy as np
from openvino.runtime import Core, get_version

# 1. 初始化Core
core = Core()

# 打印OpenVINO版本
print(f"OpenVINO version: {get_version()}")

# 2. 检测可用设备
available_devices = core.available_devices
print(f"Available devices: {available_devices}")

# 选择GPU设备，如果可用的话
device_name = "GPU" if "GPU" in available_devices else "CPU"
print(f"Using device: {device_name}")

# 3. 读取IR模型
model_path = "path_to_your_model/mobilenet-v2.xml"
model = core.read_model(model=model_path)

# 4. 编译模型到设备
compiled_model = core.compile_model(model=model, device_name=device_name)

# 5. 获取模型输入和输出信息
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 以下部分可以放在循环中以处理多个输入

# 6. 读取并准备输入图像
image = cv2.imread('path_to_your_image/image.jpg')
image = cv2.resize(image, (input_layer.shape[3], input_layer.shape[2]))  # 调整图像大小
image = image.transpose((2, 0, 1))  # 将图像格式从HWC转换为CHW
image = image.reshape(1, *image.shape)  # 调整图像形状为批量大小

# 7. 运行推理
results = compiled_model([image])

# 8. 处理输出
output = results[output_layer]

# 找到概率最高的类别
predicted_class = np.argmax(output)
print(f"Predicted class: {predicted_class}")
```
