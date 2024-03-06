# 要求（无需作答）
- 全程录屏
- 如有必要允许任意使用工具（使用GPT等生成式工具将折算该题目成绩为80%）、搜索内容（但不强制）
- 注意时限（答题时间45分钟）
- 遇到无法回答的问题可以选择记录思路后跳过
- 总分100分，额外加分10分, 主观分10分

# 通用问题

## 经历【5分】
- 个人github（如果有）
	- 有https://github.com/mlc-ai
- 个人博客（如果有）
	- 
- 个人满意的展示项目（如果有）
	- 需要远程连接ssh，会断网
- 最近看到最感兴趣的项目
	- 那肯定是sora
- 最近看到最感兴趣的论文
	- 顶会论文对sd的语义解读
    - What the DAAM: Interpreting Stable Diffusion Using Cross Attention

## python/linux基础操作【15分】
- python包环境管理（6分）
	- 简答：conda, venv 和 pip 的主要区别和使用场景（3分）
        - conda指的是anaconda的一个开发环境，它一般用于创作虚拟环境，区分每个工程
        - venv是具体的虚拟环境
        - pip是下载的命令
	- 简答：测试一个GitHub repo时, 如何确认并防止依赖冲突和降级覆盖（3分）
		- 
- pytorch
    - 简答：Pytorch 支持哪些数值精度，使用混合/半精度计算时可能遇到哪些问题？ (3分)
    - 支持8、16位精度，混合/半精度计算时参数准确性降低，容易造成模型精度下降，可能发生模型鲁棒性降低
- linux 远程开发[本地模拟]（6分）
    - ssh 命令连接某远程测试机ssh.linkto.co，端口为 2333，用户名为dev （1分）
    - 视频演示
    - 创建新的tmux窗口（1分）
    - 确认本机IP和外网IP （2分）
    - 安装并使用ncdu导出/etc路径下的存储分布（2分）	
	
## CV【10分】
创建 func_cv.py, 实现以下功能函数
-  opencv 读取图像，转换为HSV颜色空间（3分）
-  使用 alpha blending 将两张图片 合成在一起（2分）
- 【额外加分题】调用 fpie 实现泊松融合，并解释原理（+5分）




## python[ 二选一]【20分+5分】
### 题一：创建 pipe.py, 实现以下功能函数
- 实现一个消费者模式的任务排队算法，为输入图片调用func_cv.py中的处理函数（10分）
- 实现 Task类，支持设置任务优先级（5分）
- 实现公平等待，考虑等待时长调整消费优先级(简易起见建模为：w=e^kt)（5分）
- 【额外加分题】使用 pdb 手动断点调试，获取某一中间过程输出（+5分）

### 题二：pytorch
创建 unet.py, 实现以下功能
- 使用pytorch实现一个三层UNet结构，每层由2个MLP层和self-Attention层组成（10分）
- 见代码
- 实现一个cross attention 层，接受额外的外部输入（5分）
- - 见代码
- 简答：介绍 attention 层的加速方案（5分）
- Attention 层的加速方案主要有以下几种：

稀疏注意力机制：传统的注意力机制在计算注意力权重时，会对所有的输入元素进行加权求和，这会导致计算量巨大。稀疏注意力机制通过引入稀疏性，使得每个输出元素只关注输入元素的一小部分，从而大大减少计算量。例如，局部注意力机制（Local Attention）和自注意力机制（Self-Attention）就是典型的稀疏注意力机制。
混合精度训练：混合精度训练是一种使用不同精度的浮点数进行神经网络训练的方法。在 Attention 层中，可以采用混合精度训练来加速计算。例如，可以将权重和激活值使用半精度浮点数（FP16）进行计算，而将梯度使用全精度浮点数（FP32）进行更新。这样可以在不损失精度的情况下，显著提高计算速度。
硬件加速：利用专门的硬件加速器，如 GPU、TPU 或 FPGA，可以显著提高 Attention 层的计算速度。这些硬件加速器具有并行计算和优化的内存访问模式，能够充分利用计算资源，从而加速 Attention 层的计算。
算法优化：通过对算法进行优化，如使用高效的矩阵乘法库、减少不必要的内存访问等，也可以提高 Attention 层的计算速度。此外，还有一些研究工作致力于改进 Attention 机制的计算方式，如使用更高效的注意力计算方法或引入近似计算等。
模型剪枝和量化：通过模型剪枝和量化技术，可以进一步减小模型的计算量和存储需求，从而加速 Attention 层的计算。剪枝技术通过去除模型中的冗余连接，减少模型的复杂度；而量化技术则通过将浮点数转换为定点数，减小模型的存储需求并提高计算速度。
综上所述，可以通过多种方式来加速 Attention 层的计算，包括引入稀疏性、使用混合精度训练、利用硬件加速器、算法优化以及模型剪枝和量化等。这些方法可以单独使用，也可以组合使用，以达到最佳的加速效果。
- 【额外加分题】使用 pdb 手动断点调试，获取中间self-Attention层的输出特征（+5分）
见演示，因使用ssh远程连接服务器，连接后会断网，因此无法在线演示
## diffusion【15分】
- 简答：stable diffusion 的 构成组件 和 生成 pipeline（3分）
    - sd主要包括两部分，生成噪音和去噪，包括VAE和clip
    - pipeline将文本转化为token，通过clip嵌入，将文本嵌入和噪声构成latent输入到去噪模型中
- 简答：lora 的原理和优势（3分）
	- lora是一种微调方法，他相对调用更少的资源，只是在预训练模型部分层中增加了权重参数，没有训练所有参数
- 简答：GAN 和 Diffusion 的主要特点和区别（三句话以内）（3分）
    - gan主要是由生成器判别器注册
    - diffusion模型主要是模拟噪声扩散然后去噪
- 简答：指出以下代码的作用及执行中会遇到的问题（6分）
```
from diffusers import DDIMScheduler,StableDiffusionPipeline
prompt = "photo of 1girl"
negative_prompt = "realistic, photo-realistic, 3D, bad quality, bad anatomy, worst quality, low quality, lowres, extra fingers, blur, blurry, ugly, wrong proportions, watermark, image artifacts, bad eyes, bad hands, bad arms"
generator = torch.Generator().manual_seed(42)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16, 
).cuda()

images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height = 400,
    width = 1024  #缺少逗号
    num_images_per_prompt=4,
    num_inference_steps=5,
    generator=generator,
).images
images.save("result.png") 一组图像要一次保存
```
## 部署【20分】
### API（+15分）
设计一个Base Class 用于上述Pytorch UNet模型部署(如选择题一，使用resnet)，包含以下功能
- 支持批量处理
- 支持多个模型的部署和切换
- 确定某一种图片序列化传输方案
- 选取某一种远程函数调用方案并说明理由

# (快到时间了，自己肯定写不完，直接生成)
要设计一个基类（Base Class）用于PyTorch UNet模型（如选择题一，使用ResNet作为编码器）的部署，并满足批量处理、多模型部署和切换、图片序列化传输方案以及远程函数调用方案的需求，我们需要考虑以下几个关键点：

批量处理：基类应能够处理批量输入数据，并且内部逻辑应支持并行化或向量化计算以提高效率。

多模型部署和切换：基类应包含一个模型字典，用于存储和管理多个模型。切换模型时，应能通过简单的方法（如设置属性或调用函数）来切换当前活动的模型。

图片序列化传输方案：应定义一种序列化方案，如将图片转换为NumPy数组或PyTorch张量，然后通过如pickle、JSON、Protocol Buffers或Msgpack等方式进行序列化，以便在网络中传输。

远程函数调用方案：选择一个适用于分布式计算或微服务的远程函数调用方案，如RESTful API、gRPC、Thrift或Apache Dubbo等。选择时需要考虑通信效率、易用性、可维护性、安全性和扩展性。

以下是基于上述考虑的一个简化的基类设计示例：

python
import torch  
import torchvision.transforms as transforms  
import torch.nn as nn  
import pickle  
  
class ModelServerBase:  
    def __init__(self):  
        self.models = {}  # 用于存储多个模型的字典  
        self.current_model = None  # 当前活动的模型  
        self.transform = transforms.Compose([  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])  # 示例变换，用于预处理图像  
  
    def load_model(self, model_name, model_path):  
        """  
        加载模型到内存  
        :param model_name: 模型名称  
        :param model_path: 模型文件路径  
        """  
        model = torch.load(model_path)  
        self.models[model_name] = model.eval()  # 确保模型在评估模式  
  
    def switch_model(self, model_name):  
        """  
        切换当前活动的模型  
        :param model_name: 要切换的模型名称  
        """  
        if model_name in self.models:  
            self.current_model = self.models[model_name]  
        else:  
            raise ValueError(f"Model {model_name} not found!")  
  
    def preprocess_image(self, image):  
        """  
        预处理图像以供模型使用  
        :param image: 输入图像  
        :return: 处理后的图像张量  
        """  
        return self.transform(image)  
  
    def serialize_image(self, image):  
        """  
        序列化图像以供网络传输  
        :param image: 要序列化的图像  
        :return: 序列化后的数据  
        """  
        # 示例：将图像转换为张量并序列化  
        tensor = self.preprocess_image(image)  
        return pickle.dumps(tensor.cpu().numpy())  
  
    def deserialize_image(self, serialized_data):  
        """  
        反序列化从网络接收的图像数据  
        :param serialized_data: 序列化后的数据  
        :return: 反序列化后的图像张量  
        """  
        # 示例：反序列化并转换为PyTorch张量  
        numpy_array = pickle.loads(serialized_data)  
        tensor = torch.from_numpy(numpy_array).float()  
        return tensor  
  
    def predict(self, serialized_images):  
        """  
        对批量图像进行预测  
        :param serialized_images: 序列化后的图像列表  
        :return: 预测结果列表  
        """  
        if not self.current_model:  
            raise ValueError("No model is currently set!")  
  
        # 反序列化图像  
        images = [self.deserialize_image(img) for img in serialized_images]  
  
        # 将图像张量放到同一个batch中  
        images = torch.stack(images)  
  
        # 进行预测  
        with torch.no_grad():  
            predictions = self.current_model(images)  
  
        return predictions.cpu().numpy()  
  
def remote_call(self, url, data):  
        """  
        使用HTTP POST请求进行远程函数调用  
        理由：HTTP POST请求适合传输大量数据，并且易于实现和使用。  
        此外，它提供了良好的错误处理和重试机制。  
        """  
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}  
        response = requests.post(url, data=data, headers=headers)  
        return response.json()

### 【额外加分题】（+5分）
使用以下某一种格式完成推理模型的导出，并用于加速上色模型部署
- onnx
import torch  
import your_model  # 导入你的上色模型  
  
# 加载训练好的模型  
model = your_model.load_model()  
model.eval()  
  
# 准备一个示例输入  
example_input = torch.randn(1, 3, 256, 256)  # 假设输入是一个256x256的RGB图像  
  
# 将模型转换为ONNX格式  
torch.onnx.export(model, example_input, "colored_model.onnx")

