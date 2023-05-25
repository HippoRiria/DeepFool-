import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
from torch.autograd import Variable
from deepfool import deepfool#导入deepfool算法
from deepfool_target import deepfool_t
import os
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库

#开始程序
'''
modelpath="./models/ckpt.pth"
net=torch.load(modelpath)#从本地导入神经网络模型
net=VGG('VGG19')
checkpoint = torch.load(modelpath)
net.load_state_dict(checkpoint['net'])
#net = {key: net[key].cuda() for key in net}
net.eval()
'''



# 模型选择
#net2 = models.resnet34(pretrained=True)#设定网络为resnet34
#net2=models.resnet50(pretrained=True)
#net1=models.resnet101(pretrained=True)

# net=models.alexnet(pretrained=True)

net2 = models.vgg11 (pretrained=True)
net1 = models.vgg19 (pretrained=True)

#net1=models.googlenet(pretrained=True)#15x15
# net = models.squeezenet1_1(pretrained=True)  # 1.0最小尺寸为21x21，1.1最小为17x17

# Switch to evaluation mode
net1.eval ()  # 测试模型
net2.eval ()
# 导入图片
im_orig = Image.open (
    'D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_00000266.JPEG')  # 导入图片 143、
# im_orig=Image.open('D:/毕设/DeepFool/test_im1.jpg')
# 图片的值
mean = [0.485, 0.456, 0.406]  # 均值，三分量顺序是RGB，后期要调整，opencv和numpy可以自己算得出来
std = [0.229, 0.224, 0.225]  # 方差

# 去掉平均值
im = transforms.Compose ([
    transforms.Scale (256),  # 图片尺寸
    transforms.CenterCrop (224),  # 进行中心裁剪
    transforms.ToTensor (),  # 转换图片格式，变成张量
    transforms.Normalize (mean=mean, std=std)]) (im_orig)  # 对原图进行标准化操作

time = 0
num_classes = 1000  # 分类数
r, loop_i, label_orig, label_pert, pert_image, time = deepfool (im, net1, num_classes)  # deepfool执行，输入图片与神经网络参数
# r, loop_i, label_orig, label_pert, pert_image,time = deepfool_t(im, net1, num_classes)#deepfool_t执行，输入图片与神经网络参数

# 输出标签
label1 = label_pert
label1_ori = label_orig
print ('label1 is:', label1)
print ('')

# Switch to evaluation mode
net2.eval ()  # 测试模型
net = net2.cuda ()
x = Variable (pert_image, requires_grad=True)  # x存入被扰动的图片
fs = net.forward (x)  # 向前传播，得到各个分类器的值
label2 = np.argmax (fs.data.cpu ().numpy ().flatten ())  # 把数组折叠成一维，选最大的那个分类器
print ('label2 is:', label2)
print ('')

im = im.cuda ()
f_image = net.forward (
    Variable (im[None, :, :, :], requires_grad=True)).data.cpu ().numpy ().flatten ()  # 先得到原图的参数，然后向前传播

I = (np.array (f_image)).flatten ().argsort ()[::-1]  # argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

I = I[0:num_classes]  # 分类以及分类器的编号，这里返回了分类器原标签的编号
label2_org = I[0]  # 取最大的编号，得到标签
print ('label2_org is:', label2_org)
print ('')
