import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
from deepfool import deepfool#导入deepfool算法
from deepfool_target import deepfool_t
import os
from models_a import *
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库

from torch.autograd import Variable

#开始程序

# 图片的值
mean = [0.485, 0.456, 0.406]  # 均值，三分量顺序是RGB，后期要调整，opencv和numpy可以自己算得出来
std = [0.229, 0.224, 0.225]  # 方差


testimg=Image.open("D:/毕设/DeepFool/1.jpg")


im1 = transforms.Compose([
    transforms.Scale(256),#图片尺寸
    transforms.CenterCrop(224),#进行中心裁剪
    transforms.ToTensor(),#转换图片格式，变成张量
    transforms.Normalize(mean = mean,std = std)])(testimg)#对原图进行标准化操作

# 导入图片
# image_o = Image.open('D:/毕设/pictures/扰动样本.jpg')#导入图片
# image_o = Image.open('D:/毕设/pictures/扰动样本d.jpg')#导入图片
#image_o = Image.open('D:/毕设/pictures/ILSVRC2017_test_00000184.JPEG')  # 导入图片
#image_o = Image.open('D:/毕设/pictures/ILSVRC2017_test_00000012.JPEG')
image_o=Image.open('D:/毕设/Deepfool/NES-brid.png').convert('RGB')
#image_o = Image.open('D:/毕设/DeepFool/NES-ori-test.png').convert('RGB')#导入图片
# im_orig=Image.open('D:/毕设/DeepFool/test_im1.jpg')


# 去掉平均值
is_cuda = torch.cuda.is_available ()  # 引入GPU计算
im = transforms.Compose([
    transforms.Scale(256),  # 图片尺寸
    transforms.CenterCrop(224),  # 进行中心裁剪
    transforms.ToTensor(),  # 转换图片格式，变成张量
    transforms.Normalize(mean=mean, std=std)])(image_o)  # 对原图进行标准化操作
total_image=im
total_image=total_image.cuda()

for i in range(7):
    if i==0:
        net = models.alexnet(pretrained=True)  # 有面数限制,最小为63x63
        print("1、目前计算至：alexnet")
        print("")
    elif i==1:
        net = models.vgg11(pretrained=True)  # 支持最小输入尺寸为32x32，vgg最小都是32x32
        print("2、目前计算至：vgg11")
        print("")
    elif i==2:
        net = models.squeezenet1_1(pretrained=True)  # 1.0最小尺寸为21x21，1.1最小为17x17
        print("3、目前计算至：squeezenet1_1")
        print("")
    elif i==3:
        net = models.densenet121(pretrained=True)  # 全为29x29，参数看文档https://pytorch.org/vision/stable/models.html#id15
        print("4、目前计算至：densenet121")
        print("")
    elif i==4:
        net = models.resnet34(pretrained=True)  # 设定网络为resnet34
        print("5、目前计算至：resnet134")
        print("")
    elif i==5:
        net = models.efficientnet_b1(pretrained=True)
        print("6、目前计算至：efficientnet_b1")
        print("")
    elif i==6:
        net = models.googlenet(pretrained=True)  # 15x15
        print("7、目前计算至：googlenet")
        print("")

    # Switch to evaluation mode
    net.eval()#测试模型
    net=net.cuda()

    time = 0
    num_classes = 1000  # 分类数
    #r, loop_i, label_orig, label_pert, pert_image, time = deepfool(im, net, num_classes)  # deepfool执行，输入图片与神经网络参数
    #r, loop_i, label_orig, label_pert, pert_image,time = deepfool_t(im, net, num_classes)#deepfool_t执行，输入图片与神经网络参数
    '''返回的扰动r是numpy类型，nsga2如何处理？'''
    f_image = net.forward(Variable(total_image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()  # 先得到原图的参数，然后向前传播

    I = (np.array(f_image)).flatten().argsort()[::-1]  # argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I = I[0:1000]  # 分类以及分类器的编号，这里返回了分类器原标签的编号
    label = I[0]  # 取最大的编号，得到标签
    print("得到标签：", label)
    print('')



