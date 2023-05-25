import copy
#from torch.autograd.gradcheck import zero_gradients#梯度归零函数，pytorch1.9.0后被删除
#梯度归零换为x.grad.zero_()   x为nn模型
import time
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
from deepfool_target import deepfool_t
from deepfool import  deepfool
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import cv2
import scipy.misc
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库


is_cuda = torch.cuda.is_available ()  # 引入GPU计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片的值
mean = [0.485, 0.456, 0.406]  # 均值，三分量顺序是RGB，后期要调整，opencv和numpy可以自己算得出来
std = [0.229, 0.224, 0.225]  # 方差

#图片展示
def clip_tensor(A, minv, maxv):#切面张量
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)#lambda函数减少代码冗余，把函数缩小为一行


#tf定义，把Tensor后的图片给转换回来
tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage()])



testimg=Image.open("D:/毕设/DeepFool/1.jpg")


im1 = transforms.Compose([
    transforms.Scale(256),#图片尺寸
    transforms.CenterCrop(224),#进行中心裁剪
    transforms.ToTensor(),#转换图片格式，变成张量
    transforms.Normalize(mean = mean,std = std)])(testimg)#导入白板图

# 导入图片
# image_o = Image.open('D:/毕设/pictures/扰动样本.jpg')#导入图片
# image_o = Image.open('D:/毕设/pictures/扰动样本d.jpg')#导入图片
image_o = Image.open('D:/毕设/pictures/ILSVRC2017_test_00000143.JPEG')  # 导入图片
# im_orig=Image.open('D:/毕设/DeepFool/test_im1.jpg')


# 去掉平均值
im = transforms.Compose([
    transforms.Scale(256),  # 图片尺寸
    transforms.CenterCrop(224),  # 进行中心裁剪
    transforms.ToTensor(),  # 转换图片格式，变成张量
    transforms.Normalize(mean=mean, std=std)])(image_o)  #对原图进行标准化操作

net = models.alexnet (pretrained=True)  # 有面数限制,最小为63x63
print ("1、首先计算：alexnet 的扰动并生成图片")
print ("")
net.eval ()  # 测试模型

time = 0
num_classes = 1000  # 分类数
r, loop_i, label_orig, label_pert, pert_image, time = deepfool(im, net, num_classes)  # deepfool执行，输入图片与神经网络参数

r=abs (r) / np.linalg.norm (r.flatten ())#单位化扰动
r_tot=r
iteration=0
max_iter=100


for i in range(1,7):
    if i==1:
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
    x = Variable (pert_image, requires_grad=True)  # x存入被扰动的图片
    fs = net.forward (x)  # 向前传播，得到各个分类器的值
    label = np.argmax (fs.data.cpu ().numpy ().flatten ())  # 把数组折叠成一维，选最大的那个分类器
    print('start iteration')
    while (label==label_orig) and (iteration<max_iter):
        r_tot = np.float32(r_tot + r)#得到总梯度
        pert_image = pert_image + (1 + 0.02) * torch.from_numpy (r).cuda ()

        x = Variable (pert_image, requires_grad=True)  # x存入被扰动的图片
        fs = net.forward (x)  # 向前传播，得到各个分类器的值
        label = np.argmax (fs.data.cpu ().numpy ().flatten ())  # 把数组折叠成一维，选最大的那个分类器

        iteration+=1
        print('iter=',iteration)
    print('end this')
    iteration=0
    print('')

for i in range(7):
    if  i==0:
        net = models.alexnet (pretrained=True)  # 有面数限制,最小为63x63
        print ("1、首先计算：alexnet 的扰动并生成图片")
        print ("")
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
    x = Variable (pert_image, requires_grad=True)  # x存入被扰动的图片
    fs = net.forward (x)  # 向前传播，得到各个分类器的值
    label = np.argmax (fs.data.cpu ().numpy ().flatten ())  # 把数组折叠成一维，选最大的那个分类器
    print('label is:',label)
    print('')

im1 = im1 + (1 + 0.02) * torch.from_numpy (r_tot)


#图片输出
plt.figure(1)
plt.imshow(tf(pert_image.cpu()[0]))
plt.title('pert_image')
plt.show()

#扰动输出
plt.figure(2)
plt.imshow(tf(im1.cpu()[0]))
plt.title('r')
plt.show()
