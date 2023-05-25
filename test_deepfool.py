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
from torch.autograd import Variable
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

#模型选择
net = models.resnet34(pretrained=True)#设定网络为resnet34
#net=models.alexnet(pretrained=True)
#net=models.vgg11(pretrained=True)
#net=models.googlenet(pretrained=True)#15x15
#net = models.squeezenet1_1(pretrained=True)  # 1.0最小尺寸为21x21，1.1最小为17x17

# Switch to evaluation mode
net.eval()#测试模型

#导入图片
im_orig = Image.open('D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_00000012.JPEG')#导入图片 143、
#im_orig=Image.open('D:/毕设/DeepFool/test_im1.jpg')
#图片的值
mean = [ 0.485, 0.456, 0.406 ]#均值，三分量顺序是RGB，后期要调整，opencv和numpy可以自己算得出来
std = [ 0.229, 0.224, 0.225 ]#方差


#去掉平均值
im = transforms.Compose([
    transforms.Scale(256),#图片尺寸
    transforms.CenterCrop(224),#进行中心裁剪
    transforms.ToTensor(),#转换图片格式，变成张量
    transforms.Normalize(mean = mean,std = std)])(im_orig)#对原图进行标准化操作


time=0
num_classes=1000#分类数
r, loop_i, label_orig, label_pert, pert_image,time = deepfool(im, net, num_classes)#deepfool执行，输入图片与神经网络参数
#r, loop_i, label_orig, label_pert, pert_image,time = deepfool_t(im, net, num_classes)#deepfool_t执行，输入图片与神经网络参数

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')#标签导入

#str_label_orig = labels[np.int(label_orig)].split(',')[0]#原标签
str_label_orig = labels[np.int(label_orig)]
str_label_pert = labels[np.int(label_pert)].split(',')[0]#欺骗后标签

#输出标签
print("原标签 = ", str_label_orig)
print("扰动添加后的标签 = ", str_label_pert)




#print(r)
testimg=Image.open("D:/毕设/DeepFool/1.jpg")


im = transforms.Compose([
    transforms.Scale(256),#图片尺寸
    transforms.CenterCrop(224),#进行中心裁剪
    transforms.ToTensor(),#转换图片格式，变成张量
    transforms.Normalize(mean = mean,std = std)])(testimg)#对原图进行标准化操作
#testimg=testimg[None, :, :, :]
#test=copy.deepcopy(testimg)
#print(im)

im = im + (100+0.2)*torch.from_numpy(r)#deepfool算法
#im = im + (1+0.02)*torch.from_numpy(r)#deepfool_target




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
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

#图片输出
plt.figure(1)
plt.imshow(tf(pert_image.cpu()[0]))
plt.title('pert_image')
plt.show()

#扰动输出
plt.figure(2)
plt.imshow(tf(im.cpu()[0]))
plt.title('r_tot')
plt.show()
