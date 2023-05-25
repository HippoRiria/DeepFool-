import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
from deepfool import deepfool#导入deepfool算法
from deepfool_target import deepfool_t
from torch.autograd import Variable
import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库

#开始程序

'''
modelpath="./model/"
net=torch.load(modelpath)#从本地导入神经网络模型
'''

#模型选择
#net=models.alexnet(pretrained=True)#有面数限制,最小为63x63
net=models.vgg11(pretrained=True)#支持最小输入尺寸为32x32，vgg最小都是32x32
#net=models.squeezenet1_1(pretrained=True)#1.0最小尺寸为21x21，1.1最小为17x17
#net=models.densenet121(pretrained=True)#全为29x29，参数看文档https://pytorch.org/vision/stable/models.html#id15
#net = models.resnet34(pretrained=True)#设定网络为resnet34
#net=models.efficientnet_b1(pretrained=True)
#net=models.googlenet(pretrained=True)#15x15


net.eval()#测试模型

#PyTorch官方文档推荐的正则化参数
mean = [ 0.485, 0.456, 0.406 ]#均值，三分量顺序是RGB
std = [ 0.229, 0.224, 0.225 ]#方差

#自设定
mean1=[0.5,0.5,0.5]
std1=[0.5,0.5,0.5]

classes = ("airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck")#cifar10



#start
num_classes=1000#分类数

total=5
corret=0
time=0
tot_time=0
start=140
for i in range(start,total+start):
    #img = Image.open("D:/毕设/DeepFool/data/cifar10/test(num)/{}.jpg".format(str(i)))#导入图片，来自cifar10测试集，10000张
    #img = Image.open("D:/毕设/DeepFool/data/cifar10/train(num)/{}.jpg".format(str(i)))  # 导入图片，来自cifar10训练集，50000张
    img=Image.open("D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_{:0>8d}.JPEG".format(int(i)))#用ImageNet2017测试集，共5500张
    #img=Variable(img[None, :, :, :], requires_grad=True)
    #input_shape = img.cpu().numpy().shape#把图片变成参数
    #mean_rot = np.zeros(input_shape)

    print('')
    print('正在测试第 ',i,' 张图片')
    img = transforms.Compose([transforms.Scale(32),transforms.ToTensor(),transforms.Normalize(mean = mean,std = std)])(img)#对输入图片进行正则化处理
    #img = transforms.Compose([transforms.Scale(256),transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(img)  # 对输入图片进行正则化处理

    #r, loop_i, label_orig, label_pert, pert_image,time = deepfool(img, net, num_classes)  # deepfool执行，输入图片与神经网络参数
    r, loop_i, label_orig, label_pert, pert_image,time = deepfool_t(img, net, num_classes)#deepfool_t执行，输入图片与神经网络参数
    tot_time+=time
    if label_orig!=label_pert:
        corret+=1
    #mean_rot+=r


#cv2.imwrite('rot.png', r)
#mean_rot=mean_rot/total
acc=corret/total
tot_time=tot_time/total
print('攻击成功率为：{:2.3f}%'.format(acc))
print('每张图片进行攻击的平均时间为：{:2.5f}s'.format(tot_time))
#print('平均扰动为：{:2.5f}s'.format(mean_rot))
