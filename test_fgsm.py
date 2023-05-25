import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
from deepfool import deepfool#导入deepfool算法
from deepfool_target import deepfool_t
import os
import copy
from torch.autograd import Variable
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库

#开始程序

'''
modelpath="./model/"
net=torch.load(modelpath)#从本地导入神经网络模型
'''
#模型选择
#net = models.resnet34(pretrained=True)#设定网络为resnet34
#net=models.alexnet(pretrained=True)
#net=models.vgg11(pretrained=True)
net=models.googlenet(pretrained=True)

# Switch to evaluation mode
net.eval()#测试模型

#导入图片
im_orig = Image.open('D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_00000083.JPEG')#导入图片

#图片的值
mean = [ 0.485, 0.456, 0.406 ]#均值，三分量顺序是RGB，后期要调整，opencv和numpy可以自己算得出来
std = [ 0.229, 0.224, 0.225 ]#方差


#去掉平均值
im = transforms.Compose([
    transforms.Scale(256),#图片尺寸
    transforms.CenterCrop(224),#进行中心裁剪
    transforms.ToTensor(),#转换图片格式，变成张量
    transforms.Normalize(mean = mean,std = std)])(im_orig)#对原图进行标准化操作


time=0;
num_classes=1000#分类数
#r, loop_i, label_orig, label_pert, pert_image,time = deepfool(im, net, num_classes)#deepfool执行，输入图片与神经网络参数
#r, loop_i, label_orig, label_pert, pert_image,time = deepfool_t(im, net, num_classes)#deepfool_t执行，输入图片与神经网络参数
'''
labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')#标签导入

#str_label_orig = labels[np.int(label_orig)].split(',')[0]#原标签
str_label_orig = labels[np.int(label_orig)]
str_label_pert = labels[np.int(label_pert)].split(',')[0]#欺骗后标签

#输出标签
print("原标签 = ", str_label_orig)
print("扰动添加后的标签 = ", str_label_pert)

'''

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


def fgsm_attack(image, epsilon, net):
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()  # 先得到原图的参数，然后向前传播

    I = (np.array(f_image)).flatten().argsort()[::-1]  # argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I = I[0:num_classes]  # 分类以及分类器的编号
    label = I[0]  # 得到标签
    print("扰动前标签：", label)

    input_shape = image.cpu().numpy().shape  # 把图片变成参数
    pert_image = copy.deepcopy(image)  # 把图片设为原图片
    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs[0, I[0]].backward(retain_graph=True)  # 沿梯度向后传播
    grad_orig = x.grad.data.cpu().numpy().copy()  # 原梯度
    data_grad = grad_orig
    # 收集数据梯度的元素符号
    r= np.zeros(input_shape)

    sign_data_grad = np.sign(data_grad)
    perturbed_image=np.zeros(input_shape)
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = np.float32(image + epsilon*sign_data_grad)
    r=np.float32(r+epsilon*sign_data_grad)
    # 添加剪切以维持[0,1]范围
    #perturbed_image = torch.clamp(tf(perturbed_image.cpu()[0]), 0, 1)
    # 返回被扰动的图像
    return perturbed_image,r

pert_image,r=fgsm_attack(im,0.1,net)#步长

#print(r)
testimg=Image.open("D:/毕设/DeepFool/1.jpg")

im = transforms.Compose([
    transforms.Scale(256),#图片尺寸
    transforms.CenterCrop(224),#进行中心裁剪
    transforms.ToTensor(),#转换图片格式，变成张量
    transforms.Normalize(mean = mean,std = std)])(testimg)#对原图进行标准化操作
#testimg=testimg[None, :, :, :]
#test=copy.deepcopy(testimg)
#im = im + (1)*torch.from_numpy(r)
im=im+r

#图片展示

np.squeeze(im)
print(pert_image)
#图片输出
pert_image=torch.from_numpy(pert_image)
plt.figure()
#plt.imshow(tf(r.cpu()[0]))
plt.imshow(tf(pert_image.cpu()[0]))
#plt.imshow(tf(im.cpu()[0]))
#plt.imsave('1.jpg',r)
#plt.title(str_label_pert)
plt.title(1)
plt.show()

