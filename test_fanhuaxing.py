import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
import os
from models_a import *
from deepfool_target import deepfool_t
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库
from torch.autograd import Variable

#开始程序


print('start')

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
# net=models.alexnet(pretrained=True)#有面数限制,最小为63x63
# net=models.vgg11(pretrained=True)#支持最小输入尺寸为32x32，vgg最小都是32x32
# net=models.squeezenet1_1(pretrained=True)#1.0最小尺寸为21x21，1.1最小为17x17
# net=models.densenet121(pretrained=True)#全为29x29，参数看文档https://pytorch.org/vision/stable/models.html#id15
net = models.resnet34(pretrained=True)  # 设定网络为resnet34
# net=models.efficientnet_b1(pretrained=True)
# net=models.googlenet(pretrained=True)#15x15

# PyTorch官方文档推荐的正则化参数
mean = [0.485, 0.456, 0.406]  # 均值，三分量顺序是RGB
std = [0.229, 0.224, 0.225]  # 方差

# Switch to evaluation mode
net.eval()  # 测试模型
image_tot=0
start=1
for i in range(start, 1000):
    # 导入图片
    # image_o = Image.open('D:/毕设/pictures/组合图片.jpg')#导入图片
    #image_o = Image.open('D:/毕设/DeepFool/test/adversarial sample/testpicture_00000001.jpeg')#.convert('RGB') .png
    #image_o = Image.open('D:/毕设/DeepFool/test/testpicture_00000001.png').convert('RGB') #.png
    # image_o = Image.open('D:/毕设/pictures/扰动样本d.jpg')#导入图片
    # image_o = Image.open('D:/毕设/pictures/ILSVRC2017_test_00000143.JPEG')#导入图片
    # im_orig=Image.open('D:/毕设/DeepFool/test_im1.jpg')
    #image_o = Image.open('D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_00000071.JPEG')#导入图片
    image_o=Image.open("D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_{:0>8d}.JPEG".format(int(i)))#用ImageNet2017测试集，共5500张

    # 去掉平均值
    image = transforms.Compose([
        transforms.Scale(256),  # 图片尺寸
        transforms.ToTensor(),  # 转换图片格式，变成张量
        transforms.CenterCrop(224),  # 进行中心裁剪
        transforms.Normalize(mean=mean, std=std)])(image_o)  # 对原图进行标准化操作

    is_cuda = torch.cuda.is_available()  # 引入GPU计算

    if is_cuda:
        print("使用 GPU 计算")
        image = image.cuda()
        net = net.cuda()
    else:
        print("使用 CPU 计算")  # 选择用GPU跑还是CPU跑

    if i==1:
        image_tot=image
        continue
    #test
    f_image = net.forward(
        Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()  # 先得到原图的参数，然后向前传播

    I = (np.array(f_image)).flatten().argsort()[::-1]  # argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I = I[0:1000]  # 分类以及分类器的编号，这里返回了分类器原标签的编号
    label = I[0]  # 取最大的编号，得到标签
    print("得到标签：", label)
    print('iter:',i)
    if label==291:
        print('number is :',i)
    #image_tot=torch.add(image_tot,image)

image_tot=torch.div(image_tot,3)
f_image = net.forward(Variable(image_tot[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()  # 先得到原图的参数，然后向前传播

I = (np.array(f_image)).flatten().argsort()[::-1]  # argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

I = I[0:1000]  # 分类以及分类器的编号，这里返回了分类器原标签的编号
label = I[0]  # 取最大的编号，得到标签

#r, loop_i, label_orig, label_pert, pert_image, time = deepfool_t(image, net, num_classes=1000)  # deepfool_t执行，输入图片与神经网络参数
print("得到标签：", label)
#print('r_tot label:',label_pert)



# 图片展示
def clip_tensor(A, minv, maxv):  # 切面张量
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A


clip = lambda x: clip_tensor(x, 0, 255)  # lambda函数减少代码冗余，把函数缩小为一行

# tf定义，把Tensor后的图片给转换回来
tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                         transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                         transforms.Lambda(clip),
                         transforms.ToPILImage(),
                         transforms.CenterCrop(224)])

# 图片输出
imgae = image.unsqueeze_(0)
imgae_tot = image.unsqueeze_(0)
plt.figure()
# plt.imshow(tf(im.cpu()[0]))
# plt.imshow(tf(r.cpu()[0]))
#plt.imshow(tf(image.cpu()[0]))
plt.imshow(tf(image_tot.cpu()[0]))
# plt.imsave('1.jpg',r)
plt.title(label)
plt.show()
