import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
from deepfool_target import deepfool_t
from deepfool import deepfool
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import cv2
import scipy.misc
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库
import copy
import time
#开始程序，获得一排扰动

'''
modelpath="./model/"
net=torch.load(modelpath)#从本地导入神经网络模型
'''

def deepfool_get_grad(image,net, num_classes, overshoot=0.02, max_iter=50,fangcha=0.001):
    print('start get grad')

    starttime = time.perf_counter ()

    f_image = net.forward(Variable(image[None, :, :, :])).data.cpu().numpy().flatten()#先得到原图的参数，然后向前传播

    I = (np.array(f_image)).flatten().argsort()[::-1]#argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I = I[0:num_classes]#分类以及分类器的编号，这里返回了分类器原标签的编号
    label = I[0]#取最大的编号，得到标签
    print("扰动前标签：",label)

    ori_label=label#ori_label:原本判断的标签

    input_shape = image.cpu ().numpy ().shape  # 把图片变成参数
    pert_image = copy.deepcopy (image)  # 把图片设为原图片
    r_i=np.zeros((input_shape))
    r_tot = np.zeros (input_shape)  # 初始化扰动，将扰动设为输入的图片参数

    loop_i = 0  # 初始化循环次数

    # get_grad
    pred_grad=np.zeros(input_shape)
    t=6#t为目标分类器的值

    while label == ori_label and loop_i < max_iter:  # 进入循环，开始叠加扰动。当分类器标签不变时以及迭代未超过最大迭代次数时
        loop_iter=0
        for i in range(max_iter):
            ui=torch.randn(input_shape)
            ui=ui.detach().numpy()
            x1=pert_image+torch.from_numpy(fangcha*ui)
            x2=pert_image-torch.from_numpy(fangcha*ui)
            #计算x1的概率值
            out1 = net.forward (Variable(x1[None, :]))  # 前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定,这里是输出各个分类器参数
            fs_list = [out1[0, I[k]] for k in range (num_classes)]  # 初始化fs[0,I[k]]，存分类器
            out1=out1.detach().numpy()
            temp1=out1
            out1=out1[0,t]
            out1=abs(out1)
            while out1>1:
                out1=out1/10
            #计算x2的在目标分类器上的概率值
            out2 = net.forward (Variable(x2[None, :]))  # 前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定,这里是输出各个分类器参数
            fs_list = [out2[0, I[k]] for k in range (num_classes)]  # 初始化fs[0,I[k]]，存分类器
            out2=out2.detach().numpy()
            out2=out2[0,t]
            out2=abs(out2)
            while out2>1:
                out2=out2/10
            loop_iter+=1
            print('内部迭代次数:',loop_iter)
            pred_grad=pred_grad+(out1*ui)
            pred_grad=pred_grad-(out2*ui)

        r_i=np.mean(r_i)
        r_i=np.linalg.norm(r_i.flatten ())
        r_i=torch.tensor(r_i)
        pert_image=pert_image-(abs(r_i)/r_i)
        #r_tot = np.float32 (r_tot + r_i)  # 得到总梯度

        x = Variable (pert_image)  # x存入被扰动的图片
        fs = net.forward (x[None, :])  # 向前传播，得到各个分类器的值
        label = np.argmax (fs.data.cpu ().numpy ().flatten ())  # 把数组折叠成一维，选最大的那个分类器
        print('当前label:',label)
        loop_i += 1

    r_tot = (1 + overshoot) * r_tot  # 总扰动加上一个小扰动

    endtime = time.perf_counter ()
    print ("扰动后标签：", label)
    timedate = endtime - starttime
    print('迭代次数为:',loop_i)
    # print(r_tot)
    # 返回总扰动、循环次数、原标签、扰动后的分类器标签、扰动图片
    print('')
    print('')
    return r_tot, loop_i, ori_label, label, pert_image, timedate

#模型选择
#net=models.alexnet(pretrained=True)#有面数限制,最小为63x63
#net=models.vgg11(pretrained=True)#支持最小输入尺寸为32x32，vgg最小都是32x32
#net=models.squeezenet1_1(pretrained=True)#1.0最小尺寸为21x21，1.1最小为17x17
#net=models.densenet121(pretrained=True)#全为29x29，参数看文档https://pytorch.org/vision/stable/models.html#id15
#net = models.resnet34(pretrained=True)#设定网络为resnet34
#net=models.efficientnet_b1(pretrained=True)
net=models.googlenet(pretrained=True)#15x15


net.eval()#测试模型

#PyTorch官方文档推荐的正则化参数
mean = [ 0.485, 0.456, 0.406 ]#均值，三分量顺序是RGB
std = [ 0.229, 0.224, 0.225 ]#方差

#图片处理
testimg=Image.open("D:/毕设/DeepFool/1.jpg")

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
                        #transforms.CenterCrop(224)]
                         ])

#start
num_classes=1000#分类数

total=10
corret=0
timeA=0
tot_time=0
start=1
root_dir='D:/毕设/DeepFool/test/'

for i in range(start,total+start):
    im1 = transforms.Compose([
        transforms.Scale(256),  # 图片尺寸
        transforms.CenterCrop(224),  # 进行中心裁剪
        transforms.ToTensor(),  # 转换图片格式，变成张量
        transforms.Normalize(mean=mean, std=std)])(testimg)  # 对原图进行标准化操作
    img=Image.open("D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_{:0>8d}.JPEG".format(int(i)))#用ImageNet2017测试集，共5500张
    #img = Image.open('D:/毕设/DeepFool/test/adversarial sample/testpicture_00000001.jpeg')#.convert('RGB')
    #img=Variable(img[None, :, :, :], requires_grad=True)
    #input_shape = img.cpu().numpy().shape#把图片变成参数
    #mean_rot = np.zeros(input_shape)

    print('')
    print('正在测试第 ',i,' 张图片')
    imginput = transforms.Compose([transforms.Scale(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=mean, std=std)])(img)  # 对输入图片进行正则化处理

    #r, loop_i, label_orig, label_pert, pert_image,time = deepfool_t(imginput, net, num_classes)#deepfool_t执行，输入图片与神经网络参数
    r, loop_i, label_orig, label_pert, pert_image, timeA = deepfool_get_grad(imginput, net, num_classes)
    #r, loop_i, label_orig, label_pert, pert_image, time = deepfool(imginput, net,num_classes)  # deepfool执行，输入图片与神经网络参数
    tot_time+=timeA
    if label_orig!=label_pert:
        corret+=1
    '''
    test_image=imginput +(1+0.02)* torch.from_numpy(r)
    pic=plt.figure()# 5，3.5分别对应宽和高
    #pic.set_size_inches(4, 4)
    plt.imshow(tf(test_image.cpu()[0]))
    plt.axis('off')
    plt.savefig("D:/毕设/DeepFool/test/adversarial sample/testpicture_{:0>8d}.jpeg".format(int(i)),bbox_inches='tight',pad_inches=0.0,dpi=pic.dpi)
    plt.close()
    '''
    '''
    im1 = im1 + (1+0.02) * torch.from_numpy(r)
    #plt.imsave("D:/毕设/DeepFool/test/testpicture_{:0>8d}.JPEG".format(int(i)),tf(im1.cpu()[0]))
    #picName = root_dir + str(i) + '.jpg'  # 格式：标签名_序号
    #cv2.imwrite(picName, tf(im1.cpu()[0]))
    #scipy.misc.imsave("D:/毕设/DeepFool/test/testpicture_{:0>8d}.JPEG".format(int(i)),tf(im1.cpu()[0]))

    pic=plt.figure()# 5，3.5分别对应宽和高
    #pic.set_size_inches(4, 4)
    plt.imshow(tf(im1.cpu()[0]))
    plt.axis('off')
    plt.savefig("D:/毕设/DeepFool/test/testpicture_{:0>8d}.png".format(int(i)),bbox_inches='tight',pad_inches=0.0,dpi=pic.dpi)
    plt.close()
    #plt.show()
    '''

#cv2.imwrite('rot.png', r)
#mean_rot=mean_rot/total
acc=corret/total
tot_time=tot_time/total
print('攻击成功率为：{:2.3f}%'.format(acc))
print('每张图片进行攻击的平均时间为：{:2.5f}s'.format(tot_time))
#print('平均扰动为：{:2.5f}s'.format(mean_rot))




