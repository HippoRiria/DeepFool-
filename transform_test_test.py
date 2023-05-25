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
import time



def deepfool_get_grad(image,net, num_classes, overshoot=0.02, max_iter=10,fangcha=0.001):
    print('start get grad')

    starttime = time.perf_counter ()

    f_image = net.forward(Variable(image[None,:,:,:])).data.cpu().numpy().flatten()#先得到原图的参数，然后向前传播

    I = (np.array(f_image)).flatten().argsort()[::-1]#argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I = I[0:num_classes]#分类以及分类器的编号，这里返回了分类器原标签的编号
    label = I[0]#取最大的编号，得到标签
    print("扰动前标签：",label)

    ori_label=label#ori_label:原本判断的标签

    input_shape = image.cpu ().numpy ().shape  # 把图片变成参数
    pert_image = copy.deepcopy (image)  # 把图片设为原图片
    r_i=np.zeros((input_shape))
    r_tot=np.zeros((input_shape))

    loop_i = 0  # 初始化循环次数
    t=8
    # get_grad
    pred_grad=np.zeros(input_shape)

    while label == ori_label and loop_i < max_iter:  # 进入循环，开始叠加扰动。当分类器标签不变时以及迭代未超过最大迭代次数时
        loop_iter=0
        for i in range(10):
            ui=torch.randn(input_shape)
            ui=ui.detach().numpy()
            x1=pert_image+torch.from_numpy(fangcha*ui)
            x2=pert_image-torch.from_numpy(fangcha*ui)
            #计算x1的概率值
            out1 = net.forward (Variable(x1[None,:,:,:]))  # 前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定,这里是输出各个分类器参数
            fs_list = [out1[0, I[k]] for k in range (num_classes)]  # 初始化fs[0,I[k]]，存分类器
            out1=out1.detach().numpy()
            out1=out1[0,t-1]
            out1=abs(out1)
            while out1>1:
                out1=out1/10
            #计算x2的在目标分类器上的概率值
            out2 = net.forward (Variable(x2[None,:,:,:]))  # 前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定,这里是输出各个分类器参数
            fs_list = [out2[0, I[k]] for k in range (num_classes)]  # 初始化fs[0,I[k]]，存分类器
            out2=out2.detach().numpy()
            out2=out2[0,t-1]
            out2=abs(out2)
            while out2>1:
                out2=out2/10
            loop_iter+=1
            #print('内部迭代次数:',loop_iter)

            pred_grad=pred_grad+(out1*ui)
            pred_grad=pred_grad-(out2*ui)

        r_i=pred_grad/(2*max_iter*fangcha)
        l2_ri=np.linalg.norm(r_i)
        r_i=torch.tensor(r_i)
        l2_ri=torch.tensor(l2_ri)

        #r_tot+=r_i/l2_ri
        pert_image=pert_image-(r_i/l2_ri)
        pert_image = pert_image.to(torch.float32)
        #r_tot = np.float32 (r_tot + r_i)  # 得到总梯度

        x = Variable (pert_image[None,:,:,:])  # x存入被扰动的图片
        fs = net.forward (x)  # 向前传播，得到各个分类器的值
        label = np.argmax (fs.data.cpu ().numpy ().flatten ())  # 把数组折叠成一维，选最大的那个分类器
        print('当前label:',label)
        loop_i += 1

    endtime = time.perf_counter ()
    print ("扰动后标签：", label)
    timedate = endtime - starttime
    print('迭代次数为:',loop_i)
    # print(r_tot)
    # 返回总扰动、循环次数、原标签、扰动后的分类器标签、扰动图片
    print('')
    print('')
    return r_tot, loop_i, ori_label,label, pert_image, timedate

#net2=models.alexnet(pretrained=True)#有面数限制,最小为63x63
net1=models.vgg11(pretrained=True)#支持最小输入尺寸为32x32，vgg最小都是32x32
#net2=models.squeezenet1_1(pretrained=True)#1.0最小尺寸为21x21，1.1最小为17x17
#net2=models.densenet121(pretrained=True)#全为29x29，参数看文档https://pytorch.org/vision/stable/models.html#id15
net2 = models.resnet34(pretrained=True)  # 设定网络为resnet34
#net2=models.efficientnet_b1(pretrained=True)
#net2=models.googlenet(pretrained=True)#15x15


# 模型选择
#net1 = models.resnet34(pretrained=True)#设定网络为resnet34
#net2=models.resnet50(pretrained=True)
#net2=models.resnet101(pretrained=True)
#net2 = models.vgg11 (pretrained=True)
#net1 = models.vgg19 (pretrained=True)
#net2=models.vgg13(pretrained=True)

net1.eval ()  # 测试模型
net2.eval ()

#PyTorch官方文档推荐的正则化参数
mean = [ 0.485, 0.456, 0.406 ]#均值，三分量顺序是RGB
std = [ 0.229, 0.224, 0.225 ]#方差

#start
num_classes=1000#分类数

total=20
corret=0
timeA=0
tot_time=0
start=1

for i in range(start,total+start):
    img=Image.open("D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_{:0>8d}.JPEG".format(int(i)))#用ImageNet2017测试集，共5500张

    print('')
    print('正在测试第 ',i,' 张图片')
    imginput = transforms.Compose([transforms.Scale(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=mean, std=std)])(img)  # 对输入图片进行正则化处理

    #r, loop_i, label_orig, label_pert, pert_image,timeA = deepfool_t(imginput, net1, num_classes)#deepfool_t执行，输入图片与神经网络参数
    r, loop_i, label_orig, label_pert, pert_image, timeA = deepfool_get_grad(imginput, net1, num_classes)
    #r, loop_i, label_orig, label_pert, pert_image, timeA = deepfool(imginput, net1,num_classes)  # deepfool执行，输入图片与神经网络参数
    tot_time+=timeA

    # 输出标签
    label1 = label_pert
    label1_ori = label_orig
    print ('label1 is:', label1)
    print ('')

    # Switch to evaluation mode
    net2.eval ()  # 测试模型
    pert_image=pert_image.cuda()
    net = net2.cuda ()
    #x = Variable (pert_image, requires_grad=True)  # x存入被扰动的图片
    x = Variable (pert_image[None,:])  # x存入被扰动的图片
    fs = net.forward (x)  # 向前传播，得到各个分类器的值
    label2 = np.argmax (fs.data.cpu ().numpy ().flatten ())  # 把数组折叠成一维，选最大的那个分类器
    print ('label2 is:', label2)
    print ('')

    im = imginput.cuda ()
    f_image = net.forward (Variable (im[None, :, :, :], requires_grad=True)).data.cpu ().numpy ().flatten ()  # 先得到原图的参数，然后向前传播

    I = (np.array (f_image)).flatten ().argsort ()[::-1]  # argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I = I[0:num_classes]  # 分类以及分类器的编号，这里返回了分类器原标签的编号
    label2_org = I[0]  # 取最大的编号，得到标签
    print ('label2_org is:', label2_org)
    print ('')
    if label2_org!=label2:
        corret+=1

print('欺骗成功率为:',corret/total)