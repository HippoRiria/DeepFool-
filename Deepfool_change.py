import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
#from torch.autograd.gradcheck import zero_gradients#梯度归零函数，pytorch1.9.0后被删除
#梯度归零换为x.grad.zero_()   x为nn模型
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库
import time


def deepfool_change(image1,image2, net, num_classes, overshoot=0.02, max_iter=500):#图片、网络参数、分类数、最后添加的小扰动、最大迭代次数

    is_cuda = torch.cuda.is_available()#引入GPU计算

    if is_cuda:
        print("使用 GPU 计算")
        image1 = image1.cuda()
        image2=image2.cuda()
        net = net.cuda()
    else:
        print("使用 CPU 计算")#选择用GPU跑还是CPU跑

#初始化
    starttime=time.perf_counter()

    f_image1 = net.forward(Variable(image1[None, :, :, :])).data.cpu().numpy().flatten()#先得到原图的参数，然后向前传播

    I1 = (np.array(f_image1)).flatten().argsort()[::-1]#argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I1 = I1[0:num_classes]#分类以及分类器的编号，这里返回了分类器原标签的编号
    label1 = I1[0]#取最大的编号，得到标签
    label_change1=I1[1]#第二大的标签，要得到的就是他
    print("预测标签：",label1)
    print("第二大的标签：",label_change1)

    input_shape1 = image1.cpu().numpy().shape#把图片变成参数
    pert_image1 = copy.deepcopy(image1)#把图片设为原图片
    w1 = np.zeros(input_shape1)#得到分类器的权重值，初始化
    r_tot1 = np.zeros(input_shape1)#初始化扰动，将扰动设为输入的图片参数
    x1 = Variable(pert_image1[None, :], requires_grad=True)#将图片参数转换，因为神经网络只能输入variable.[None, :]，保证数据不改变的情况下,追加一个新维度。
    #test=Variable(pert_image[None, :])
    fs1 = net.forward(x1)#前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定

#picture 2
    f_image2 = net.forward(Variable(image2[None, :, :, :])).data.cpu().numpy().flatten()#先得到原图的参数，然后向前传播

    I2 = (np.array(f_image2)).flatten().argsort()[::-1]#argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I2 = I2[0:num_classes]#分类以及分类器的编号，这里返回了分类器原标签的编号
    label2 = I2[0]#取最大的编号，得到标签
    label_change2=I2[1]#第二大的标签，要得到的就是他
    print("预测标签：",label2)
    print("第二大的标签：",label_change2)

    input_shape2 = image1.cpu().numpy().shape#把图片变成参数
    pert_image2 = copy.deepcopy(image1)#把图片设为原图片
    w2 = np.zeros(input_shape2)#得到分类器的权重值，初始化
    r_tot2 = np.zeros(input_shape2)#初始化扰动，将扰动设为输入的图片参数
    x2 = Variable(pert_image2[None, :], requires_grad=True)#将图片参数转换，因为神经网络只能输入variable.[None, :]，保证数据不改变的情况下,追加一个新维度。
    #test=Variable(pert_image[None, :])
    fs2 = net.forward(x2)#前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定
    loop_i = 0#初始化循环次数

    k_i = label1#标签

#deepfool_change的优化计算算法
    while (k_i == label1 or k_i !=label_change1) and loop_i < max_iter:#进入循环，开始叠加扰动。当分类器标签不变时以及迭代未超过最大迭代次数时
        pert = np.inf#np.inf表示正无穷，初始化扰动
        fs1[0, I1[0]].backward(retain_graph=True)#沿梯度向后传播,自动计算原分类器梯度
        pic1_orig = x1.grad.data.cpu().numpy().copy()#图片1的梯度
        fs2[0, I2[0]].backward(retain_graph=True)#沿梯度向后传播,自动计算原分类器梯度
        pic2_orig = x2.grad.data.cpu().numpy().copy()#图片1的梯度
        fs1[0, I2[0]].backward(retain_graph=True)  # 继续向后传播，沿图片2的分类器向后传播

        for k in range(1, num_classes):#多分类常规情况
            x1.grad.zero_()#将梯度参数归零

            fs1[0, I1[k]].backward(retain_graph=True)#继续向后传播，沿目标分类器k向后传播
            cur_grad = x.grad.data.cpu().numpy().copy()#得到在此分类器下的梯度参数

            w_k = cur_grad - grad_orig#线性化分类器，通过分类器的梯度以及原梯度线性化分类器
            f_k = (fs1[0, I1[k]] - fs1[0, I1[0]]).data.cpu().numpy()#得到线性化后的分类器参数

            pert_k1 = abs(f_k)/np.linalg.norm(w_k.flatten())#求该分类器的扰动，绝对值

            if pert_k1 < pert:#选各个分类器中最小的扰动以及参数
                pert = pert_k1
                w = w_k


        r_i =(pert+1e-4) * w / np.linalg.norm(w)#叠加第i次扰动，1e-4=0.0001，这个数可以保持数组的稳定性
        r_tot = np.float32(r_tot + r_i)#得到总梯度





        if is_cuda:#根据用GPU还是CPU算出来的数据叠加扰动至图片
            pert_image = image1 + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image1 + (1+overshoot)*torch.from_numpy(r_tot)
        x = Variable(pert_image, requires_grad=True)#x存入被扰动的图片
        fs = net.forward(x)#向前传播，得到各个分类器的值
        k_i = np.argmax(fs.data.cpu().numpy().flatten())#把数组折叠成一维，选最大的那个分类器


        loop_i += 1

    r_tot = (1+overshoot)*r_tot1#总扰动加上一个小扰动

    endtime=time.perf_counter()
    print("扰动后标签：",k_i)
    timedate=endtime-starttime
    #print("扰动为：",r_tot)

#返回总扰动、循环次数、原标签、扰动后的分类器标签、扰动图片
    return r_tot, loop_i, label1, k_i, pert_image1,timedate



'''

#net=models.alexnet(pretrained=True)
net=models.vgg16(pretrained=True)


# Switch to evaluation mode
net.eval()#测试模型

#导入图片
im_orig = Image.open('D:/毕设/DeepFool/data/imageNet/ILSVRC2017_DET_test_new/test/ILSVRC2017_test_00000012.JPEG')#导入图片

#图片的值
mean = [ 0.485, 0.456, 0.406 ]#均值，三分量顺序是RGB，后期要调整，opencv和numpy可以自己算得出来
std = [ 0.229, 0.224, 0.225 ]#方差


#去掉平均值
im = transforms.Compose([
    transforms.Scale(256),#图片尺寸
    transforms.CenterCrop(224),#进行中心裁剪
    transforms.ToTensor(),#转换图片格式，变成张量
    transforms.Normalize(mean = mean,std = std)])(im_orig)#对原图进行标准化操作

num_classes=10#分类数


'''
#if __name__ == '__main__':
    #print('start')
    #deepfool(im,net,num_classes);