import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import time
import matplotlib.pyplot as plt
#from torch.autograd.gradcheck import zero_gradients#梯度归零函数，pytorch1.9.0后被删除
#梯度归零换为x.grad.zero_()   x为nn模型

def deepfool_t(image, net, num_classes, overshoot=0.02, max_iter=500):#图片、网络参数、分类数、最后添加的小扰动、最大迭代次数

    is_cuda = torch.cuda.is_available()#引入GPU计算方式
    #is_cuda=0;
    if is_cuda:
        print("使用 GPU 计算")
        image = image.cuda()
        net = net.cuda()
    else:
        print("使用 CPU 计算")#选择用GPU跑还是CPU跑

#初始化
    starttime=time.perf_counter()

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()#先得到原图的参数，然后向前传播

    I = (np.array(f_image)).flatten().argsort()[::-1]#argsort()返回数组从小到大的索引值，flatten()返回一个折叠成一维的数组

    I = I[0:num_classes]#分类以及分类器的编号
    label = I[0]#得到标签
    print("扰动前标签：",label)

    #for i in range(num_classes):
     #   print('number is:',I[i])


    input_shape = image.cpu().numpy().shape#把图片变成参数
    pert_image = copy.deepcopy(image)#把图片设为原图片
    w = np.zeros(input_shape)#得到分类器的权重值，初始化
    r_tot = np.zeros(input_shape)#初始化扰动，将扰动设为输入的图片参数

    loop_i = 0#初始化循环次数

    x = Variable(pert_image[None, :], requires_grad=True)#将图片参数转换，因为神经网络只能输入variable.[None, :]，保证数据不改变的情况下,追加一个新维度。
    fs = net.forward(x)#前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定
    k_i = label#标签

    t=1
    print('所选择的分类为：',t)

    print('索引分类:',I[t])
#deepfool的扰动计算算法
    while k_i != t and loop_i < max_iter:#进入循环，开始叠加扰动。当分类器标签不变时以及迭代未超过最大迭代次数时

        pert = np.inf#np.inf表示正无穷，初始化扰动
        fs[0, I[0]].backward(retain_graph=True)#沿梯度向后传播
        grad_orig = x.grad.data.cpu().numpy().copy()#原梯度

        x.grad.zero_()#将梯度参数归零

        fs[0, t].backward(retain_graph=True)#继续向后传播，沿分类器k向后传播
        cur_grad = x.grad.data.cpu().numpy().copy()#得到在此分类器下的梯度参数

        w = cur_grad - grad_orig#线性化分类器，通过分类器的梯度以及原梯度线性化分类器
        f = (fs[0, t] - fs[0, I[0]]).data.cpu().numpy()#得到线性化后的分类器

        c=w+(1e-4)
        pert = abs(f)/np.linalg.norm(w.flatten())#求该分类器的扰动长度，绝对值


        r_i =  (pert+1e-4) * w / np.linalg.norm(w)#叠加第i次扰动，1e-4=0.0001，这个数可以保持数组的稳定性.np.linalg.norm() l2 norm
        r_tot = np.float32(r_tot + r_i)#得到总扰动

        if is_cuda:#根据用GPU还是CPU算出来的数据叠加扰动至图片
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)#x存入被扰动的图片
        fs = net.forward(x)#向前传播（也就是梯度反向）
        k_i = np.argmax(fs.data.cpu().numpy().flatten())#最大的那个分类器

        loop_i += 1

    r_tot = (1+overshoot)*r_tot#总扰动加上一个小扰动

    endtime=time.perf_counter()
    print("扰动后标签：",k_i)
    timedate=endtime-starttime

    #print(r_tot)
#返回总扰动、循环次数、原标签、扰动后的分类器标签、扰动图片
    return r_tot, loop_i, label, k_i, pert_image,timedate
