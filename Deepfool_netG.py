import copy
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models#从pytorch中导入预定好的神经网络模型
from PIL import Image
from torch.autograd import Variable
import os
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary

import time


def deepfool(image, net, num_classes, overshoot=0.02, max_iter=500):#图片、网络参数、分类数、最后添加的小扰动、最大迭代次数

    is_cuda = torch.cuda.is_available()#引入GPU计算

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

    I = I[0:num_classes]#分类以及分类器的编号，这里返回了分类器原标签的编号
    label = I[0]#取最大的编号，得到标签
    print("扰动前标签：",label)

    input_shape = image.cpu().numpy().shape#把图片变成参数
    pert_image = copy.deepcopy(image)#把图片设为原图片
    w = np.zeros(input_shape)#得到分类器的权重值，初始化
    r_tot = np.zeros(input_shape)#初始化扰动，将扰动设为输入的图片参数

    loop_i = 0#初始化循环次数

    x = Variable(pert_image[None, :], requires_grad=True)#将图片参数转换，因为神经网络只能输入variable
    fs = net.forward(x)#前向传播得到函数，输出为一个Mat数组，每个Mat为t维度向量组（n行t列矩阵）。t由模型决定,这里是输出各个分类器参数

    fs_list = [fs[0,I[k]] for k in range(num_classes)]#初始化fs[0,I[k]]，存分类器
    k_i = label#标签
#deepfool的扰动计算算法
    while k_i == label and loop_i < max_iter:#进入循环，开始叠加扰动。当分类器标签不变时以及迭代未超过最大迭代次数时

        pert = np.inf#np.inf表示正无穷，初始化扰动
        fs[0, I[0]].backward(retain_graph=True)#沿梯度向后传播,自动计算原分类器梯度
        grad_orig = x.grad.data.cpu().numpy().copy()#原梯度

        for k in range(1, num_classes):#多分类常规情况
            x.grad.zero_()#将梯度参数归零

            fs[0, I[k]].backward(retain_graph=True)#继续向后传播，沿目标分类器k向后传播
            cur_grad = x.grad.data.cpu().numpy().copy()#得到在此分类器下的梯度参数

            w_k = cur_grad - grad_orig#线性化分类器，通过分类器的梯度以及原梯度线性化分类器
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()#得到线性化后的分类器

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())#求该分类器的扰动，绝对值

            if pert_k < pert:#选各个分类器中最小的扰动以及参数
                pert = pert_k
                w = w_k

        r_i =(pert+1e-4) * w / np.linalg.norm(w)#叠加第i次扰动，1e-4=0.0001，这个数可以保持数组的稳定性
        r_tot = np.float32(r_tot + r_i)#得到总梯度

        if is_cuda:#根据用GPU还是CPU算出来的数据叠加扰动至图片
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
        x = Variable(pert_image, requires_grad=True)#x存入被扰动的图片
        fs = net.forward(x)#向前传播，得到各个分类器的值
        k_i = np.argmax(fs.data.cpu().numpy().flatten())#把数组折叠成一维，选最大的那个分类器

        loop_i += 1

    r_tot = (1+overshoot)*r_tot#总扰动加上一个小扰动

    endtime=time.perf_counter()
    print("扰动后标签：",k_i)
    timedate=endtime-starttime
    #print("扰动为：",r_tot)

#返回总扰动、循环次数、原标签、扰动后的分类器标签、扰动图片
    return r_tot, loop_i, label, k_i, pert_image,timedate



class VGGNet(nn.Module):
    def __init__(self, num_classes=10):  # num_classes
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)  # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 512),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



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



if __name__ == '__main__':

    '''定义超参数'''
    file_path = './Cifar-10'
    batch_size = 256  # 批的大小
    num_epoches = 10  # 遍历训练集的次数

    '''下载训练集 CIFAR-10 10分类训练集'''  # 也可以自己下载 cifar-10-python.tar.gz 到指定目录下
    train_dataset = datasets.CIFAR10(file_path, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.CIFAR10(file_path, train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # 测试集不需要打乱

    '''创建model实例对象，并检测是否支持使用GPU'''
    model = VGGNet()  # 先实例化模型
    summary(model, input_size=(3, 32, 32), device='cpu')  # 打印模型结构

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  # 仅让id=0和1的GPU可被使用(也可以不写)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:  # 有多块GPU供使用
        print("GPU共有 %d 块" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0, 1])  # device_ids不指定的话，默认启用所有(指定)可用的GPU

    model = model.to(device)

    '''定义 loss 和 optimizer '''
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    '''   训练网络
    - 获取损失：loss = loss_func(out,batch_y)
    - 清空上一步残余更新参数：opt.zero_grad()
    - 误差反向传播：loss.backward()
    - 将参数更新值施加到net的parmeters上：opt.step()
    '''
    for epoch in range(num_epoches):
        model.train()
        print('\n', '*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)  # .format为输出格式，formet括号里的即为左边花括号的输出
        running_loss = 0.0
        num_correct = 0.0
        for i, data in enumerate(train_loader, 0):
            img, label = data
            img, label = img.to(device), label.to(device)  # 推荐使用Tensor, 替代 Variable

            out = model(img)  # 向前传播

            # 向后传播
            loss = loss_func(out, label)  # 计算loss
            optimizer.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # loss 求导, 误差反向传播，计算参数更新值
            optimizer.step()  # 更新参数：将参数更新值施加到net的parmeters上

            # 计算loss 和 acc
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)  # 预测最大值所在的位置标签
            num_correct += (pred == label).sum().item()  # 统计正确的个数
            # print('==> epoch={}, running_loss={}, num_correct={}'.format(i+1, running_loss, num_correct))

        print(
            'Train==> Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(train_dataset)), num_correct / (len(train_dataset))))

        # 测试 评估 模型
        model.eval()  # 模型进入测试阶段，参数不再更改
        eval_loss = 0
        num_correct = 0
        for data in test_loader:  # 测试模型
            img, label = data
            img, label = img.to(device).detach(), label.to(device).detach()  # 测试时不需要梯度

            out = model(img)
            loss = loss_func(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct += (pred == label).sum().item()
        print('Test==>  Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), num_correct / (len(test_dataset))))

    # 保存模型
    print('save model_________________________')
    torch.save(model.state_dict(), './VGGNet16_cifar10.pth')