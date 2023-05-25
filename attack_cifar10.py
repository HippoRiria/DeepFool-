import argparse
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision
from deepfool import deepfool#导入deepfool算法
from deepfool_to_cifar10_t import deepfool_t

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"#设置选项允许重复加载动态链接库
import torch.optim as optim
from PIL import Image
from models_a import *
from utils import progress_bar


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    # args = parser.parse_args()
    args = parser.parse_args(args=[])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if (device == 'cuda'): print('yes')
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Building model..')
    net = VGG('VGG19')#ok
    #net = torch.load(modelpath)  # 从本地导入神经网络模型
    #net = ResNet18()#ok
    #net = PreActResNet18()
    #net = GoogLeNet()#ok
    #net = DenseNet121()
    #net = ResNeXt29_2x64d()
    #net = MobileNet()
    #net = MobileNetV2()
    #net = DPN92()
    #net = ShuffleNetG2()
    #net = SENet18()
    #net = ShuffleNetV2(1)
    #net = EfficientNetB0()
    #net = RegNetX_200MF()
    #net = SimpleDLA()#ok

    net = net.to(device)


    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    modelpath = "./models/vgg19.pth"
    #modelpath="./models/googlenet.pth"
    #modelpath="./models/resnet18.pth"
    #modelpath="./models/simpledla.pth"
    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['net'])

    #im_orig = Image.open('D:/毕设/DeepFool/data/cifar10/test/cat_1123.jpg')  # 导入图片
    # im_orig=Image.open('D:/毕设/DeepFool/test_im1.jpg')
    # 图片的值
    mean = [0.485, 0.456, 0.406]  # 均值，三分量顺序是RGB，后期要调整，opencv和numpy可以自己算得出来
    std = [0.229, 0.224, 0.225]  # 方差


    # net = {key: net[key].cuda() for key in net}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # start
    num_classes = 10  # 分类数

    total = 100
    corret = 0
    time = 0
    tot_time = 0
    start =0
    for i in range(start, total + start):
        img = Image.open("D:/毕设/DeepFool/data/cifar10/test(num)/{}.jpg".format(str(i)))  # 导入图片，来自cifar10测试集，10000张
        # img = Image.open("D:/毕设/DeepFool/data/cifar10/train(num)/{}.jpg".format(str(i)))  # 导入图片，来自cifar10训练集，50000张

        print('')
        print('正在测试第 ', i, ' 张图片')
        img = transforms.Compose([transforms.Scale(32), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(img)  # 对输入图片进行正则化处理

        r, loop_i, label_orig, label_pert, pert_image, time = deepfool(img, net, num_classes)  # deepfool执行，输入图片与神经网络参数
        #r, loop_i, label_orig, label_pert, pert_image,time = deepfool_t(img, net, num_classes)#deepfool_t执行，输入图片与神经网络参数

        tot_time += time
        if label_orig != label_pert:
            corret += 1

    acc = corret / total
    tot_time = tot_time / total
    print('攻击成功率为：{:2.3f}%'.format(acc))
    print('每张图片进行攻击的平均时间为：{:2.5f}s'.format(tot_time))


