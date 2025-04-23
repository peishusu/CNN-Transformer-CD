# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import LoadDatasetFromFolder, DA_DatasetFromFolder, calMetric_iou
import numpy as np
import random
from model.network import CDNet
from train_options import parser
import itertools
from loss.losses import cross_entropy

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seeds
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)

if __name__ == '__main__':
    mloss = 0

    # load data
    # train_set = DA_DatasetFromFolder(args.hr1_train, args.hr2_train, args.lab_train, crop=False)
    train_set = DA_DatasetFromFolder(Image_dir1=f'./datasets/CLCD/train/time1/', Image_dir2=f'./datasets/CLCD/train/time2/', Label_dir=f'./datasets/CLCD/train/label/', crop=False)

    # val_set = LoadDatasetFromFolder(args, args.hr1_val, args.hr2_val, args.lab_val)
    val_set = LoadDatasetFromFolder(hr1_path=f'./datasets/CLCD/val/time1/', hr2_path=f'./datasets/CLCD/val/time2/', lab_path=f'./datasets/CLCD/val/label/')

    # train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    # 创建了一个 PyTorch DataLoader 对象，用于高效加载训练数据集 (train_set)，支持多线程数据加载、批处理和数据打乱
    # shuffle：是否在每个 epoch 开始时打乱数据顺序。
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=1, shuffle=True)

    # val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)
    #
    val_loader = DataLoader(dataset=val_set, num_workers=8, batch_size=1, shuffle=True)

    # define model
    # CDNet = CDNet(img_size = args.img_size).to(device, dtype=torch.float)
    CDNet = CDNet(img_size = 512).to(device, dtype=torch.float)

    # 这段代码是用来在多GPU环境中使用 DataParallel 进行并行计算的。具体来说，它会检查可用的 GPU 数量，并在多个 GPU 上并行地运行模型
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))

    # set optimization
    lr = 2e-4 # 设置学习率
    # 创建了一个 Adam 优化器，用于优化 CDNet 模型的所有参数。学习率设置为 2e-4，使用常见的 beta 参数 (0.9, 0.999)
    optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr= lr, betas=(0.9, 0.999))
    # optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr= lr, betas=(0.9, 0.999))

    # 自定义的损失函数，
    CDcriterionCD = cross_entropy().to(device, dtype=torch.float)

    # training
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        # 使用了 tqdm 来包装 train_loader，从而实现一个进度条的显示
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'SR_loss':0, 'CD_loss':0, 'loss': 0 }

        # 训练流程
        CDNet.train() # 设置为训练模式
        # 这行代码会在每次迭代时从 train_bar 中获取一个批次的数据。并不是仅仅一张图，图片的数量是由batchsize所决定的
        for hr_img1, hr_img2, label in train_bar:

            # 测试
            print(f"图片1的形状: {hr_img1.shape[0]} {hr_img1.shape[1]} {hr_img1.shape[2]} {hr_img1.shape[3]}") # Tensor的格式 [1,3,512,512]
            print(f"图片2的形状: {hr_img2.shape[0]} {hr_img2.shape[1]} {hr_img2.shape[2]} {hr_img2.shape[3]}") # Tensor的格式 [1,3,512,512]
            print(f"标签的形状: {label.shape[0]} {label.shape[1]} {label.shape[2]} {label.shape[3]}") # Tensor的格式 [1,2,512,512]

            # running_results['batch_sizes'] += args.batchsize
            running_results['batch_sizes'] += 1


            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float) #预期标签
            label = torch.argmax(label, 1).unsqueeze(1).float()

            result1, result2, result3= CDNet(hr_img1, hr_img2)  # 真实标签
            print(
                f"result1: {result1.shape[0]} {result1.shape[1]} {result1.shape[2]} {result1.shape[3]}")  # Tensor的格式 [1,2,512,512]
            print(
                f"result2: {result2.shape[0]} {result2.shape[1]} {result2.shape[2]} {result2.shape[3]}")  # Tensor的格式 [1,2,512,512]
            print(
                f"result3: {result3.shape[0]} {result3.shape[1]} {result3.shape[2]} {result3.shape[3]}")  # Tensor的格式 [1,2,512,512]


            CD_loss = CDcriterionCD(result1, label) +CDcriterionCD(result2, label)+CDcriterionCD(result3, label)

            # 每次反向传播之前，需要清除模型的梯度。PyTorch 在反向传播时会累加梯度，所以每次优化步骤开始之前，必须将梯度清零。否则梯度会在每次调用 .backward() 时累加，导致错误的梯度更新。
            CDNet.zero_grad()
            # 这一步执行反向传播，它会计算损失函数相对于模型参数的梯度。具体来说，CD_loss 是损失函数的值，.backward() 会根据损失值反向传播，计算每个模型参数的梯度。
            CD_loss.backward()
            # 这一步根据计算出的梯度来更新模型参数。优化器（如 Adam、SGD 等）会根据梯度更新参数，以最小化损失函数。
            optimizer.step()

            # running_results['CD_loss'] += CD_loss.item() * args.batchsize
            # 用来累计或存储训练过程中每个 batch 的损失值
            running_results['CD_loss'] += CD_loss.item() * 1


            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, num_epochs,
                    # 这部分计算得到的是当前批次的平均损失
                    running_results['CD_loss'] / running_results['batch_sizes'],))

        # eval
        CDNet.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0,0
            valing_results = {'batch_sizes': 0, 'IoU': 0}

            for hr_img1, hr_img2, label in val_bar:
                # valing_results['batch_sizes'] += args.val_batchsize
                valing_results['batch_sizes'] += 1


                hr_img1 = hr_img1.to(device, dtype=torch.float)
                hr_img2 = hr_img2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                label = torch.argmax(label, 1).unsqueeze(1).float()

                cd_map,_,_ = CDNet(hr_img1, hr_img2)

                CD_loss = CDcriterionCD(cd_map, label)

                cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()

                gt_value = (label > 0).float()
                prob = (cd_map > 0).float()
                prob = prob.cpu().detach().numpy()

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)
                intr, unn = calMetric_iou(gt_value, result)
                inter = inter + intr
                unin = unin + unn

                valing_results['IoU'] = (inter * 1.0 / unin)

                val_bar.set_description(
                    desc='IoU: %.4f' % (  valing_results['IoU'],))

        # save model parameters
        val_loss = valing_results['IoU']
        model_dir = './train_result/'
        os.makedirs(model_dir, exist_ok=True)  # 自动创建目录（如果不存在）
        if val_loss > mloss or epoch==1:
            mloss = val_loss
            torch.save(CDNet.state_dict(),  model_dir+'netCD_epoch_%d.pth' % (epoch))
