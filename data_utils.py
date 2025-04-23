#coding=utf-8
from os.path import join
import torch
from PIL import Image, ImageEnhance
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os
import imageio


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp


def getDataList(img_path):
    dataline = open(img_path, 'r').readlines()
    datalist =[]
    for line in dataline:
        temp = line.strip('\n')
        datalist.append(temp)
    return datalist


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)  # 获取输入形状（如 [N, 1, H, W]）
    shape[1] = num_classes   # 将第 1 维（原 1）改为 num_classes
    shape = tuple(shape) # 转换为元组（如 (N, 2, H, W)）
    result = torch.zeros(shape) # 创建全 0 的张量，形状 (N, num_classes, H, W)
    result = result.scatter_(1, input.cpu(), 1)  # 如果 input[0,0,0,0] = 1，则 result[0,1,0,0] = 1（在类别 1 的位置填 1）。
    return result


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



class LoadDatasetFromFolder(Dataset):
    # def __init__(self, args, hr1_path, hr2_path, lab_path):
    def __init__(self, hr1_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder, self).__init__()
        # 获取图片列表
        suffix = ['.jpg', '.png', '.tif']  # 支持.jpg/.png/.tif格式的图片
        datalist = [name for name in os.listdir(hr1_path) for item in suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        # lr2_img = self.transform(Image.open(self.lr2_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))

        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return hr1_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, args, Time1_dir, Time2_dir, Label_dir):
        super(TestDatasetFromFolder, self).__init__()

        datalist = [name for name in os.listdir(Time1_dir) for item in args.suffix if
                    os.path.splitext(name)[1] == item]

        self.image1_filenames = [join(Time1_dir, x) for x in datalist if is_image_file(x)]
        self.image2_filenames = [join(Time2_dir, x) for x in datalist if is_image_file(x)]
        self.image3_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert('RGB'))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert('RGB'))

        label = self.label_transform(Image.open(self.image3_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        image_name =  self.image1_filenames[index].split('/', -1)
        image_name = image_name[len(image_name)-1]

        return image1, image2, label, image_name

    def __len__(self):
        return len(self.image1_filenames)

'''
    作用：图像增强
'''
class trainImageAug(object):
    def __init__(self, crop = True, augment = True, angle = 30):
        self.crop =crop # 制是否进行随机裁剪（默认裁剪为256x256区域）。
        self.augment = augment # 控制是否启用数据增强（翻转+旋转
        self.angle = angle # 随机旋转的角度范围（如angle=30表示旋转角度在[-30°, 30°]之间）。
    # 一个类的实例被像函数一样调用时（即使用 对象() 的语法），Python 会自动触发该实例的 __call__ 方法。
    def __call__(self, image1, image2, mask):
        # 进行图片的裁剪
        if self.crop:
            w = np.random.randint(0,256) # 随机起始x坐标
            h = np.random.randint(0,256)   # 随机起始y坐标
            box = (w, h, w+256, h+256) # 定义裁剪区域 (left, upper, right, lower)
            image1 = image1.crop(box)  # 裁剪图像1
            image2 = image2.crop(box) # 裁剪图像2
            mask = mask.crop(box) # 同步裁剪掩码
        # 进行图片的增强
        if self.augment:
            # 生成随机概率
            prop = np.random.uniform(0, 1)
            # 水平反转
            if prop < 0.15:
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # 垂直翻转
            elif prop < 0.3:
                image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
                image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            # 随机旋转
            elif prop < 0.5:
                image1 = image1.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                image2 = image2.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                mask = mask.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))

        return image1, image2, mask

def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [
                            transforms.ToTensor(), # 将输入的图像（PIL Image 或 NumPy 数组，格式为 H×W×C）转换为 PyTorch 张量（格式为 C×H×W），并自动将像素值从 [0, 255] 缩放到 [0, 1]。
                           ]
    if normalize:
        transform_list += [
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))] #对张量进行标准化，这里使用的均值 (0.5, 0.5, 0.5) 和标准差 (0.5, 0.5, 0.5) 会将像素值从 [0, 1] 线性映射到 [-1, 1]（公式：(x - mean) / std）。
    # 返回一个 transforms.Compose 对象，按顺序组合所有选择的变换。
    return transforms.Compose(transform_list)


class DA_DatasetFromFolder(Dataset):
    def __init__(self, Image_dir1, Image_dir2, Label_dir, crop=True, augment = True, angle = 30):
        super(DA_DatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        # 初始化图片增强器
        self.data_augment = trainImageAug(crop=crop, augment = augment, angle=angle)
        #
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    # 当通过索引访问数据时自动调用（如dataset[i]或DataLoader迭代时）
    def __getitem__(self, index):
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index])
        # 回去调用 trainImageAug类的__call__方法，得到增强后的图片
        image1, image2, label = self.data_augment(image1, image2, label)
        # 两个 标准化后的张量 image1 和 image2，形状为 (C, H, W)，值范围可能是 [0, 1] 或 [-1, 1]（取决于是否包含 Normalize）。
        image1, image2 = self.img_transform(image1), self.img_transform(image2)
        #张量形状为 (1, H, W)，值范围保持原始像素值（如 0, 1）
        label = self.lab_transform(label)
        # label.unsqueeze(0).long()：将 (1, H, W) 的标签张量扩展为 (1, 1, H, W) 并转换为 torch.long 类型。

        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        return image1, image2, label

    # 调用时机：当调用len(dataset)或DataLoader确定批次数量时
    def __len__(self):
        return len(self.image_filenames1)