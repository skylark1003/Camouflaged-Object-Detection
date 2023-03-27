import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from PyQt5 import QtGui
from PyQt5.QtGui import QImage

# 这是一个用于数据增强的 Python 代码，
# 主要是通过对图像进行旋转、翻转、裁剪、噪声、对比度、颜色增强等操作，来扩充数据集，从而增强深度学习模型的泛化能力。

# cv_random_flip: 随机水平翻转图像；
# several data augumentation strategies
def cv_random_flip(img, fix, gt):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        fix = fix.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, fix, gt

# randomCrop: 随机裁剪图像；
def randomCrop(image, fix, gt):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), fix.crop(random_region), gt.crop(random_region)

# randomRotation: 随机旋转图像；
def randomRotation(image, fix, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        fix = fix.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return image, fix, gt

# colorEnhance: 随机调整图像的亮度、对比度、色彩和清晰度；
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

# randomGaussian: 随机添加高斯噪声；
def randomGaussian(image, mean=0, sigma=0.15):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomGaussian1(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

# randomPeper: 随机添加椒盐噪声；
def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

# 训练数据集类SalObjDataset
class SalObjDataset(data.Dataset):
    # __init__函数初始化了数据集的路径和训练大小，
    # 读取图像、GT和fixation map数据，
    # 并根据训练大小进行图像、GT和fixation map的resize和转换（包括颜色增强、随机翻转、随机裁剪、随机旋转、随机噪声等）。
    def __init__(self, image_root, gt_root, fix_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.fixs = [fix_root + f for f in os.listdir(fix_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.fixs = sorted(self.fixs)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.fix_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    # getitem 函数返回增强后的图像、GT 和 Fixation Map。
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        fix = self.binary_loader(self.fixs[index])
        image, fix, gt = cv_random_flip(image, fix, gt)
        image, fix, gt = randomCrop(image, fix, gt)
        image, fix, gt = randomRotation(image, fix, gt)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        fix = self.gt_transform(fix)
        return image, gt, fix
    # filter_files函数对读取的数据进行过滤，只保留尺寸一致的图像、GT和fixation map数据。
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        assert len(self.images) == len(self.fixs)
        images = []
        gts = []
        fixs = []
        for img_path, gt_path, fix_path in zip(self.images, self.gts, self.fixs):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            fix = Image.open(fix_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                fixs.append(fix_path)
        self.images = images
        self.gts = gts
        self.fixs = fixs
    # rgb_loader和binary_loader函数分别用于读取RGB图像和二值图像。
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')
    # resize函数用于对图像、GT和fixation map进行resize操作。
    def resize(self, img, gt, fix):
        assert img.size == gt.size
        assert img.size == fix.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), fix.resize((w, h), Image.NEAREST)
        else:
            return img, gt, fix

    def __len__(self):
        return self.size

# get_loader函数用于返回一个DataLoader对象，
# 该对象将训练数据集SalObjDataset转化为可供模型训练的数据集对象。
def get_loader(image_root, gt_root, fix_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, fix_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# test_dataset类用于读取测试数据集。
class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0
    # load_data函数用于加载一张测试图像，并返回其处理后的图像、图像高度、图像宽度和图像名。
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, HH, WW, name
    # rgb_loader和binary_loader函数同样用于读取RGB图像和二值图像。
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
