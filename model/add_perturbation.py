import os
import cv2
import time
import numpy as np
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from albumentations import (Resize, RandomCrop, VerticalFlip, HorizontalFlip, Normalize, Compose, Lambda)
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import segmentation_models_pytorch as smp
import visdom
import torchvision.transforms as transforms

os.environ['TORCH_HOME']='G:/data/torch-model'

def get_transforms(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    list_transforms = []
    # if phase == "train":
    #     list_transforms.extend(
    #         [
    #             HorizontalFlip(),
    #             VerticalFlip()
    #         ]
    #     )
    list_transforms.extend(
        [
            # Resize(256, 256, interpolation=Image.BILINEAR),
            # Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
            # # opencv读取图片的颜色通道顺序为BGR,因此直接转换会导致图片颜色变化,这里将通道顺序改为RGB
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


class JetDataset(Dataset):
    def __init__(self, idx, image_path, phase="train"):
        assert phase in ("train", "val", "test")
        self.idx = idx
        self.image_path = image_path
        self.phase = phase
        self.transform = get_transforms(phase)

    def __getitem__(self, index):
        real_idx = self.idx[index]

        image_list = sorted([os.path.join(f'{self.image_path}/image', i) for i in os.listdir(f'{self.image_path}/image')])
        label_list = sorted([os.path.join(f'{self.image_path}/label', i) for i in os.listdir(f'{self.image_path}/label')])

        image_path = image_list[real_idx]
        mask_path = label_list[real_idx]

        image = cv2.imread(image_path)
        image = image[:, :, ::-1].copy()

        mask = cv2.imread(mask_path)

        image = self.transform(image=image)
        label = self.transform(image=mask / 255)
        # augmented = self.transforms(image=image, label_biaozhu=label_biaozhu / 128)

        return image['image'].float(), torch.unsqueeze(label['image'][2, ...], 0).float()

    def __len__(self):
        return len(self.idx)


def add_gaussian_noise_3d(tensor, noise_std=2):
    """
    向三维张量添加高斯噪声。

    参数:
        tensor: 原始的三维张量，大小为(n_samples, n_channels, n_features)。
        noise_std: 高斯噪声的标准差。

    返回:
        带有高斯噪声的三维张量。
    """
    noise = torch.normal(mean=0.0, std=noise_std, size=tensor.size())
    return tensor + noise


def apply_feature_masking_3d(tensor, mask_ratio=0.1):
    """
    对三维张量应用特征掩码。

    参数:
        tensor: 原始的三维张量，大小为(n_samples, n_channels, n_features)。
        mask_ratio: 需要掩码的特征比例。

    返回:
        应用了特征掩码的三维张量。
    """
    # 计算每个样本需要被掩码的特征数量
    n_features = tensor.size(2)
    n_masked_features = int(n_features * mask_ratio)

    # 对每个样本的每个通道应用掩码
    masked_tensor = tensor.clone()
    for i in range(tensor.size(0)):  # 遍历样本
        for j in range(tensor.size(1)):  # 遍历通道
            # 随机选择要掩码的特征索引
            mask_indices = torch.randperm(n_features)[:n_masked_features]
            # 将选中的特征置零
            masked_tensor[i, j, mask_indices] = 0
    return masked_tensor

def add_laplace_noise_3d(tensor, noise_scale=1.0):
    """
    向三维张量添加拉普拉斯噪声。

    参数:
        tensor: 原始的三维张量，大小为(n_samples, n_channels, n_features)。
        noise_scale: 拉普拉斯噪声的尺度参数。

    返回:
        带有拉普拉斯噪声的三维张量。
    """
    laplace_dist = torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale)  # 位置参数设为 0，尺度参数为 noise_scale
    noise = laplace_dist.sample(tensor.size())
    return tensor + noise


def fft_transform_tensor(tensor):
    """
    对三维张量的每个样本和通道应用二维傅里叶变换。

    参数:
        tensor: 输入的四维张量，大小为(n_samples, n_channels, height, width)。

    返回:
        二维傅里叶变换后的四维张量。
    """
    # 初始化一个与输入张量形状相同的复数张量来存储傅里叶变换结果
    transformed = torch.zeros_like(tensor, dtype=torch.complex64)

    # 遍历所有样本和通道
    for sample_idx in range(tensor.size(0)):
        for channel_idx in range(tensor.size(1)):
            # 对每个样本和通道的二维数据应用二维傅里叶变换
            transformed[sample_idx, channel_idx] = torch.fft.fftn(tensor[sample_idx, channel_idx])

    return transformed

def calculate_iou_dice(output, label):

    output_img = (output > 0).astype(np.uint8)
    label_img = (label > 0).astype(np.uint8)

    # 计算交集
    intersection = np.logical_and(output_img, label_img)

    # 计算并集
    union = np.logical_or(output_img, label_img)

    # 计算IoU
    iou = np.sum(intersection) / np.sum(union)

    dice_coefficient = 2 * np.sum(intersection) / (np.sum(output_img) + np.sum(label_img))

    return iou, dice_coefficient

def load_image_to_tensor(image_path):
    """
    从指定路径加载PNG图像并将其转换为张量。

    参数:
        image_path: 图像文件的路径。

    返回:
        一个张量，表示加载的图像。
    """
    # 使用PIL加载图像
    image = Image.open(image_path).convert('RGB')  # 使用.convert('RGB')确保图像为三通道

    # 定义一个转换，将PIL图像转换为张量
    transform = transforms.ToTensor()

    # 应用转换
    tensor = transform(image)

    return tensor

# if __name__=="__main__":
    #
    # path = './jet_stream_dataset'
    # file_list = sorted([os.path.join(f'{path}/image', i) for i in os.listdir(f'{path}/image')])
    # train_temp_idx, test_idx = train_test_split(range(len(file_list)), random_state=67373, test_size=0.1)
    # train_idx, val_idx = train_test_split(range(len(train_temp_idx)), random_state=67373, test_size=0.1)
    # print(len(train_idx))
    # print(len(val_idx))
    # print(len(test_idx))
    #
    # train_set = JetDataset(train_idx, path, 'train')
    # test_set = JetDataset(test_idx, path, 'test')
    #
    # train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    #
    #
    # images, labels = train_set.__getitem__(0)
    #
    #
    # images = images.unsqueeze(0)
    # labels = labels.unsqueeze(0)
    #
    # #从图片去读image && label
    # images = load_image_to_tensor('./pseudo_data/16100208.png')
    #
    # images = images.unsqueeze(0)
    #
    #
    # MODELPATH = './model.pth'
    #
    # model = smp.Unet('resnet18', classes=1, activation=None)
    # state = torch.load(MODELPATH, map_location=lambda storage, loc: storage)
    #
    # model.load_state_dict(state["state_dict"])
    #
    # feature = model.encoder(images)
    #
    #
    # # 对embedding加扰动
    # feature[5] = add_gaussian_noise_3d(feature[5])
    #
    #
    #
    # output1 = model.decoder(*feature)
    # output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    # output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")
    #
    # output = model(images).squeeze(0).squeeze(0).detach().numpy()
    # output = np.where(output > 0.5, 255, 0).astype("uint8")
    #
    # iou1, _ = calculate_iou_dice(output1, output)
    #
    # # 对embedding加扰动
    # feature[5] = add_laplace_noise_3d(feature[5])
    #
    # output1 = model.decoder(*feature)
    # output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    # output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")
    #
    # output = model(images).squeeze(0).squeeze(0).detach().numpy()
    # output = np.where(output > 0.5, 255, 0).astype("uint8")
    #
    # iou2, _ = calculate_iou_dice(output1, output)
    #
    # # 对embedding加扰动
    # feature[5] = apply_feature_masking_3d(feature[5])
    #
    # output1 = model.decoder(*feature)
    # output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    # output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")
    #
    # output = model(images).squeeze(0).squeeze(0).detach().numpy()
    # output = np.where(output > 0.5, 255, 0).astype("uint8")
    #
    # iou3, _ = calculate_iou_dice(output1, output)
    #
    # # image = images[0].numpy()
    # # image = np.transpose(image, (1, 2, 0))
    # # img = Image.fromarray(np.uint8(image))
    # # img.show()
    # #
    # # label = labels[0].numpy()
    # # label = np.transpose(label, (1, 2, 0))
    # # label = Image.fromarray(np.uint8(label.squeeze(2)))
    # # label.show()
    #
    # print((iou1 + iou2 + iou3) / 3.0)


def filter_pseudo_labels(image_path, model_path, threshold) -> bool:

    def add_gaussian_noise_3d(tensor, noise_std=2):
        """
        向三维张量添加高斯噪声。

        参数:
            tensor: 原始的三维张量，大小为(n_samples, n_channels, n_features)。
            noise_std: 高斯噪声的标准差。

        返回:
            带有高斯噪声的三维张量。
        """
        noise = torch.normal(mean=0.0, std=noise_std, size=tensor.size())
        return tensor + noise

    def apply_feature_masking_3d(tensor, mask_ratio=0.2):
        """
        对三维张量应用特征掩码。

        参数:
            tensor: 原始的三维张量，大小为(n_samples, n_channels, n_features)。
            mask_ratio: 需要掩码的特征比例。

        返回:
            应用了特征掩码的三维张量。
        """
        # 计算每个样本需要被掩码的特征数量
        n_features = tensor.size(2)
        n_masked_features = int(n_features * mask_ratio)

        # 对每个样本的每个通道应用掩码
        masked_tensor = tensor.clone()
        for i in range(tensor.size(0)):  # 遍历样本
            for j in range(tensor.size(1)):  # 遍历通道
                # 随机选择要掩码的特征索引
                mask_indices = torch.randperm(n_features)[:n_masked_features]
                # 将选中的特征置零
                masked_tensor[i, j, mask_indices] = 0
        return masked_tensor

    def add_laplace_noise_3d(tensor, noise_scale=2.0):
        """
        向三维张量添加拉普拉斯噪声。

        参数:
            tensor: 原始的三维张量，大小为(n_samples, n_channels, n_features)。
            noise_scale: 拉普拉斯噪声的尺度参数。

        返回:
            带有拉普拉斯噪声的三维张量。
        """
        laplace_dist = torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale)  # 位置参数设为 0，尺度参数为 noise_scale
        noise = laplace_dist.sample(tensor.size())
        return tensor + noise


    MODELPATH = model_path

    model = smp.Unet('resnet18', classes=1, activation=None)
    state = torch.load(MODELPATH, map_location=lambda storage, loc: storage)

    model.load_state_dict(state["state_dict"])

    images = load_image_to_tensor(image_path)
    images = images.unsqueeze(0)
    feature = model.encoder(images)


    # 对embedding加扰动
    feature[5] = add_gaussian_noise_3d(feature[5])


    output1 = model.decoder(*feature)
    output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")

    output = model(images).squeeze(0).squeeze(0).detach().numpy()
    output = np.where(output > 0.5, 255, 0).astype("uint8")

    iou1, _ = calculate_iou_dice(output1, output)

    # 对embedding加扰动
    feature[5] = add_laplace_noise_3d(feature[5])

    output1 = model.decoder(*feature)
    output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")

    output = model(images).squeeze(0).squeeze(0).detach().numpy()
    output = np.where(output > 0.5, 255, 0).astype("uint8")

    iou2, _ = calculate_iou_dice(output1, output)

    # 对embedding加扰动
    feature[5] = apply_feature_masking_3d(feature[5])

    output1 = model.decoder(*feature)
    output1 = model.segmentation_head(output1).squeeze(0).squeeze(0).detach().numpy()
    output1 = np.where(output1 > 0.5, 255, 0).astype("uint8")

    output = model(images).squeeze(0).squeeze(0).detach().numpy()
    output = np.where(output > 0.5, 255, 0).astype("uint8")

    iou3, _ = calculate_iou_dice(output1, output)

    return (iou1 + iou2 + iou3) / 3.0 > threshold, (iou1 + iou2 + iou3) / 3.0