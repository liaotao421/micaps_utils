import os
import re

import cv2
import time
import numpy as np
from segmentation_models_pytorch.losses import DiceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Resize, RandomCrop,VerticalFlip, HorizontalFlip, Normalize, Compose)
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import segmentation_models_pytorch as smp
import visdom
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# vis = visdom.Visdom()


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
        global extracted_number
        real_idx = self.idx[index]

        image_list = sorted([os.path.join(f'{self.image_path}/image', i) for i in os.listdir(f'{self.image_path}/image')])
        label_list = sorted([os.path.join(f'{self.image_path}/label', i) for i in os.listdir(f'{self.image_path}/label')])

        image_path = image_list[real_idx]
        mask_path = label_list[real_idx]

        # 使用正则表达式匹配文件名中的数字部分
        match = re.search(r'(\d+)', image_path)

        if match:
            extracted_number = match.group(1)

        image = cv2.imread(image_path)
        image = image[:, :, ::-1].copy()
        mask = cv2.imread(mask_path)

        image = self.transform(image=image)
        label = self.transform(image=mask / 255)
        # augmented = self.transforms(image=image, label_biaozhu=label_biaozhu / 128)

        return image['image'].float(), torch.unsqueeze(label['image'][2, ...], 0).float(), extracted_number

    def __len__(self):
        return len(self.idx)


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    # 计算iou和dice前进行的二值掩码
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(probability, truth, threshold=0.5, reduction='none'):
    '''
    Calculates dice of positive and negative images seperately
    probability and truth must be torch tensors
    :param probability: 预测的结果 （8,1,256,256） type:tensor
    :param truth: 标签（8,1,256,256）type:tensor
    :param threshold:
    :param reduction:
    :return:
    '''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.contiguous().view(batch_size, -1)
        truth = truth.contiguous().view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        # 要把输出做一个sigmoid函数
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        # Compute the arithmetic mean along the specified axis, ignoring NaNs.
        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.6f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''
    computes iou for one ground truth mask and predicted mask
    :param pred: 预测结果 (1,256,256) type:ndarray
    :param label: 标签 (1,256,256) type:ndarray
    :param classes: 标注的类别 1
    :param ignore_index: 忽略索引？
    :param only_present:？
    :return:
    '''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []

    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np

    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def provider(
        path,
        phase,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=0,
):
    file_list = sorted([os.path.join(f'{path}/image', i) for i in os.listdir(f'{path}/image')])

    train_temp_idx, test_idx = train_test_split(range(len(file_list)), random_state=67373, test_size=0.1)
    train_idx, val_idx = train_test_split(range(len(train_temp_idx)), random_state=67373, test_size=0.1)

    if phase == 'train':
        index = train_idx
    elif phase == 'validation':
        index = val_idx
    else:
        index = test_idx

    dataset = JetDataset(index, path, phase=phase)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        generator=torch.Generator(device='cuda')
    )

    return dataloader

# vis = visdom.Visdom()


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 0
        self.batch_size = {"train": 4, "val": 1}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 1e-3
        self.num_epochs = 80
        # float("inf") 设置一个无限大的变量
        self.best_loss = float("inf")
        self.best_dice = float(0)
        self.phases = ["train", "val"]
        self.device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        # binary cross entropy with Sigmoid
        # self.criterion = DiceLoss(mode='binary')
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=4, verbose=True)
        self.net = self.net.to(self.device)
        # https://blog.csdn.net/AugustMe/article/details/108364073
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                path=path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)

        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | : {start}")
        # https://blog.csdn.net/weixin_44211968/article/details/123774649
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets, _ = batch
            loss, outputs = self.forward(images, targets)
            # 先加再除和先除再加，一样的。朝三暮四和朝四暮三的区别。
            loss = loss / self.accumulation_steps

            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            else:
                pass
                # validation
                # 使用visdom可视化
                # batch_preds = torch.sigmoid(outputs.squeeze(1))
                # batch_preds = batch_preds.detach().cpu().numpy()

                # im = cv2.imread(path[0])
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # im = cv2.resize(im, (256, 256)).transpose((2, 0, 1))
                # vis.images(im, opts=dict(title='target'))
                # vis.images(batch_preds, opts=dict(title='pred'))

            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            # 计算iou和dice
            meter.update(targets, outputs)
        #
        # vis.close()
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()

        return epoch_loss, dice

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            # 训练完一个epoch就开始validation
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss, dice = self.iterate(epoch, "val")

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                self.best_dice = dice
                torch.save(state, "./model.pth")

            print()


'''
ckpt_path = "./model.pth"
device = torch.device("cuda")
model = smp.Unet("resnet50", encoder_weights=None, classes=1, activation=None)
model.to(device)
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])
'''


model = smp.Unet('resnet18', classes=1, activation=None)
path = './jet_stream_dataset'

model_trainer = Trainer(model)
model_trainer.start()