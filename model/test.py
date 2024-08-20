import os
import re
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import sys
from torch.utils.data import DataLoader, Dataset
from albumentations import (Resize, RandomCrop,VerticalFlip, HorizontalFlip, Normalize, Compose, Crop, PadIfNeeded, RandomBrightness, Rotate)
from albumentations.pytorch import ToTensorV2
import cv2
from torch.nn import functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp

os.environ['TORCH_HOME']='G:/data/torch-model'


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




# %%
class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 0
        self.batch_size = {"test": 8}
        self.accumulation_steps = 32 // self.batch_size['test']
        self.phases = ["test"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model

        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                path=IMAGEPATH,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}

    def forward(self, images):
        images = images.to(self.device)
        outputs = self.net(images)

        return outputs

    def iterate(self, phase):
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: 0 | phase: {phase} | : {start}")
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        for batch in tqdm(dataloader):
            images, pathes, dates = batch
            with torch.no_grad():
                outputs = self.forward(images)

            batch_preds = torch.sigmoid(outputs)
            # 预测结果以图片的形式存在输入图片的相同路径下，后面带有 label_biaozhu 后缀
            # 预测结果的分辨率统一为 256*256，若需恢复原分辨率需要调用albumentations 中的 Resize
            for i in range(batch_preds.shape[0]):
                numpy_output = batch_preds[i].squeeze(0).detach().cpu().numpy()
                r = np.where(numpy_output > 0.5, 255, 0).astype("uint8")
                cv2.imwrite('./jet_stream_dataset/resimg/' + f"{dates[i]}_result.png", r)

    def start(self):
        self.iterate("test")

MODELPATH = "./model.pth"  # 模型路径
IMAGEPATH = "./jet_stream_dataset"  # 图片路径

if os.path.exists(MODELPATH):

    model = smp.Unet('resnet18', classes=1, activation=None)
    state = torch.load(MODELPATH, map_location=lambda storage, loc: storage)

    model.load_state_dict(state["state_dict"])
else:
    model = smp.Unet('resnet50', classes=1, activation=None)

device = torch.device("cuda")
model.to(device)
model_trainer = Trainer(model)
model_trainer.start()