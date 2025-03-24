from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
import numpy as np
from PIL import Image
import random
from itertools import permutations
import h5py
from sklearn.utils import shuffle

_check_pil = lambda x: isinstance(x, Image.Image)
_check_np_img = lambda x: isinstance(x, np.ndarray)


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img, depth = sample["image"], sample["depth"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        return {"image": img, "depth": depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {"image": image, "depth": depth}


def loadZipToMem(mat_path):  # 保持原函数名
    print("Loading .mat file...", end="")
    with h5py.File(mat_path, 'r') as f:
        images = np.array(f['images']).transpose(0, 3, 2, 1)  # [N, H, W, C]
        depths = np.array(f['depths']).transpose(0, 2, 1)  # [N, H, W]

    images = images.astype(np.float32)
    # depths = depths.astype(np.float32)
    depths = (depths * 1000).clip(0, 10000).astype(np.float32)

    nyu2_train = [(i, i) for i in range(images.shape[0])]
    nyu2_train = shuffle(nyu2_train, random_state=0)
    print(f"Loaded ({len(nyu2_train)} samples).")
    return {'images': images, 'depths': depths}, nyu2_train


class depthDatasetMemory(Dataset):  # 保持原类名
    def __init__(self, data, nyu2_train, transform=None):
        self.data = data
        self.nyu_dataset = nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        # 修改点：直接使用原始 float32 数据，避免恢复 uint8
        image_data = self.data['images'][int(sample[0])]
        image = Image.fromarray(image_data.astype(np.uint8))  # 从 float32 转换

        # 深度图处理保持不变
        depth_data = self.data['depths'][int(sample[1])]
        depth = Image.fromarray(depth_data.astype(np.float32), mode='F')
        depth = depth.resize((320, 240))

        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)

        depth = self.depth_to_tensor(depth)


        depth = torch.clamp(depth, 10, 10000)
        return {"image": image, "depth": depth}

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            return torch.from_numpy(pic.transpose((2, 0, 1))).float() / 255.0  # 唯一一次归一化

        # 处理单通道浮点型深度图
        if pic.mode == 'F':
            arr = np.array(pic).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(0)  # 增加通道维度 [1, H, W]

        # 保持原始RGB处理逻辑
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.mode))
        return img.permute(2, 0, 1).float().div(255)

    def depth_to_tensor(self, depth):
        """单独处理深度图转换"""
        if depth.mode == 'F':
            arr = np.array(depth).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        return self.to_tensor(depth)


def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor(is_test=False)
    ])


def getTrainingTestingData(path, batch_size):
    data, nyu2_train = loadZipToMem(path)
    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())
    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size,
                                                                                  shuffle=False)


def load_testloader(path, batch_size=1):
    data, nyu2_train = loadZipToMem(path)
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())
    return DataLoader(transformed_testing, batch_size, shuffle=False)