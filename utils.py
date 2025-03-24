import numpy as np
import matplotlib
import matplotlib.cm as cm
from PIL import Image

import torch


def DepthNorm(depth, max_depth=10000):
    return depth / max_depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def colorize(value, vmin=10, vmax=10000, cmap="binary"):
#
#     value = value.cpu().numpy()[0, :, :]
#
#     # normalize
#     vmin = value.min() if vmin is None else vmin
#     vmax = value.max() if vmax is None else vmax
#     if vmin != vmax:
#         value = (value - vmin) / (vmax - vmin)
#     else:
#         value = value * 0
#
#     cmapper = cm.get_cmap(cmap)
#     value = cmapper(value, bytes=True)
#
#     img = value[:, :, :3]
#
#     return img.transpose((2, 0, 1))

import numpy as np
from matplotlib import cm

def colorize(value, vmin=None, vmax=None, cmap="viridis"):
    """
    鲁棒的深度图彩色化函数
    Args:
        value: 输入张量 (支持 [B, C, H, W], [C, H, W], [H, W])
        vmin: 归一化最小值（默认数据最小值）
        vmax: 归一化最大值（默认数据最大值）
        cmap: 颜色映射名称（如 "viridis", "jet", "binary"）
    Returns:
        img: 彩色化后的图像 [3, H, W]
    """
    # 压缩批次和通道维度
    value = value.squeeze().cpu().numpy()  # 处理任意维度输入

    # 动态计算归一化范围
    vmin = np.min(value) if vmin is None else vmin
    vmax = np.max(value) if vmax is None else vmax

    # 归一化并限制范围
    if vmin != vmax:
        value = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    else:
        value = np.zeros_like(value)

    # 应用颜色映射并确保三维
    cmapper = cm.get_cmap(cmap)
    colored = cmapper(value, bytes=True)  # [H, W, 4]

    # 提取 RGB 并处理维度
    img = colored[..., :3]  # [H, W, 3]
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)  # [3, H, W]
    else:
        # 处理意外情况：强制转为三维（例如灰度图）
        img = np.expand_dims(img, axis=-1).repeat(3, axis=-1)  # [H, W, 3]
        img = img.transpose(2, 0, 1)

    return img


def load_from_checkpoint(ckpt, model, optimizer, epochs, loss_meter=None):

    checkpoint = torch.load(ckpt)
    ckpt_epoch = epochs - (checkpoint["epoch"] + 1)
    if ckpt_epoch <= 0:
        raise ValueError(
            "Epochs provided: {}, epochs completed in ckpt: {}".format(
                epochs, checkpoint["epoch"] + 1
            )
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

    return model, optimizer, ckpt_epoch


def init_or_load_model(
    depthmodel,
    enc_pretrain,
    epochs,
    lr,
    ckpt=None,
    device=torch.device("cuda:0"),
    loss_meter=None,
):

    if ckpt is not None:
        checkpoint = torch.load(ckpt)

    model = depthmodel(encoder_pretrained=enc_pretrain)

    if ckpt is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ckpt is not None:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    start_epoch = 0
    if ckpt is not None:
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch <= 0:
            raise ValueError(
                "Epochs provided: {}, epochs completed in ckpt: {}".format(
                    epochs, checkpoint["epoch"] + 1
                )
            )

    return model, optimizer, start_epoch


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(
            np.asarray(Image.open(file).resize((640, 480)), dtype=float) / 255, 0, 1
        ).transpose(2, 0, 1)

        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)
