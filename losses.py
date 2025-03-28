import torch
import math
import torch.nn.functional as F
import numpy as np

""" Loss file implementation refered from 
https://github.com/ialhashim/DenseDepth/blob/master/PyTorch/loss.py
"""


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

    return window


def ssim(
    img1, img2, val_range, window_size=11, window=None, size_average=True, full=False
):

    L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret


# def image_gradients(img, device):
#
#     """works like tf one"""
#     if len(img.shape) != 4:
#         raise ValueError("Shape mismatch. Needs to be 4 dim tensor")
#
#     img_shape = img.shape
#     batch_size, channels, height, width = img.shape
#
#     dy = img[:, :, 1:, :] - img[:, :, :-1, :]
#     dx = img[:, :, :, 1:] - img[:, :, :, :-1]
#
#     shape = np.stack([batch_size, channels, 1, width])
#     dy = torch.cat(
#         [
#             dy,
#             torch.zeros(
#                 [batch_size, channels, 1, width], device=device, dtype=img.dtype
#             ),
#         ],
#         dim=2,
#     )
#     dy = dy.view(img_shape)
#
#     shape = np.stack([batch_size, channels, height, 1])
#     dx = torch.cat(
#         [
#             dx,
#             torch.zeros(
#                 [batch_size, channels, height, 1], device=device, dtype=img.dtype
#             ),
#         ],
#         dim=3,
#     )
#     dx = dx.view(img_shape)
#
#     return dy, dx
def image_gradients(img, device):
    """
    计算深度图的 x 和 y 方向梯度（边缘镜像填充优化）
    Args:
        img: 输入张量 [B, C, H, W]
    Returns:
        dy: 垂直梯度 (B, C, H, W)
        dx: 水平梯度 (B, C, H, W)
    """
    if len(img.shape) != 4:
        raise ValueError("Input tensor must be 4D: [B, C, H, W]")

    # 反射填充右边缘和下边缘各1像素
    padded = F.pad(img, (0, 1, 0, 1), mode='reflect')  # (left, right, top, bottom)

    # 计算水平梯度 (dx = right - left)
    dx = padded[:, :, :, 1:] - padded[:, :, :, :-1]  # [B, C, H+1, W]
    dx = dx[:, :, :-1, :]  # 裁剪高度回 H

    # 计算垂直梯度 (dy = bottom - top)
    dy = padded[:, :, 1:, :] - padded[:, :, :-1, :]  # [B, C, H, W+1]
    dy = dy[:, :, :, :-1]  # 裁剪宽度回 W

    return dy, dx


# Now we define the actual depth loss function
# def depth_loss(y_true, y_pred, theta=0.1, device="cuda", maxDepth=1000.0 / 10.0):
#
#     # Edges
#     y_true_mm = y_true * 10000.0  # [0.0, 10000.0]
#     y_pred_mm = y_pred * 10000.0  # [0.0, 10000.0]  ckpt1-8
#
#     # 计算梯度
#     dy_true, dx_true = image_gradients(y_true_mm, device)
#     dy_pred, dx_pred = image_gradients(y_pred_mm, device)
#
#     # 计算 L1 损失
#     l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
#     return l_edges
def depth_loss(y_true, y_pred,  device="cuda"):

    # Edges
    dy_true, dx_true = image_gradients(y_true, device)
    dy_pred, dx_pred = image_gradients(y_pred, device)
    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))

    return l_edges