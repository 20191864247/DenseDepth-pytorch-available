import time
import argparse as arg
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.utils as vision_utils
from torch.utils.tensorboard import SummaryWriter

from model import DenseDepth
from losses import ssim as ssim_criterion
from losses import depth_loss as gradient_criterion
from data import getTrainingTestingData
from utils import (
    AverageMeter,
    DepthNorm,
    colorize,
    load_from_checkpoint,
    init_or_load_model,
)


def main():

    # CLI arguments
    parser = arg.ArgumentParser(
        description="Training method for\t"
        "High Quality Monocular Depth Estimation via Transfer Learning"
    )
    parser.add_argument(
        "--epochs",
        "-e",
        default=35,
        type=int,
        help="total number of epochs to run training for",
    )
    parser.add_argument(
        "--lr", "-l", default=0.0001, type=float, help="initial learning rate"
    )
    parser.add_argument("--batch", "-b", default=8, type=int, help="Batch size")
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="",
        type=str,
        help="path to last saved checkpoint to resume training from",
    )
    parser.add_argument(
        "--resume_epoch",
        "-r",
        default=-1,
        type=int,
        help="epoch to resume training from",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        type=str,
        help="device to run training on. Use CUDA",
    )
    parser.add_argument(
        "--enc_pretrain", "-p", default=True, type=bool, help="Use pretrained encoder"
    )
    parser.add_argument(
        "--data", default=r"E:\NYU\nyu_depth_v2_labeled.mat", type=str, help="path to dataset"
    )
    parser.add_argument(
        "--theta", "-t", default=0.1, type=float, help="coeff for L1 (depth) Loss"
    )
    parser.add_argument(
        "--save", "-s", default="", type=str, help="location to save checkpoints in"
    )

    args = parser.parse_args()

    # Some sanity checks
    if len(args.save) > 0 and not args.save.endswith("/"):
        raise ValueError(
            "save location should be path to directory or empty. (Must end with /"
        )
    if len(args.save) > 0 and not os.path.isdir(args.save):
        raise NotADirectoryError("{} not a dir path".format(args.save))

    # Load data
    print("Loading Data ...")
    trainloader, testloader = getTrainingTestingData(args.data, batch_size=args.batch)
    print("Dataloaders ready ...")
    num_trainloader = len(trainloader)
    num_testloader = len(testloader)

    # Training utils
    model_prefix = "densedepth_"
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    theta = args.theta

    save_count = 0
    epoch_loss = []
    batch_loss = []
    sum_loss = 0

    # loading from checkpoint if provided
    if len(args.checkpoint) > 0:
        print("Loading from checkpoint ...")

        model, optimizer, start_epoch = init_or_load_model(
            depthmodel=DenseDepth,
            enc_pretrain=args.enc_pretrain,
            epochs=args.epochs,
            lr=args.lr,
            ckpt=args.checkpoint,
            device=device,
        )
        print("Resuming from: epoch #{}".format(start_epoch))

    else:
        print("Initializing fresh model ...")

        model, optimizer, start_epoch = init_or_load_model(
            depthmodel=DenseDepth,
            enc_pretrain=args.enc_pretrain,
            epochs=args.epochs,
            lr=args.lr,
            ckpt=None,
            device=device,
        )

    # Logging
    log_dir = os.path.join(args.save, 'logs') if args.save else 'runs'
    writer = SummaryWriter(log_dir=log_dir)
    # 在初始化 SummaryWriter 后添加
    print(f"[DEBUG] 日志将保存到: {os.path.abspath(writer.log_dir)}")

    # Loss functions
    l1_criterion = nn.L1Loss()

    # Starting training
    print("Device: ", device)
    print("开始训练 ... ")

    for epoch in range(start_epoch, args.epochs):

        model.train()
        model = model.to(device)

        batch_time = AverageMeter()
        loss_meter = AverageMeter()

        epoch_start = time.time()
        end = time.time()

        for idx, batch in enumerate(trainloader):

            optimizer.zero_grad()

            image_x = torch.Tensor(batch["image"]).to(device)
            depth_y = torch.Tensor(batch["depth"]).to(device=device)

            normalized_depth_y = DepthNorm(depth_y)
            print(f"[DEBUG] 归一化深度范围: {normalized_depth_y.min().item():.4f} - {normalized_depth_y.max().item():.4f}")

            preds = model(image_x)

            # calculating the losses
            l1_loss = l1_criterion(preds, normalized_depth_y)

            ssim_loss = torch.clamp(
                (1 - ssim_criterion(preds, normalized_depth_y, 1.0)) * 0.5,
                min=0,
                max=1,
            )

            gradient_loss = gradient_criterion(normalized_depth_y, preds, device=device)

            net_loss = (
                (1.0 * ssim_loss)
                + (1.0 * torch.mean(gradient_loss))
                + (theta * torch.mean(l1_loss))
            )

            loss_meter.update(net_loss.data.item(), image_x.size(0))
            net_loss.backward()
            optimizer.step()

            # Time metrics
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(
                datetime.timedelta(
                    seconds=int(batch_time.val * (num_trainloader - idx))
                )
            )

            # Logging
            num_iters = epoch * num_trainloader + idx
            if idx % 5 == 0:
                print(
                    "Epoch: #{0} Batch: {1}/{2}\t"
                    "Time (current/total) {batch_time.val:.3f}/{batch_time.sum:.3f}\t"
                    "eta {eta}\t"
                    "LOSS (current/average) {loss.val:.4f}/{loss.avg:.4f}\t".format(
                        epoch,
                        idx,
                        num_trainloader,
                        batch_time=batch_time,
                        eta=eta,
                        loss=loss_meter,
                    )
                )


            # if idx % 300 == 0:
            # LogProgress(model, writer, testloader, num_iters, device)
            # print(torch.cuda.memory_allocated()/1e+9)
            del image_x
            del depth_y
            del preds
        # print(torch.cuda.memory_allocated()/1e+9)

        if epoch % 1 == 0:
            print(
                "----------------------------------\n"
                "Epoch: #{0}, Avg. Net Loss: {avg_loss:.4f}\n"
                "----------------------------------".format(
                    epoch, avg_loss=loss_meter.avg
                )
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "loss": loss_meter.avg,
                },
                args.save + "ckpt_{}_{}.pth".format(epoch, int(loss_meter.avg * 100)),
            )

            writer.add_scalar("Train/L1 Loss", l1_loss.item(), epoch * len(trainloader) + idx)
            writer.add_scalar("Train/SSIM Loss", ssim_loss.item(), epoch * len(trainloader) + idx)
            writer.add_scalar("Train/Grad Loss", gradient_loss.item(), epoch * len(trainloader) + idx)
            writer.add_scalar("Train/Loss", loss_meter.val, num_iters)
            # model = model.to(device)
            LogProgress(model, writer, testloader, epoch, device)

        # cpu检查点
        # if epoch % 5 == 0:
        #
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state_dict": model.cpu().state_dict(),
        #             "optim_state_dict": optimizer.state_dict(),
        #             "loss": loss_meter.avg,
        #         },
        #         args.save + "ckpt_{}_{}.pth".format(epoch, int(loss_meter.avg * 100)),
        #     )
        #
        #     # save_count = (args.epochs % 5) + save_count


def LogProgress(model, writer, test_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)  # 毫米级深度 [0~10000]
        preds = model(image)
        preds_mm = preds * 10000  # 反归一化到毫米

        # 记录图像（显式指定范围）
        single_image = image[2].unsqueeze(0)  # 保持批次维度为1
        single_depth = depth[2]
        single_pred = preds_mm[2]

        # 记录图像（显式指定范围）
        writer.add_image("Input", vision_utils.make_grid(single_image, normalize=True), epoch)
        writer.add_image("GroundTruth", colorize(single_depth, vmin=0, vmax=10000), epoch)
        writer.add_image("Prediction", colorize(single_pred, vmin=0, vmax=10000), epoch)

        # 差异图（限制显示范围，同样使用单个样本）
        diff = torch.abs(single_pred - single_depth)
        writer.add_image("Difference", colorize(diff, vmin=0, vmax=1000), epoch)

        # 直方图（可保留批次数据）
        writer.add_histogram("Pred Depth", preds_mm, epoch)

    del image, depth, preds


if __name__ == "__main__":
    main()
