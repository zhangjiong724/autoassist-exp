from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataloader import *
from assistant import AssistantSampler, AssistantImagenetSampler
import time, os
import numpy as np
import resnet_mnist as avail_models
import resnet_cifar10 as avail_models
from image_loader import ImagenetRandomSampler

model_names = sorted(
    name
    for name in avail_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(avail_models.__dict__[name])
)

file_dir = os.path.dirname(os.path.realpath(__file__))


def train(
    args,
    model,
    device,
    train_loader,
    test_loader,
    optimizer,
    sampler=None,
    snet=None,
    train_test_loader=None,
):
    model.train()
    total_time = 0
    loss_sum = 0.0
    loss_square_sum = 0.0
    total_instances = 0

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (args.lr_decay ** (epoch // 30))
        print("Current learning rate %f" % (lr))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer

    N = len(train_loader.dataset)

    if args.spl:
        loss_chart = np.zeros([N], dtype=float) + 1000.0  # begin with every one
        pace = np.linspace(0.25, 1.0, 5)
        assert args.epochs >= 5
        print(N)

    if args.screener:
        if "mnist" in args.dataset:
            optimizer_s = optim.Adam(snet.parameters(), lr=0.0001)
        elif "cifar10" in args.dataset:
            optimizer_s = optim.Adam(snet.parameters(), lr=0.0025)
        elif "imagenet" in args.dataset:
            optimizer_s = optim.Adam(snet.parameters(), lr=0.0025)
        snet.train()

    alpha = 0.75

    for epoch in range(1, args.epochs + 1):
        optimizer = adjust_learning_rate(optimizer, epoch)
        t0 = time.time()
        t1 = time.time()
        time_sampling = 0

        if args.spl:
            if epoch > 5:
                SortN = len(train_loader.dataset)
            else:
                SortN = int(len(train_loader.dataset) * pace[epoch - 1])
            lambda_t = np.sort(loss_chart)[SortN - 1] + 1e-6
            print("Current SortN= %d, Lambda_t = %f" % (SortN, lambda_t))

        for batch_idx, (data0, target0, weight, index) in enumerate(train_loader):
            time_sampling += time.time() - t1
            data, target, weight = (
                data0.to(device),
                target0.to(device),
                weight.float().to(device),
            )
            output = model(data)
            # loss = F.nll_loss(output, target, reduce = False).mean()
            losses = F.nll_loss(output, target, reduce=False)

            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy().T  # get the index of the max log-probability

            total_instances *= alpha
            total_instances += (1.0 - alpha) * data.size(0)

            loss_sum *= alpha
            loss_sum += (1.0 - alpha) * losses.sum().item()

            loss_square_sum *= alpha
            loss_square_sum += (1.0 - alpha) * losses.pow(2).sum().item()

            loss_mean = loss_sum / total_instances
            loss_square_mean = loss_square_sum / total_instances

            loss_dev = np.sqrt(loss_square_mean - loss_mean ** 2)

            if args.assistant:
                t1 = time.time()

                new_weights = sampler.train_step(
                    data0,
                    target0.numpy(),
                    losses.detach().cpu().numpy(),
                    loss_mean,
                    loss_dev,
                    pred,
                )
                time_sampling += time.time() - t1

            # self paced learning
            if args.spl:
                loss_chart[index] = losses.cpu().detach().numpy()

                q_t = 4.0
                base_weight = 0.01

                def self_paced_weight(losses):
                    act = torch.nn.ReLU()
                    sp_binary = (losses < lambda_t).float().to(device)
                    sp_weights = act(1.0 - losses / lambda_t) ** (1.0 / (q_t - 1.0))
                    weight_sum = sp_weights.sum() + base_weight * args.batch_size
                    return (args.batch_size / weight_sum) * (sp_weights + base_weight)

                train_weight = self_paced_weight(losses)
            elif args.assistant:
                new_weights = (args.batch_size / new_weights.sum()) * new_weights
                train_weight = torch.from_numpy(new_weights).float().to(device)
            else:
                train_weight = weight

            if not args.screener:
                optimizer.zero_grad()
                loss = (losses * train_weight).mean()
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                x_w = snet(data).squeeze()
                loss = torch.mean(losses * x_w)
                loss.backward(retain_graph=True)
                optimizer.step()  # update net

                optimizer_s.zero_grad()
                loss_s = snet.snet_loss(x_w, losses, snet.parameters())
                loss_s.backward()
                optimizer_s.step()

            print_loss = losses.mean().item()

            if batch_idx % args.log_interval == args.log_interval - 1:
                total_time += time.time() - t0
                print(
                    "Train Epoch: {} [{}/{}]\tmean_loss: {:6.2f}\tTime: {:6.2f}\tTime_sampling: {:6.2f}\tLoss: {:.6f}".format(
                        epoch - 1,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        loss_mean,
                        time.time() - t0,
                        time_sampling,
                        print_loss,
                    )
                )
                if (batch_idx // args.log_interval) % 5 == 0:
                    test(args, model, device, test_loader, total_time)
                t0 = time.time()
                time_sampling = 0
            t1 = time.time()
            if batch_idx >= len(train_loader):
                break


def test(
    args,
    model,
    device,
    test_loader,
    total_time,
    all_loss_log_file=None,
    train_or_test="Test",
):
    model.eval()
    test_loss = 0
    correct = 0

    all_losses = []

    with torch.no_grad():
        for data, target, weight, index in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if all_loss_log_file is not None:
                test_losses = F.nll_loss(output, target, reduce=False).cpu().numpy()
                all_losses.append(test_losses)
                test_loss += test_losses.sum().item()
            else:
                test_loss += (
                    F.nll_loss(output, target, reduce=False).sum().item()
                )  # sum up batch loss
                pred = output.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if all_loss_log_file is not None:
        all_losses = np.concatenate(all_losses)
        np.save(all_loss_log_file, all_losses)
        model.train()

    else:
        print("=" * 80)
        print(
            train_or_test,
            " set: Average loss: {:.4f} | Accuracy: {:.4f} | Time: {:6.2f} ".format(
                test_loss, 100.0 * correct / len(test_loader.dataset), total_time
            ),
        )
        print("=" * 80)
        model.train()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch AutoAssist Network example")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset choice [mnist, mnist_rot, cifar10, cifar10_rot, imagenet32x32, imagenet]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lr-decay", type=float, default=1.0, help="learning rate decay rate"
    )
    parser.add_argument(
        "--base_prob",
        type=float,
        default=0.3,
        help="base pass probability (default: 0.3)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--assistant", action="store_true", default=False, help="use assistant"
    )
    parser.add_argument("--spl", action="store_true", default=False, help="use SPL")
    parser.add_argument(
        "--screener", action="store_true", default=False, help="use screener"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="BaseNet",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dataset == "mnist":
        import resnet_mnist as models
        from screener import Screener_MNIST as Screener

        num_classes = 10
        train_data = MNIST(
            file_dir + "/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        test_data = MNIST(
            file_dir + "/data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    elif args.dataset == "mnist_rot":
        import resnet_mnist as models
        from screener import Screener_MNIST as Screener

        num_classes = 10
        train_data = MNIST(
            file_dir + "/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomRotation(90),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        test_data = MNIST(
            file_dir + "/data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    elif args.dataset == "cifar10":
        import resnet_cifar10 as models
        from screener import Screener_CIFAR10 as Screener

        num_classes = 10
        args.lr_dacay = 0.5
        train_data = CIFAR10(
            file_dir + "/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )

        test_data = CIFAR10(
            file_dir + "/data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )
    elif args.dataset == "cifar10_rot":
        import resnet_cifar10 as models
        from screener import Screener_CIFAR10 as Screener

        args.lr_dacay = 0.5
        num_classes = 10
        train_data = CIFAR10(
            file_dir + "/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomRotation(90),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )

        test_data = CIFAR10(
            file_dir + "/data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )
    elif args.dataset == "imagenet32x32":
        import resnet_cifar10 as models
        from screener import Screener_CIFAR10 as Screener

        num_classes = 1000
        args.lr_dacay = 0.5
        train_data = ImageNet32X32(
            file_dir + "/data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )
        print("out:", train_data.mean_image)
        test_data = ImageNet32X32(
            file_dir + "/data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
            mean_image=train_data.mean_image,
        )
    elif args.dataset == "imagenet":
        import resnet_imagenet as models
        from screener import Screener_ImageNet as Screener

        num_classes = 1000
        args.lr_dacay = 0.5
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        print("Loading imagenet train")
        train_data = ImageNet(
            file_dir + "/data/imagenet_raw/",
            split="train",
            download=True,
            prefetch=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        print("Loaded imagenet val")
        test_data = ImageNet(
            file_dir + "/data/imagenet_raw/",
            split="val",
            download=True,
            prefetch=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    if args.screener:
        snet = Screener().to(device)
    else:
        snet = None

    sampler = None
    if args.assistant:
        if args.dataset == "imagenet":
            sampler = AssistantImagenetSampler(
                train_data,
                base_prob=args.base_prob,
                num_classes=num_classes,
                repeat_chunk=1,
            )
        else:
            sampler = AssistantSampler(
                train_data, base_prob=args.base_prob, num_classes=num_classes
            )
        train_kwargs = {"num_workers": 0, "pin_memory": True, "sampler": sampler}
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False, **train_kwargs
        )
    else:
        train_kwargs = {"num_workers": 1, "pin_memory": True}
        if args.dataset == "imagenet":
            train_kwargs["sampler"] = ImagenetRandomSampler(train_data, repeat_chunk=1)
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=False, **train_kwargs
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True, **train_kwargs
            )

    test_kwargs = {"num_workers": 1, "pin_memory": True}
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=True, **test_kwargs
    )

    train_test_kwargs = {"num_workers": 1, "pin_memory": True}
    train_test_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.test_batch_size, shuffle=True, **train_test_kwargs
    )

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(
        args,
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        sampler=sampler,
        snet=snet,
        train_test_loader=train_test_loader,
    )


if __name__ == "__main__":
    main()
