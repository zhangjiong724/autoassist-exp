from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataloader import *
from assistant import AssistantSampler, AssistantImagenetSampler
import time, os, sys
import numpy as np
import resnet_mnist as avail_models
from train_others import test
from image_loader import ImagenetRandomSampler


model_names = sorted(
    name
    for name in avail_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(avail_models.__dict__[name])
)

file_dir = os.path.dirname(os.path.realpath(__file__))

from queue import Empty, Full
import multiprocessing


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


mp = torch.multiprocessing.get_context("spawn")

BQ_thread_killer = thread_killer()
BQ_thread_killer.set_tokill(False)
boss_queue = mp.Queue(maxsize=20)

AQ_thread_killer = thread_killer()
AQ_thread_killer.set_tokill(False)
assistant_queue = mp.Queue(maxsize=20)

cuda_transfers_thread_killer = thread_killer()
cuda_transfers_thread_killer.set_tokill(False)
cuda_batches_queue = mp.Queue(maxsize=3)

device = torch.device("cuda")


def BQ_feeder(rank, args, train_data, sampler):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    if args.assistant:
        train_kwargs = {"num_workers": 0, "sampler": sampler}
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False, **train_kwargs
        )
    else:
        # train_kwargs = {'num_workers': 1, 'pin_memory': True}
        train_kwargs = {}
        if args.dataset == "imagenet":
            train_kwargs["sampler"] = ImagenetRandomSampler(train_data, repeat_chunk=1)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, **train_kwargs
        )

    while BQ_thread_killer() == False:
        for (
            batch,
            (batch_images, batch_labels, batch_weights, batch_indices),
        ) in enumerate(train_loader):
            if boss_queue.qsize() >= 5 and not assistant_queue.empty():
                # print("Try to train Assistant with BQ.size =  %d"%( boss_queue.qsize()))
                batch, (
                    batch_indices,
                    mean_loss,
                    batch_predictions,
                    batch_losses,
                ) = assistant_queue.get(block=False)
                data0, target0, _, _ = train_data[batch_indices]
                data0 = torch.stack(list(data0))
                target0 = torch.stack(list(target0))
                sampler.train_step(
                    data0,
                    target0.numpy(),
                    batch_losses,
                    mean_loss,
                    0,
                    batch_predictions,
                    args.sec_method,
                )
                # print("Succeeded")
            # We fill the queue with new fetched batch until we reach the max       size.
            boss_queue.put(
                (batch, (batch_images, batch_labels, batch_weights, batch_indices))
            )
            # boss_queue.put("queue_size=%d"%(boss_queue.qsize()), block=False)
            # print("I am rank %d, queue size = %d"%(rank, boss_queue.qsize()))
            if BQ_thread_killer() == True:
                print("BQ rank %d, exiting" % (rank))
                return
    print("I am rank %d, exiting" % (rank))
    return


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch AutoAssist Network for image classification"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset choice [mnist, mnist_rot, cifar10, cifar10_rot, imagenet32x32]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 10)",
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
        "--sec_method",
        type=str,
        default="mean",
        help="method to shrink examples (default: mean)",
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
        "--log-train-loss-file",
        type=str,
        default="tmp_loss",
        help="file to log the total train loss",
    )
    parser.add_argument(
        "--log-train-loss-interval",
        type=int,
        default=0,
        help="epoch interval to log the total train loss, 0 means no logging",
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
            download=True,
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
            download=True,
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
        from screener import Screener_CIFAR10 as Screener

        num_classes = 1000
        args.lr_dacay = 0.5
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        print("Loading imagenet train")
        train_data = ImageNet(
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
        print("Loading imagenet val")
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
    else:
        print("Unrecognized dataset")

    test_kwargs = {"num_workers": 1, "pin_memory": True}
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=True, **test_kwargs
    )

    print("=> creating model '{}' on '{}'".format(args.arch, device))
    model = models.__dict__[args.arch](num_classes=num_classes).to(device).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # setting up queues
    BQ_workers = 1
    AQ_workers = 1
    if args.dataset in ["imagenet"]:
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

    processes = []
    for i in range(BQ_workers):
        t = multiprocessing.Process(
            target=BQ_feeder, args=(i, args, train_data, sampler)
        )
        t.start()
        processes.append(t)

    # Wait for the Assistants to init
    while boss_queue.qsize() < 10:
        time.sleep(1)
    t0 = time.time()

    if args.dataset == "imagenet":
        batches_per_epoch = 1281167 // args.batch_size + 1
    else:
        batches_per_epoch = len(train_data) // args.batch_size + 1
    total_time = 0
    loss_sum = 0.0
    loss_square_sum = 0.0
    total_instances = 0
    alpha = 0.75
    if args.log_train_loss_interval > 0:
        if not os.path.exists(args.log_train_loss_file):
            os.mkdir(args.log_train_loss_file)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (args.lr_decay ** (epoch // 30))
        print("Current learning rate %f" % (lr))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer

    sampler = None

    for epoch in range(1, args.epochs + 1):

        if args.lr_decay:
            optimizer = adjust_learning_rate(optimizer, epoch)

        for batch_idx in range(batches_per_epoch):
            _, (data0, target0, weight0, index) = boss_queue.get(block=True)
            data, target, weight = (
                data0.to(device),
                target0.to(device),
                weight0.float().to(device),
            )
            optimizer.zero_grad()
            output = model(data)
            losses = F.nll_loss(output, target, reduce=False)

            pred = (
                output.max(1, keepdim=True)[1].cpu().numpy().T
            )  # get the index of the max log-probability
            total_instances *= alpha
            total_instances += (1.0 - alpha) * data.size(0)

            loss_sum *= alpha
            loss_sum += (1.0 - alpha) * losses.sum().item()

            loss_square_sum *= alpha
            loss_square_sum += (1.0 - alpha) * losses.pow(2).sum().item()

            loss_mean = loss_sum / total_instances
            loss_square_mean = loss_square_sum / total_instances

            loss_dev = np.sqrt(loss_square_mean - loss_mean ** 2)

            assistant_queue.put(
                (batch_idx, (index, loss_mean, pred, losses.detach().cpu().numpy()))
            )

            loss = losses.mean()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == args.log_interval - 1:
                total_time += time.time() - t0
                print(
                    "Train Epoch: {} [{}/{}]\tmean_loss: {:6.2f}\tTime: {:6.2f}\tTime_sampling: {:6.2f}\tLoss: {:.6f}".format(
                        epoch - 1,
                        batch_idx * len(data),
                        batches_per_epoch,
                        loss_mean,
                        time.time() - t0,
                        0,
                        loss.item(),
                    )
                )
                if (batch_idx // args.log_interval) % 5 == 0:
                    test(args, model, device, test_loader, total_time)
                sys.stdout.flush()
                t0 = time.time()

    BQ_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for p in processes:
        print("Terminating process")
        p.terminate()
    print(
        "Training %d batches done in %f seconds"
        % (batches_per_epoch * args.epochs, time.time() - t0)
    )


if __name__ == "__main__":
    main()
