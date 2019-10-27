# AutoAssist Pytorch Image Classification Implementation
Code for image classifiction accompanying the NeurIPS 2019 paper [AutoAssist: A Framework to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1905.03381.pdf).


# Prerequisites
    - Python (v3.7)
    - PyTorch (v1.0.1)
    - torchvision (v0.2.2)
    - h5py, numpy

# Main Usage
To train AutoAssist model:
```
python train_autoassist.py --assistant [OPTIONS]
OPTIONS:
    --dataset DATASET           dataset choice [mnist, mnist_rot, cifar10, cifar10_rot, imagenet32x32, imagenet]
    --batch-size BSZ            batch size for training 
    --test-batch-size TEST_BSZ  batch size for testing 
    --epochs EPOCH              number of epochs to train
    --lr LR                     learning rate
    --lr-decay LR_DECAY         learning rate decay ratio, 1.0 means no decay
    --base_prob BASE_PROB       base pass probability for Assistant
    --momentum MOMENT           SGD momentum (default: 0.9)')
    --no-cuda                   disables CUDA training
    --seed SEED                 random seed
    --log-interval LOG_INT      how many batches to wait before logging training status
    --arch ARCH                 model architecture [resnet18, resnet34, resnet101]
```
To train other models:
```
python train_others.py [OPTIONS]
OPTIONS:
    --dataset DATASET           dataset choice [mnist, mnist_rot, cifar10, cifar10_rot, imagenet32x32, imagenet]
    --batch-size BSZ            batch size for training 
    --test-batch-size TEST_BSZ  batch size for testing 
    --epochs EPOCH              number of epochs to train
    --lr LR                     learning rate
    --lr-decay LR_DECAY         learning rate decay ratio, 1.0 means no decay
    --base_prob BASE_PROB       base pass probability for Assistant
    --momentum MOMENT           SGD momentum (default: 0.9)')
    --spl                       use self paced learning
    --screener                  use screener-net
    --no-cuda                   disables CUDA training
    --seed SEED                 random seed
    --log-interval LOG_INT      how many batches to wait before logging training status
    --arch ARCH                 model architecture [resnet18, resnet34, resnet101]
```

# Quick Start on MNIST dataset
For a quick start, please execute ```scripts/run_*```. For example the baseline SGD model:
```
    $ bash ./scripts/run_base # for the baseline SGD model
    $ bash ./scripts/run_spl # for the self paced learning model
    $ bash ./scripts/run_snet # for the ScreenerNet model
    $ bash ./scripts/run_autoassist # for the AutoAssist model
```

