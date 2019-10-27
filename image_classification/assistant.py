import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import numpy as np
import threading, time
from torchvision import datasets, transforms
from sklearn.linear_model import SGDClassifier


eps = 1e-10


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def sigmoid(x):
    """Compute softmax values for each sets of scores in x."""
    return 1.0 / (1 + np.exp(-x))


class AssistantImagenetSampler(Sampler):
    r""" Generate instances based on Assistant model.
    Arguments:
        data_source (Dataset): dataset to sample from
        base_prob (float): ground probability for an instance to pass
        num_classes (int): number of classes for classification
        repeat_chunk: (int) number of times to repeat a single chunk 
    """

    def __init__(self, data_source, base_prob=0.3, num_classes=1000, repeat_chunk=1):
        self.data_source = data_source
        try:
            self.indices = data_source.current_indices
        except:
            self.indices = list(range(len(data_source)))
        self.repeat = repeat_chunk

        img_example, _, _, _ = data_source[0]
        data_dim = torch.numel(img_example)

        self.assistant = AssistantModelBinary(data_dim, num_classes)

        self.base_prob = base_prob

        self.total_samples = 1
        self.total_success = 1
        self.sec_loss = 10.0

    def __iter__(self):
        # given idx i, return an instance
        rpt = 0
        while True:
            shuffled_indices = [
                self.indices[i] for i in torch.randperm(len(self.indices))
            ]
            cur_len = len(self.indices)
            for index in shuffled_indices:
                coin = np.random.uniform()
                # coin2 = np.random.uniform()
                self.total_samples += 1  # not thread safe!!
                if coin < self.base_prob:
                    self.total_success += 1  # not thread safe
                    yield index
                else:
                    # continue
                    coin = (coin - self.base_prob) / (
                        1.0 - self.base_prob
                    )  # renormalize coin, still independent variable

                    # compute importance
                    x, y, _, _ = self.data_source[index]
                    y = y.item()
                    keep_prob = self.assistant.get_importance(x, y)
                    if coin < keep_prob:
                        self.total_success += 1  # not thread safe
                        yield index
            rpt += 1
            if rpt == self.repeat:
                rpt = 0
                self.data_source.load_next_chunk()
                self.indices = self.data_source.current_indices

    def __len__(self):
        return len(self.data_source)

    def rate(self):
        return float(self.total_success) / self.total_samples

    def train_step(self, X, Y, loss, mean, dev, pred, method="mean"):
        self.sec_loss *= 0.0
        acc, probs = self.assistant.train_step(
            X, Y, loss, mean, dev, pred, method=method
        )
        self.sec_loss += 1 * acc
        return probs


class AssistantSampler(Sampler):
    r""" Generate instances based on Assistant model.
    Arguments:
        data_source (Dataset): dataset to sample from
        base_prob (float): ground probability for an instance to pass
        num_classes (int): number of classes for classification
    """

    def __init__(self, data_source, base_prob=0.3, num_classes=10):
        self.data_source = data_source

        img_example, _, _, _ = data_source[0]
        data_dim = torch.numel(img_example)

        self.assistant = AssistantModelBinary(data_dim, num_classes)

        self.base_prob = base_prob

        self.total_samples = 1
        self.total_success = 1
        self.threadLock = threading.Lock()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.sec_loss = 10.0

    def __iter__(self):
        # given idx i, return an instance
        index = -1
        while True:
            index = (index + 1) % len(self.data_source)
            coin = np.random.uniform()
            # coin2 = np.random.uniform()
            self.total_samples += 1  # not thread safe!!
            x, y, weight, acc = self.data_source[index]
            # x = x.view(-1).numpy()
            y = y.item()
            if coin < self.base_prob:
                self.total_success += 1  # not thread safe
                yield index
            else:
                # continue
                coin = (coin - self.base_prob) / (
                    1.0 - self.base_prob
                )  # renormalize coin, still independent variable

                # compute importance
                keep_prob = self.assistant.get_importance(x, y)
                if coin < keep_prob:
                    self.total_success += 1  # not thread safe
                    yield index

    def __len__(self):
        return len(self.data_source)

    def rate(self):
        return float(self.total_success) / self.total_samples

    def train_step(self, X, Y, loss, mean, dev, pred, method="mean"):
        self.sec_loss *= 0.0
        acc, probs = self.assistant.train_step(
            X, Y, loss, mean, dev, pred, method=method
        )
        self.sec_loss += 1 * acc
        return probs


class AssistantModelBinary(nn.Module):
    """
    predict p( not_trivial | x_i,  y_i) = sigmoid( W*x_i + U[y_i] )
        where:
            not_trivial = ( loss_i > loss_mean - loss_stddev)
    Arguments:
        dimension (int): input data vector dimension
        num_classes (int): number of classes for classification
    """

    def __init__(self, dimension, num_class):
        super(AssistantModelBinary, self).__init__()
        self.dimension = dimension
        self.num_class = num_class
        self.classes = np.arange(num_class)
        self.lr = 0.01
        self.lam = 1e-3
        self.fitted = 0

        self.W = 0.001 * np.random.randn(dimension).astype(np.float32)
        self.U = 0.001 * np.random.randn(num_class).astype(np.float32)

    def get_importance(self, x, y):
        return sigmoid(self.W.dot(x.view(self.dimension)) + self.U[y])

    def make_target_by_mean(self, loss, mean, dev, Y, pred):
        return np.array(loss > mean, dtype=int)

    def make_target_by_pred(self, loss, mean, dev, Y, pred):
        return np.array(Y != pred, dtype=int)

    def train_step(self, X, Y, loss, mean, dev, pred, method):
        self.fitted += 1
        batch_size = Y.shape[0]
        lr = self.lr / Y.shape[0]

        if method == "mean":
            label = self.make_target_by_mean(loss, mean, dev, Y, pred).reshape(
                (batch_size,)
            )
        elif method == "pred":
            label = self.make_target_by_pred(loss, mean, dev, Y, pred).reshape(
                (batch_size,)
            )

        X = X.reshape([batch_size, self.dimension])

        prob = sigmoid(X.numpy().dot(self.W) + self.U[Y])  # shape = (batch_size,)
        predict = np.array(prob > 0.5, dtype=int)
        acc = np.sum(label == predict) * 1.0

        grad = prob - label
        # gradient update

        for i in range(batch_size):
            self.U[Y[i]] -= lr * (grad[i] + self.lam * self.U[Y[i]])
        self.W -= lr * (grad.dot(X) + self.lam * self.W).squeeze()

        return acc / batch_size, prob
