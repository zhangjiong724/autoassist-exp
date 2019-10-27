# coding: utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os, sys, glob, time, math
import os.path, random
import errno
import numpy as np
import torch, threading
import codecs
import shutil
import torchvision
from torchvision.datasets.utils import download_url, check_integrity
import multiprocessing
from image_loader import default_loader

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data

ARCHIVE_DICT = {
    "train": {
        "url": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar",
        "md5": "1d675b47d978889d74fa0da5fadfb00e",
    },
    "val": {
        "url": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar",
        "md5": "29b22e2961454d5413ddabcf34fc5622",
    },
    "devkit": {
        "url": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz",
        "md5": "fa75699e90414af021442c21a62c3abf",
    },
}


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]
    raw_folder = "mnist/raw"
    processed_folder = "mnist/processed"
    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            train_data, train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file)
            )
            self.train_data, self.train_labels = self.preprocess(
                train_data, train_labels
            )
            self.correct = np.zeros((len(self.train_data),), dtype=float)
            self.weights = np.ones((len(self.train_data),), dtype=float)
        else:
            test_data, test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file)
            )
            self.test_data, self.test_labels = self.preprocess(test_data, test_labels)
            self.correct = np.zeros((len(self.test_data),), dtype=float)
            self.weights = np.ones((len(self.test_data),), dtype=float)

    def preprocess(self, raw_data, raw_labels):
        N = raw_labels.size(0)
        ret_data = np.empty((N,), dtype=object)
        ret_labels = np.empty((N,), dtype=object)
        for index in range(N):
            img, target = raw_data[index], raw_labels[index]
            img = Image.fromarray(img.numpy(), mode="L")
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            ret_data[index] = img
            ret_labels[index] = target
        return ret_data, ret_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target, self.weights[index], index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, self.training_file)
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition("/")[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            download_url(
                url,
                root=os.path.join(self.root, self.raw_folder),
                filename=filename,
                md5=None,
            )
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(
                file_path
            ) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print("Processing...")

        training_set = (
            read_image_file(
                os.path.join(self.root, self.raw_folder, "train-images-idx3-ubyte")
            ),
            read_label_file(
                os.path.join(self.root, self.raw_folder, "train-labels-idx1-ubyte")
            ),
        )
        test_set = (
            read_image_file(
                os.path.join(self.root, self.raw_folder, "t10k-images-idx3-ubyte")
            ),
            read_label_file(
                os.path.join(self.root, self.raw_folder, "t10k-labels-idx1-ubyte")
            ),
        )
        with open(
            os.path.join(self.root, self.processed_folder, self.training_file), "wb"
        ) as f:
            torch.save(training_set, f)
        with open(
            os.path.join(self.root, self.processed_folder, self.test_file), "wb"
        ) as f:
            torch.save(test_set, f)

        print("Done!")

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    urls = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    ]
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    class_to_idx = {_class: i for i, _class in enumerate(classes)}


class EMNIST(MNIST):
    """`EMNIST <https://www.nist.gov/itl/iad/image-group/emnist-dataset/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
    splits = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")

    def __init__(self, root, split="byclass", **kwargs):
        if split not in self.splits:
            raise ValueError(
                'Split "{}" not found. Valid splits are: {}'.format(
                    split, ", ".join(self.splits)
                )
            )
        self.split = split
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)

    def _training_file(self, split):
        return "training_{}.pt".format(split)

    def _test_file(self, split):
        return "test_{}.pt".format(split)

    def download(self):
        """Download the EMNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        import shutil
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        filename = self.url.rpartition("/")[2]
        raw_folder = os.path.join(self.root, self.raw_folder)
        file_path = os.path.join(raw_folder, filename)
        download_url(self.url, root=file_path, filename=filename, md5=None)

        print("Extracting zip archive")
        with zipfile.ZipFile(file_path) as zip_f:
            zip_f.extractall(raw_folder)
        os.unlink(file_path)
        gzip_folder = os.path.join(raw_folder, "gzip")
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith(".gz"):
                print("Extracting " + gzip_file)
                with open(
                    os.path.join(raw_folder, gzip_file.replace(".gz", "")), "wb"
                ) as out_f, gzip.GzipFile(
                    os.path.join(gzip_folder, gzip_file)
                ) as zip_f:
                    out_f.write(zip_f.read())
        shutil.rmtree(gzip_folder)

        # process and save as torch files
        for split in self.splits:
            print("Processing " + split)
            training_set = (
                read_image_file(
                    os.path.join(
                        raw_folder, "emnist-{}-train-images-idx3-ubyte".format(split)
                    )
                ),
                read_label_file(
                    os.path.join(
                        raw_folder, "emnist-{}-train-labels-idx1-ubyte".format(split)
                    )
                ),
            )
            test_set = (
                read_image_file(
                    os.path.join(
                        raw_folder, "emnist-{}-test-images-idx3-ubyte".format(split)
                    )
                ),
                read_label_file(
                    os.path.join(
                        raw_folder, "emnist-{}-test-labels-idx1-ubyte".format(split)
                    )
                ),
            )
            with open(
                os.path.join(
                    self.root, self.processed_folder, self._training_file(split)
                ),
                "wb",
            ) as f:
                torch.save(training_set, f)
            with open(
                os.path.join(self.root, self.processed_folder, self._test_file(split)),
                "wb",
            ) as f:
                torch.save(test_set, f)

        print("Done!")


def get_int(b):
    return int(codecs.encode(b, "hex"), 16)


def read_label_file(path):
    with open(path, "rb") as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, "rb") as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [["test_batch", "40351d587109b95175f43aff81a1287e"]]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)

        self._load_meta()

        self.data, self.targets = self.preprocess(self.data, self.targets)

        self.data = torch.from_numpy(np.stack(self.data))
        self.targets = torch.from_numpy(np.stack(self.targets))

        self.correct = np.zeros((len(self.data),), dtype=float)
        self.weights = np.ones((len(self.data),), dtype=float)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def preprocess(self, raw_data, raw_labels):
        N = raw_labels.size
        ret_data = np.empty((N,), dtype=object)
        ret_labels = np.empty((N,), dtype=object)
        for index in range(N):
            img, target = raw_data[index], raw_labels[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            ret_data[index] = img
            ret_labels[index] = target
        return ret_data, ret_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target, self.weights[index], index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]

    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


class ImageNet32X32(data.Dataset):
    """`ImageNet32X32`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    processed_folder = "imagenet32x32"
    img_size = 32

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(
        self, root, train=True, transform=None, target_transform=None, mean_image=None
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.mean_image = mean_image
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            self.train_data, self.train_labels = self.load_alldata()
            self.correct = np.zeros((len(self.train_data),), dtype=float)
            self.weights = np.ones((len(self.train_data),), dtype=float)
        else:
            self.test_data, self.test_labels = self.load_alldata()
            self.correct = np.zeros((len(self.test_data),), dtype=float)
            self.weights = np.ones((len(self.test_data),), dtype=float)

    def load_alldata(self):
        X = []
        Y = []
        if self.train:
            for idx in range(1, 11):
                x, y = self.load_databatch("train_data_batch_" + str(idx))
                X.append(x)
                Y.append(y)
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
        else:
            X, Y = self.load_databatch("val_data")

        return torch.tensor(X), torch.tensor(Y)

    def load_databatch(self, dataname):
        data_file = os.path.join(self.root, self.processed_folder, dataname)

        def unpickle(file):
            with open(file, "rb") as fo:
                dict = pickle.load(fo)
            return dict

        d = unpickle(data_file)
        x = d["data"]
        y = d["labels"]

        if self.train:
            mean_image = d["mean"]
            self.mean_image = mean_image
        else:
            mean_image = self.mean_image

        x = x / np.float32(255)
        mean_image = mean_image / np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i - 1 for i in y]
        data_size = x.shape[0]

        x -= mean_image
        img_size = self.img_size

        img_size2 = img_size * img_size

        x = np.dstack(
            (x[:, :img_size2], x[:, img_size2 : 2 * img_size2], x[:, 2 * img_size2 :])
        )
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train, X_train_flip), axis=0)
        Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

        return X_train, Y_train.astype("int64")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target, self.weights[index], index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return glob.glob(
            os.path.join(self.root, self.processed_folder, "train_data_batch_*")
        ) and glob.glob(os.path.join(self.root, self.processed_folder, "val_data"))

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class ImageNet(torchvision.datasets.ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root,
        split="train",
        download=False,
        prefetch=False,
        num_workers=48,
        loader=default_loader,
        **kwargs
    ):
        root = self.root = os.path.expanduser(root)
        self.split = self._verify_split(split)

        if download:
            self.download()
        wnid_to_classes = self._load_meta_file()[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root
        self.loader = loader

        idcs = [idx for _, idx in self.imgs]
        self.wnids = self.classes
        self.wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, self.wnids)}
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for clss, idx in zip(self.classes, idcs) for cls in clss
        }
        self.weights = np.ones((len(self.samples),), dtype=float)

        self.prefetch = prefetch
        self.prefetched_samples = None
        self.prefetched_targets = None
        self.chunk_datadir = []
        self.chunk_indices = []

        if self.prefetch:
            import h5py

            print("Prefetching ImageNet...")
            prefetch_chunk = 100000
            if split == "train":
                h5f_dir = root + "/h5f/"
            else:
                h5f_dir = root + "/h5f_val/"
            for steps in range(math.ceil(len(self) / prefetch_chunk)):
                h5f_chunk_file = h5f_dir + "chunk_" + str(steps)
                start_file = steps * prefetch_chunk
                end_file = min((steps + 1) * prefetch_chunk, len(self))
                self.chunk_datadir.append(h5f_chunk_file)
                self.chunk_indices.append(list(range(start_file, end_file)))
                if os.path.exists(h5f_chunk_file):
                    print("%s already exist, continue..." % (h5f_chunk_file))
                    continue
                t0 = time.time()
                print(
                    "==== Fetching chunk %d: %d to %d ======"
                    % (steps, start_file, end_file)
                )
                async_samples = multiprocessing.Manager().list(
                    [None] * (end_file - start_file)
                )
                async_targets = multiprocessing.Manager().list(
                    [None] * (end_file - start_file)
                )
                fetch_jobs = []
                for i in range(num_workers):
                    p = multiprocessing.Process(
                        target=self.fetch_data_worker,
                        args=(i, num_workers, async_samples, async_targets, start_file),
                    )
                    fetch_jobs.append(p)
                    p.start()
                for p in fetch_jobs:
                    p.join()

                sample_data_chunk = np.stack(async_samples, axis=0)
                target_data_chunk = np.array(async_targets)
                h5file_towrite = h5py.File(h5f_chunk_file, "w")
                h5file_towrite.create_dataset("samples", data=sample_data_chunk)
                h5file_towrite.create_dataset("targets", data=target_data_chunk)
                h5file_towrite.close()
                print(
                    "%s created using %f seconds" % (h5f_chunk_file, time.time() - t0)
                )

            self.chunk_idx = -1
            self.chunk_sequence = list(range(len(self.chunk_datadir)))
            self.load_next_chunk()

    def load_next_chunk(self):
        num_chunks = len(self.chunk_datadir)
        old_chunk_idx = self.chunk_idx
        chunk_seq_idx = self.chunk_idx + 1
        if chunk_seq_idx >= num_chunks:
            random.shuffle(self.chunk_sequence)
            print("Current chunk sequence:", self.chunk_sequence)
        self.chunk_idx = self.chunk_sequence[chunk_seq_idx % num_chunks]

        self.current_indices = self.chunk_indices[self.chunk_idx]
        self.current_chunk_size = len(self.current_indices)

        self.start_index = self.current_indices[0]
        if old_chunk_idx != self.chunk_idx or self.prefetched_samples is None:
            print(
                "Loading chunk %d (%d to %d)..."
                % (self.chunk_idx, self.current_indices[0], self.current_indices[-1])
            )
            t0 = time.time()
            cur_chunk_data = self.chunk_datadir[self.chunk_idx]
            cur_h5f = h5py.File(self.chunk_datadir[self.chunk_idx], "r")
            # clear mem first
            self.prefetched_samples = None
            self.prefetched_targets = None
            self.prefetched_samples = torch.tensor(cur_h5f["samples"][:])
            self.prefetched_targets = torch.tensor(cur_h5f["targets"][:])
            print("finished with %f seconds" % (time.time() - t0), flush=True)

    def download(self):
        if not check_integrity(self.meta_file):
            tmpdir = os.path.join(self.root, "tmp")

            archive_dict = ARCHIVE_DICT["devkit"]
            download_and_extract_tar(
                archive_dict["url"],
                self.root,
                extract_root=tmpdir,
                md5=archive_dict["md5"],
            )
            devkit_folder = _splitexts(os.path.basename(archive_dict["url"]))[0]
            meta = parse_devkit(os.path.join(tmpdir, devkit_folder))
            self._save_meta_file(*meta)

            shutil.rmtree(tmpdir)

        if not os.path.isdir(self.split_folder):
            archive_dict = ARCHIVE_DICT[self.split]
            download_and_extract_tar(
                archive_dict["url"],
                self.root,
                extract_root=self.split_folder,
                md5=archive_dict["md5"],
            )

            if self.split == "train":
                prepare_train_folder(self.split_folder)
            elif self.split == "val":
                val_wnids = self._load_meta_file()[1]
                prepare_val_folder(self.split_folder, val_wnids)
        else:
            msg = (
                "You set download=True, but a folder '{}' already exist in "
                "the root directory. If you want to re-download or re-extract the "
                "archive, delete the folder."
            )
            print(msg.format(self.split))

    @property
    def meta_file(self):
        return os.path.join(self.root, "meta.bin")

    def _load_meta_file(self):
        if check_integrity(self.meta_file):
            return torch.load(self.meta_file)
        else:
            raise RuntimeError(
                "Meta file not found or corrupted.",
                "You can use download=True to create it.",
            )

    def _save_meta_file(self, wnid_to_class, val_wnids):
        torch.save((wnid_to_class, val_wnids), self.meta_file)

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.prefetch:
            sample = self.prefetched_samples[index % self.current_chunk_size]
            target = self.prefetched_targets[index % self.current_chunk_size]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, torch.tensor(target), self.weights[index], index

    @property
    def valid_splits(self):
        return "train", "val"

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def fetch_data_worker(
        self, proc_id, num_procs, prefetched_samples, prefetched_targets, start_file
    ):
        data_len = len(prefetched_samples)
        chunk_size = math.ceil(data_len / num_procs)
        start_id = proc_id * chunk_size + start_file
        end_id = min((proc_id + 1) * chunk_size, data_len) + start_file
        print("Worker %d: %d to %d" % (proc_id, start_id, end_id))
        n_done = 0
        t0 = time.time()
        for index in range(start_id, end_id):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            prefetched_samples[index - start_file] = sample.numpy()
            prefetched_targets[index - start_file] = target
            n_done += 1
            if n_done % 1000 == 0 and proc_id == 0:
                print("Worker %d finished %d/%d" % (proc_id, n_done, end_id - start_id))
        if proc_id == 0:
            print(
                "Worker %d finished %d images in %f s"
                % (proc_id, n_done, time.time() - t0)
            )


def extract_tar(src, dest=None, gzip=None, delete=False, ignore=0):
    import tarfile

    if dest is None:
        dest = os.path.dirname(src)
    if gzip is None:
        gzip = src.lower().endswith(".gz")

    mode = "r:gz" if gzip else "r"
    with tarfile.open(src, mode) as tarfh:
        tarfh.extractall(path=dest)

    if delete:
        os.remove(src)


def download_and_extract_tar(
    url, download_root, extract_root=None, filename=None, md5=None, **kwargs
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if filename is None:
        filename = os.path.basename(url)

    if not check_integrity(os.path.join(download_root, filename), md5) and False:
        download_url(url, download_root, filename=filename, md5=md5)

    extract_tar(os.path.join(download_root, filename), extract_root, **kwargs)


def parse_devkit(root):
    idx_to_wnid, wnid_to_classes = parse_meta(root)
    val_idcs = parse_val_groundtruth(root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    return wnid_to_classes, val_wnids


def parse_meta(devkit_root, path="data", filename="meta.mat"):
    import scipy.io as sio

    metafile = os.path.join(devkit_root, path, filename)
    meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    meta = [
        meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0
    ]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(", ")) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes


def parse_val_groundtruth(
    devkit_root, path="data", filename="ILSVRC2012_validation_ground_truth.txt"
):
    with open(os.path.join(devkit_root, path, filename), "r") as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


def prepare_train_folder(folder):
    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_tar(archive, os.path.splitext(archive)[0], delete=True, ignore=0.25)


def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))


def _splitexts(root):
    exts = []
    ext = "."
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, "".join(reversed(exts))
