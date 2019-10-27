from PIL import Image

import os
import os.path
import sys
from torch.utils.data.sampler import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as img_file:
        with Image.open(img_file) as cur_img:
            img = cur_img.convert("RGB")
            cur_img.close()
        img_file.close()

    return img


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        print("Use accimage")
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImagenetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, data_source, repeat_chunk=1):
        self.data_source = data_source
        self.indices = data_source.current_indices
        self.repeat = repeat_chunk

    def __iter__(self):
        rpt = 0
        while True:
            shuffled_indices = [
                self.indices[i] for i in torch.randperm(len(self.indices))
            ]
            cur_len = len(self.indices)
            for i in shuffled_indices:
                yield i
            rpt += 1
            if rpt == self.repeat:
                rpt = 0
                self.data_source.load_next_chunk()
                self.indices = self.data_source.current_indices

    def __len__(self):
        return len(self.data_source)
