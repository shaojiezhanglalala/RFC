import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset


def _tolist(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    else:
        raise TypeError


class Visda17(Dataset):
    def __init__(self, info_path=None, transform=None, return_idx=False):
        lines = open(info_path).readlines()
        samples = []
        for line in lines:
            tokens = line.split()
            samples.append((tokens[0], int(tokens[1])))
        assert len(samples) > 0

        self.data = samples
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        path, target = self.data[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.return_idx:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class SubVisda17(Dataset):
    def __init__(self, info_path=None, transform=None, mode=None,
                 indices=None, probs=None, psl=None, return_idx=False):
        assert mode in ['warmup', 'eval_train', 'label', 'unlabel', 'test', 'select']

        sample_list = open(info_path).readlines()
        self.paths = []
        self.targets = []
        if indices is not None:
            sample_list = [sample_list[i] for i in indices]
        for i, line in enumerate(sample_list):
            tokens = line.split()
            self.paths.append(tokens[0])
            self.targets.append(int(tokens[1]))
        if psl is not None:
            self.targets = _tolist(psl)
        assert len(self.paths) > 0 and len(self.paths) == len(self.targets)

        self.transform = transform
        self.mode = mode
        self.indices = indices
        self.probs = probs  # GMM probs;
        self.return_idx = return_idx

    def __getitem__(self, index):
        if self.mode == 'warmup' or self.mode == 'eval_train' or self.mode == 'test':
            path, target = self.paths[index], self.targets[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            if self.return_idx:
                return img, target, index
            else:
                return img, target
        elif self.mode == 'label':
            path, target, prob = self.paths[index], self.targets[index], self.probs[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            if self.return_idx:
                return img, target, index
            else:
                return img, target, prob
        elif self.mode == 'unlabel':
            path, target = self.paths[index], self.targets[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            if self.return_idx:
                return img, target, index
            else:
                return img
        elif self.mode == 'select':
            path, target, sam_indice = self.paths[index], self.targets[index], self.indices[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            if self.return_idx:
                return img, target, sam_indice
            else:
                return img, target
        else:
            raise ValueError

    def __len__(self):
        return len(self.paths)

