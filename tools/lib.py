import os
import os.path as osp
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import torch.backends.cudnn
from collections import OrderedDict
import imageio
import sys
sys.path.append("..")
from embeddings.sample_embedding import load_sample_embedding
from embeddings.importance_weight import load_importance_weight
from embeddings.region_embedding import load_region_embedding


def set_seed(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_raw_data(PATHS, dataset, arch, data_names, device):
    data_list = []
    for data_name in data_names:
        path = PATHS[dataset][arch][data_name]
        if "label" in data_name:
            item = torch.from_numpy(np.load(path)).long()
        else:
            item = torch.from_numpy(np.load(path)).float()
        item = item.to(device)
        data_list.append(item)
    return data_list


def load_raw_data_adv(ADV_PATHS, dataset, arch, adv_type, adv_iter, data_names, device):
    data_list = []
    for data_name in data_names:
        path = ADV_PATHS[dataset][arch][data_name][adv_type][adv_iter]
        if "label" in data_name:
            item = torch.from_numpy(np.load(path)).long()
        else:
            item = torch.from_numpy(np.load(path)).float()
        item = item.to(device)
        data_list.append(item)
    return data_list


def get_target_probs(logits, temperature):
    probs = torch.exp(logits / temperature)
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs


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


class MovingAverageMeter(object):
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = self.alpha * self.avg + (1 - self.alpha) * val

class WeightBuffer(object):
    def __init__(self, max_elements: int):
        assert max_elements > 0
        self.max_elements = max_elements
        self.buffer = []

    def enqueue(self, element):
        if len(self.buffer) < self.max_elements:
            self.buffer.append(element)
        else:
            self.buffer = self.buffer[1:] + [element]

    def get_mean(self):
        if len(self.buffer) == 0:
            return None
        if len(self.buffer) == 1:
            return self.buffer[0]
        state_dict = OrderedDict()
        for param_name in self.buffer[0].keys():
            param = [self.buffer[i][param_name] for i in range(len(self.buffer))]
            param = torch.stack(param, dim=0).mean(dim=0)
            state_dict[param_name] = param
        return state_dict




def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # for ix, param_group in enumerate(optimizer.param_groups):
    #     param_group['lr'] = lr[0]
    return


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def create_gif(image_list, gif_name, duration=0.1, verbose=0):
    frames = []
    for image_name in image_list:
        if verbose == 1:
            print(image_name)
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def get_all_results(result_folder, dataset, arch, feature_layer):
    sample_emb, sample_trans, class_direction = load_sample_embedding(
        osp.join(result_folder, f"sample-embedding/{dataset}_{arch}")
    )
    W_region, W_kernel = load_importance_weight(
        osp.join(result_folder, f"importance-weight/{dataset}_{arch}/{feature_layer}")
    )
    region_emb, region_trans = load_region_embedding(
        osp.join(result_folder, f"region-embedding/{dataset}_{arch}/{feature_layer}")
    )
    return sample_emb, sample_trans, class_direction, W_region, W_kernel, region_emb, region_trans