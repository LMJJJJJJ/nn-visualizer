import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from .utils import makedirs, get_kappa_function, softmax_class_vmf, kl_div
from .plot import plot_sample_emb_3d, plot_curves

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime


EPS = 1e-8

class SampleEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, transformation=None, controller=40.0, kappa_m=10.0, device="cuda"):
        super(SampleEmbedding, self).__init__()

        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.controller = nn.Parameter(torch.FloatTensor([controller]).squeeze().to(device), requires_grad=False)
        self.kappa_m = nn.Parameter(torch.FloatTensor([kappa_m]).squeeze().to(device), requires_grad=False)

        # initialize the transformation matrix
        if transformation is not None:
            self.transformation = transformation.to(device)
        else:
            self.transformation = nn.Linear(in_dim, out_dim, bias=False).to(device)
            init.normal_(self.transformation.weight, mean=0.0, std=0.1)

        # initialize the kappa function
        cache_dir = ".cache/sample"
        makedirs(cache_dir)
        kappa_dir = osp.join(cache_dir, f"kappa_max={kappa_m}_dim={out_dim}.bin")
        if osp.exists(kappa_dir):
            self.kappa_func = get_kappa_function(kappa_dir)
        else:
            raise NotImplementedError

        self.datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def forward(self, features):
        return self.transformation(features / self.controller)

    def mu(self, features, target_probs):
        emb = self.forward(features)
        emb_norm = emb.norm(p=2, dim=1, keepdim=True)
        emb = emb / (emb_norm.detach() + EPS)
        kappa = self.kappa(features)
        kappa = kappa.unsqueeze(1).expand_as(target_probs)
        mu = torch.mm(target_probs.t() * kappa.t(), emb)
        return mu / mu.norm(p=2, dim=1, keepdim=True)

    def pi(self, target_probs):
        return target_probs.sum(0).squeeze() / target_probs.shape[0]

    def kappa(self, features):
        emb = self.forward(features).detach().cpu().numpy()
        emb_norm = np.linalg.norm(emb, ord=2, axis=1)
        assert emb_norm.max() <= self.kappa_func.x.max(), f"current z_norm max: {emb_norm.max().item()}"
        return torch.from_numpy(self.kappa_func(emb_norm)).float().to(self.device)

    def loss(self, features, target_probs):
        emb = self.forward(features)
        emb_norm = emb.norm(p=2, dim=1, keepdim=True)
        emb = emb / (emb_norm.detach() + EPS)
        estimated_probs = softmax_class_vmf(emb, self.mu(features, target_probs), self.kappa(features), weights=self.pi(target_probs))
        return kl_div(target_probs, estimated_probs)

    def rotate(self, cur_emb, prev_emb, t=0.0, metric="l2"):
        '''
        performs the rotation. the rotation matrix is obtained using ICP
        **[NOTE]** only linear transformation can apply this trick
        :param align: the target sample embeddings to be aligned
        :param t: the dropped t-proportion outliers, if not specified, don't drop
        :param metric: 'l2' or 'cos'
        :return: (None) but rotates the current transformation by an angle
        '''
        if metric == 'l2':
            prev_data = prev_emb.data
            cur_data = cur_emb.data
        elif metric == 'cos':
            prev_data = prev_emb.data / prev_emb.data.norm(p=2, dim=1, keepdim=True)
            cur_data = cur_emb.data / cur_emb.data.norm(p=2, dim=1, keepdim=True)
        else:
            raise Exception(f"metric {metric} not supported!")

        if t > 0:
            # to drop t-proportion outliers
            distance = torch.norm(prev_data - cur_data, p=2, dim=1)
            mask = torch.argsort(distance, descending=True)[int(t*cur_data.shape[0]):]
            prev_data = prev_data[mask]
            cur_data = cur_data[mask]

        # the rotation matrix is solved using ICP
        u, _, v = torch.svd(torch.matmul(cur_data.t(), prev_data.data))
        rot_mat = torch.matmul(u, v.t())

        self.transformation.weight.data = torch.matmul(rot_mat.data.t(), self.transformation.weight.data)

    def fit(self, features, target_probs, align=None, lr=0.001, momentum=0.9, wd=5e-4, n_step=2000, verbose=0, verbose_dir=None):
        features, target_probs = features.to(self.device), target_probs.to(self.device)
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

        if verbose == 0:
            pbar = range(n_step)
        elif verbose == 1:
            pbar = tqdm(range(n_step), mininterval=1, ncols=100)
            plot_dict = {"loss": []}
        else:
            raise NotImplementedError

        for iteration in pbar:
            optimizer.zero_grad()
            loss = self.loss(features, target_probs)
            loss.backward()
            optimizer.step()
            if verbose == 1:
                pbar.set_postfix_str("loss={:0.6f}".format(loss.item()))
                plot_dict["loss"].append(loss.item())

        if align is not None:
            cur_emb = self.forward(features)
            self.rotate(cur_emb=cur_emb, prev_emb=align)
            self.rotate(cur_emb=cur_emb, prev_emb=align, t=0.1, metric="cos")

        if verbose == 1:
            if verbose_dir is None:
                verbose_dir = f"verbose/sample/{self.datetime}"
            makedirs(verbose_dir)
            plot_curves(plot_dict=plot_dict, save_folder=verbose_dir)

    def visualize(self, features, target_probs, labels, class_names, colors=None, save_path=None):
        features = features.to(self.device)
        emb = self.forward(features)
        if target_probs is not None:
            target_probs = target_probs.to(self.device)
            mu = self.mu(features, target_probs)
        else:
            mu = None  # do not plot the class directions

        if self.out_dim == 3:
            plot_sample_emb_3d(
                embeddings=emb, labels=labels,
                class_directions=mu,
                class_names=class_names, colors=colors,
                save_path=save_path
            )
        else:
            raise NotImplementedError


if __name__ == '__main__':
    pass
