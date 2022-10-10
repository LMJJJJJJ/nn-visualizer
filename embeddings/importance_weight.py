import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
from .utils import kl_div, makedirs
from .plot import plot_curves
from tqdm import tqdm
from torch import optim

EPS = 1e-7


def _parse_data(data, device):
    target_probs = data["target_probs"].to(device)
    regional_features = data["regional_features"].to(device)
    images = data["images"].to(device)
    return target_probs, regional_features, images


def _parse_params(params):
    kl_sample_num = params["kl_sample_num"]
    lr = params["lr"]
    momentum = params["momentum"]
    wd = params["wd"]
    n_step = params["n_step"]
    return kl_sample_num, lr, momentum, wd, n_step


def _init_target_sim_mat(target_probs):
    target_probs /= target_probs.norm(2, dim=1, keepdim=True).detach()
    target_sim_mat = torch.matmul(target_probs, target_probs.transpose(0, 1))
    return target_sim_mat


def _normalize_importance_weight(W_kernel, W_region, p=1):
    W_kernel.weight.data = W_kernel.weight.data.abs().detach()
    W_region.weight.data = W_region.weight.data.abs().detach()
    W_kernel.weight.data /= W_kernel.weight.data.norm(p, dim=1, keepdim=True).detach()
    W_region.weight.data /= W_region.weight.data.norm(p, dim=1, keepdim=True).detach()
    return W_kernel, W_region


def _get_target_sim_i(target_sim_mat, i, j_list, kappa):
    target_sim_i = target_sim_mat[i][j_list]
    target_sim_i = torch.exp(kappa * target_sim_i)
    return target_sim_i / torch.sum(target_sim_i).detach()


def _get_estimated_sim_i(regional_features, W_kernel, W_region, i, j_list, kappa):
    N, K, R = regional_features.shape
    device = regional_features.device

    W_K_i = W_kernel(torch.Tensor([i]).long().to(device)).squeeze().abs()  # W non-negative => abs
    W_R_i = W_region(torch.Tensor([i]).long().to(device)).squeeze().abs()
    P_i_r = torch.zeros(len(j_list), R).to(device)
    for j_id, j in enumerate(j_list):
        f_i = regional_features[i]
        f_j = regional_features[j]
        f_i = f_i.view(K, R, 1).expand(K, R, R)
        f_j = f_j.view(K, 1, R).expand(K, R, R)
        dist_ij = (f_i * f_j).detach()  # torch.Size([K, R, R])
        r_mat_j = torch.sum(dist_ij * W_K_i.view(K, 1, 1).detach(), dim=0)  # torch.Size([R, R]), (Ri, Rj)
        r_mat_j, r_prime = torch.max(r_mat_j, dim=1)

        f_i = regional_features[i]
        f_j = regional_features[j][:, r_prime]
        dist_ij = (f_i * f_j).detach()  # torch.Size([K, R])
        P_i_r[j_id] = torch.exp(kappa * torch.matmul(W_K_i, dist_ij))  # torch.Size([R])
    P_i_r = P_i_r / (torch.sum(P_i_r, dim=0) + EPS)  # normalize
    P_i = torch.pow(P_i_r, W_R_i)  # torch.Size([n_samples-1, R]
    P_i = P_i.prod(dim=1)  # torch.Size([n_samples-1]

    P_i = P_i / (torch.sum(P_i) + EPS)
    return P_i


def visualize_w_r(raw_image, w_r, save_folder, save_name):
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.title(r"raw image $x$")
    plt.axis("off")
    plt.imshow(raw_image.numpy().transpose(1, 2, 0), interpolation=None)
    plt.subplot(1, 2, 2)
    plt.title(r"estimated regional importance $w^{(r)}$")
    plt.imshow(w_r, cmap='Reds', vmin=0.0, vmax=0.01, interpolation=None)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, save_name), dpi=200)
    plt.close("all")


class ImportanceWeight(nn.Module):
    def __init__(self, N, K, H, W, kappa_m=1000.0, kappa_p=5.0, device="cuda"):
        super(ImportanceWeight, self).__init__()

        self.device = device
        R = H * W
        self.N, self.K, self.H, self.W, self.R = N, K, H, W, R
        self.kappa_m = kappa_m
        self.kappa_p = kappa_p

        # initialize region importance weight
        self.W_region = nn.Embedding(N, R, max_norm=1.0, norm_type=1)
        self.W_region.weight.data = torch.ones(N, R) / R
        self.W_region.to(device)
        # initialize kernel importance weight
        self.W_kernel = nn.Embedding(N, K, max_norm=1.0, norm_type=1)
        self.W_kernel.weight.data = torch.ones(N, K) / K
        self.W_kernel.to(device)

    def forward(self, inds):
        return self.W_region(inds), self.W_kernel(inds)

    def normalize(self, p=1):
        self.W_kernel, self.W_region = _normalize_importance_weight(W_kernel=self.W_kernel, W_region=self.W_region, p=p)

    def loss(self, target_sim_mat, regional_features, i, kl_sample_num):
        N = regional_features.shape[0]
        population = torch.arange(N)[torch.arange(N) != i].tolist()
        if kl_sample_num > 0:
            j_list = random.sample(population, kl_sample_num)
        else:
            j_list = population
        return kl_div(
            self._target_sim_i(target_sim_mat=target_sim_mat, i=i, j_list=j_list, kappa=self.kappa_p),
            self._estimated_sim_i(regional_features=regional_features, W_kernel=self.W_kernel,
                                  W_region=self.W_region, i=i, j_list=j_list, kappa=self.kappa_m)
        )

    def _target_sim_i(self, target_sim_mat, i, j_list, kappa):
        return _get_target_sim_i(
            target_sim_mat=target_sim_mat,
            i=i, j_list=j_list,
            kappa=kappa
        )

    def _estimated_sim_i(self, regional_features, W_kernel, W_region, i, j_list, kappa):
        return _get_estimated_sim_i(
            regional_features=regional_features,
            W_kernel=W_kernel, W_region=W_region,
            i=i, j_list=j_list, kappa=kappa
        )

    def estimate(self, data, params, verbose_dir):
        makedirs(osp.join(verbose_dir, "visualization"))
        target_probs, regional_features, images = _parse_data(data, self.device)
        kl_sample_num, lr, momentum, wd, n_step = _parse_params(params)

        assert target_probs.shape[0] == regional_features.shape[0]
        N = regional_features.shape[0]  # the number of samples in total
        K = regional_features.shape[1]  # the number of channels / kernels in total
        R = regional_features.shape[2] * regional_features.shape[3]  # the number of regions in total
        assert N == self.N and K == self.K and R == self.R
        images = images.cpu()
        regional_features = regional_features.view(N, K, R)
        regional_features /= (regional_features.norm(2, dim=1, keepdim=True).detach() + EPS)
        target_sim_mat = _init_target_sim_mat(target_probs)

        print("---------- parameters in [ImportanceWeight] ----------")
        print("target_probs shape:", target_probs.shape)
        print("regional_features shape:", regional_features.shape)
        print("target_sim_mat shape:", target_sim_mat.shape)
        print("device:", self.device)
        print("-----------------------------------------------------")

        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

        for i in range(N):
            pbar = tqdm(range(n_step), mininterval=1, ncols=100, desc=f"Sample {i}")
            plot_dict = {"loss": []}

            for _ in pbar:
                self.normalize()
                optimizer.zero_grad()
                loss = self.loss(target_sim_mat, regional_features, i, kl_sample_num)
                loss.backward()
                optimizer.step()
                pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
                plot_dict["loss"].append(loss.item())
            self.normalize()

            plot_curves(
                plot_dict=plot_dict, save_folder=osp.join(verbose_dir, "visualization"),
                save_name=f"sample_{str(i).zfill(4)}"
            )
            w_r = self.W_region.weight.data[i].reshape(self.W, self.H).detach().cpu().numpy()
            visualize_w_r(
                raw_image=images[i], w_r=w_r,
                save_folder=osp.join(verbose_dir, "visualization"),
                save_name=f"sample_{str(i).zfill(4)}.png"
            )
            torch.save(self.state_dict(), osp.join(verbose_dir, "w.pth"))


def load_importance_weight(folder, device=torch.device("cpu")):
    weights = torch.load(osp.join(folder, "w.pth"), map_location=device)
    W_region = weights["W_region.weight"]
    W_kernel = weights["W_kernel.weight"]
    return W_region, W_kernel


if __name__ == '__main__':
    pass