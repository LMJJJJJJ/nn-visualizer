import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
import os.path as osp
from .utils import kl_div, get_kappa_function, AverageMeter, load_obj, save_obj
from .plot import plot_curves
import random
from tqdm import tqdm


EPS = 1e-8


def _init_r_prime_lookup(regional_features, w_k, cached_dir):
    cached_path = osp.join(cached_dir, "r_prime_lookup.list")
    if osp.exists(cached_path):
        return load_obj(cached_path)
    N, K, H, W = regional_features.shape
    R = H * W
    regional_features = regional_features.view(N, K, R)
    regional_features /= (regional_features.norm(2, dim=1, keepdim=True).detach() + 1e-8)
    r_prime_lookup = []  # r_prime_lookup[i][j] describes how j is aligned to i, i.e. f[i] and f[j][:, r_prime_lookup[i][j]] are aligned
    for i in tqdm(range(N), desc=f"[RegionEmbedding] Initializing r'-matrix"):
        r_prime_lookup_i = []
        w_k_i = w_k[i]  # torch.Size([K])
        for j in range(N):
            f_i = regional_features[i]
            f_j = regional_features[j]
            f_i = f_i.view(K, R, 1).expand(K, R, R)
            f_j = f_j.view(K, 1, R).expand(K, R, R)
            dist_ij = (f_i * f_j).detach()  # torch.Size([K, R_i, R_j])
            r_mat_j = torch.sum(dist_ij * w_k_i.view(K, 1, 1).detach(), dim=0)  # torch.Size([R, R])
            r_mat_j, r_prime = torch.max(r_mat_j, dim=1)
            r_prime_lookup_i.append(r_prime)
        r_prime_lookup.append(torch.stack(r_prime_lookup_i, dim=0))
    r_prime_lookup = torch.stack(r_prime_lookup, dim=0)
    save_obj(r_prime_lookup, cached_path)
    return r_prime_lookup


def _init_target_sim_mat(target_probs):
    # This is a constant distribution throughout training
    target_probs /= target_probs.norm(2, dim=1, keepdim=True).detach()
    target_sim_mat = torch.matmul(target_probs, target_probs.transpose(0, 1))
    return target_sim_mat


def _get_target_sim_i(target_sim_mat, i, j_list, kappa):
    target_sim_i = target_sim_mat[i][j_list]
    target_sim_i = torch.exp(kappa * target_sim_i)
    return target_sim_i / torch.sum(target_sim_i).detach()


def _get_estimated_sim_i(region_embs, i, j_list, kappa_func, w_r, r_prime_lookup):
    N, R, dim = region_embs.shape
    device = region_embs.device
    w_r_i = w_r[i].to(device)
    q_i_r = torch.zeros(len(j_list), R).to(device)
    for j_id, j in enumerate(j_list):
        r_prime = r_prime_lookup[i][j]
        f_i = region_embs[i]  # torch.Size([R, dim])
        f_j = region_embs[j][r_prime]  # torch.Size([R, dim])
        f_i_ = f_i / (f_i.norm(p=2, dim=1, keepdim=True).detach() + EPS)
        f_j_ = f_j / (f_j.norm(p=2, dim=1, keepdim=True).detach() + EPS)
        sim = torch.sum(f_i_ * f_j_, dim=1)  # torch.Size([R])
        kappa = torch.from_numpy(kappa_func(f_i.norm(p=2, dim=1).detach().cpu())).float().to(device)
        q_i_r[j_id] = torch.exp(kappa * sim)  # torch.Size([R])
    q_i_r = q_i_r / (torch.sum(q_i_r, dim=0) + EPS)  # normalize
    q_i = torch.pow(q_i_r, w_r_i)  # torch.Size([n_samples-1, R]
    q_i = q_i.prod(dim=1)  # torch.Size([n_samples-1])
    return q_i / torch.sum(q_i)


def _parse_data(data, device):
    target_probs = data["target_probs"].to(device)
    regional_features = data["regional_features"].to(device)
    sample_embs = data["sample_embs"].to(device)
    w_k = data["w_k"].to(device)
    w_r = data["w_r"].to(device)
    return target_probs, regional_features, sample_embs, w_k, w_r


def _parse_params(params):
    lambda_kl = params["lambda_kl"]
    lambda_mi = params["lambda_mi"]
    kl_sample_num = params["kl_sample_num"]
    lr = params["lr"]
    momentum = params["momentum"]
    wd = params["wd"]
    n_step = params["n_step"]
    return lambda_kl, lambda_mi, kl_sample_num, lr, momentum, wd, n_step


class RegionEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, transform=None, controller=100.0, kappa_m=1000.0, kappa_p=5.0, device="cuda"):
        super(RegionEmbedding, self).__init__()
        self.device = device
        self.out_dim = out_dim
        self.controller = nn.Parameter(torch.FloatTensor([controller]).squeeze().to(device), requires_grad=False)
        self.kappa_m = nn.Parameter(torch.FloatTensor([kappa_m]).squeeze().to(device), requires_grad=False)
        self.kappa_func = get_kappa_function(f".cache/region/kappa_max={self.kappa_m}_dim={out_dim}.bin")
        self.kappa_p = nn.Parameter(torch.FloatTensor([kappa_p]).squeeze().to(device), requires_grad=False)

        if transform is not None:
            self.transform = transform
        else:
            # use below to calculate the embedding for each region
            # (given 1. the region feature/transformation; 2. the importance matrix)
            self.transform = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
            self.transform.to(device)
            print("[RegionEmbedding] Finish initializing the transform.")

    def forward(self, regional_features):
        return self.transform(regional_features / self.controller)

    def loss_1(self, target_sim_mat, region_embs, w_r, r_prime_lookup, i, kl_sample_num):
        N = region_embs.shape[0]
        population = torch.arange(N)[torch.arange(N) != i].tolist()
        if kl_sample_num > 0:
            j_list = random.sample(population, kl_sample_num)
        else:
            j_list = population
        return kl_div(
            self._target_sim_i(target_sim_mat=target_sim_mat, i=i, j_list=j_list, kappa=self.kappa_p),
            self._estimated_sim_i(region_embs=region_embs, i=i, j_list=j_list, kappa_func=self.kappa_func,
                                  w_r=w_r, r_prime_lookup=r_prime_lookup)
        )

    def loss_2(self, sample_embs, region_embs, w_r, i):
        return torch.dot(
            w_r[i],
            -torch.cosine_similarity(region_embs[i],
                                     sample_embs[i].reshape(-1, self.out_dim), dim=1)
        )

    def _target_sim_i(self, target_sim_mat, i, j_list, kappa):
        return _get_target_sim_i(
            target_sim_mat=target_sim_mat,
            i=i, j_list=j_list,
            kappa=kappa
        )

    def _estimated_sim_i(self, region_embs, i, j_list, kappa_func, w_r, r_prime_lookup):
        return _get_estimated_sim_i(
            region_embs=region_embs, i=i, j_list=j_list,
            kappa_func=kappa_func, w_r=w_r,
            r_prime_lookup=r_prime_lookup
        )

    def get_emb(self, raw_regional_features):
        N = raw_regional_features.shape[0]
        K = raw_regional_features.shape[1]
        R = raw_regional_features.shape[2] * raw_regional_features.shape[3]
        regional_features = raw_regional_features.view(N, K, R)
        regional_features = regional_features.transpose(1, 2)
        with torch.no_grad():
            return self.forward(regional_features)

    def fit(self, data, params, verbose_dir=None):
        """
        Train the regional embedding
        :param data: dict,

                       - regional_features: with shape [N, K, H, W], K is the number of channels,
                                            HW are the height/width of the feature map
                       - target_probs: with shape [N, C], C is the # of categories
                       - sample_embs: with shape [N, d'(=3)], the trained sample embeddings
                       - w_k: with shape [N, K], the importance weight $v^{(k)}$ of kernels
                       - w_r: with shape [N, R=HW], the importance weight $w^{(r)}$ of regions
        :param params:
        :param verbose_dir:
        :return:
        """
        target_probs, regional_features, sample_embs, w_k, w_r = _parse_data(data, self.device)
        lambda_kl, lambda_mi, kl_sample_num, lr, momentum, wd, n_step = _parse_params(params)

        assert target_probs.shape[0] == regional_features.shape[0] == sample_embs.shape[0] == w_k.shape[0] == w_k.shape[0]
        r_prime_lookup = _init_r_prime_lookup(regional_features, w_k, verbose_dir)
        target_sim_mat = _init_target_sim_mat(target_probs)
        N = regional_features.shape[0]
        K = regional_features.shape[1]
        R = regional_features.shape[2] * regional_features.shape[3]
        regional_features = regional_features.view(N, K, R)
        regional_features = regional_features.transpose(1, 2)

        print("---------- parameters in [RegionEmbedding] ----------")
        print("target_probs shape:", target_probs.shape)
        print("regional_features shape:", regional_features.shape)
        print("sample_embs shape:", sample_embs.shape)
        print("w_k shape:", w_k.shape)
        print("w_r shape", w_r.shape)
        print("output_dim:", self.out_dim)
        print("device:", self.device)
        print("-----------------------------------------------------")

        optimizer = optim.SGD(self.transform.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        plot_dict = {"loss": [], "loss-1": [], "loss-2": []}

        for step in range(n_step):
            sample_ids = list(range(N))
            random.shuffle(sample_ids)
            pbar = tqdm(sample_ids, mininterval=1, ncols=100)
            pbar.set_description(f"[Epoch {step}/{n_step}]")
            loss_avg, loss_1_avg, loss_2_avg = AverageMeter(), AverageMeter(), AverageMeter()
            for idx, i in enumerate(pbar):

                region_embs = self.forward(regional_features)

                # train
                optimizer.zero_grad()
                loss_1 = self.loss_1(target_sim_mat, region_embs, w_r, r_prime_lookup, i, kl_sample_num)
                loss_2 = self.loss_2(sample_embs, region_embs, w_r, i)
                loss = loss_1 * lambda_kl + loss_2 * lambda_mi
                loss.backward(retain_graph=True)
                optimizer.step()

                if idx % 10 == 0 or idx == N - 1:
                    pbar.set_postfix_str("loss={:.6f}".format(loss_avg.avg))
                loss_avg.update(loss.item())
                loss_1_avg.update(loss_1.item())
                loss_2_avg.update(loss_2.item())

            plot_dict["loss"].append(loss_avg.avg)
            plot_dict["loss-1"].append(loss_1_avg.avg)
            plot_dict["loss-2"].append(loss_2_avg.avg)
            plot_curves(
                plot_dict, verbose_dir, xlabels=["epoch"] * 3,
                ylabels=[r"$loss_{total}$", r"$loss_1=KL(q_i\Vert p_i)$",
                         r"$loss_2=w^{(R)}\cdot cos(f^{(word)}, f^{(sample)})$"]
            )


def load_region_embedding(folder, device=torch.device("cpu")):
    region_emb = np.load(osp.join(folder, "region_emb.npy"))
    region_trans = torch.load(osp.join(folder, "region_transform.pth"), map_location=device)
    return region_emb, region_trans


if __name__ == '__main__':
    pass
