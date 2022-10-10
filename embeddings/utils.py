import os
import os.path as osp
import pickle
from scipy.interpolate import interp1d
import numpy as np
import torch


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


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


def get_kappa_function(path):
    X, Y = load_obj(path)
    return interp1d(X, Y, kind='linear')


def softmax_class_vmf(x_query, x_proto, x_kappa, weights=None, eps=1e-8):
    '''
    Given the embeddings of each sample and the mean embedding DIRECTION of each class, claculate the softmax result
    :param x_query: torch.Size([n_sample, output_dim]) n_query=n_sample, output_dim=d
    :param x_proto: torch.Size([n_class, output_dim])
    :param x_kappa: kappa list of all samples, torch.Size([n_sample])
    :param weights: torch.Size([n_class])
    :return: torch.Size([n_sample, n_class])
    '''
    n_query = x_query.size(0)  # n_query is n_sample
    n_class = x_proto.size(0)
    d = x_query.size(1)
    assert d == x_proto.size(1)

    # compute query distribution over class
    y = (x_proto.unsqueeze(0).expand(n_query, n_class, d) * x_query.unsqueeze(1).expand(n_query, n_class, d)).sum(2)
    y = y * x_kappa.unsqueeze(1).expand_as(y)
    y = torch.exp(y)  # torch.Size([n_sample, n_class])

    if weights is not None:
        # apply class weights
        y = y * weights.unsqueeze(0).expand_as(y)
    y = y / (y.sum(1, keepdim=True).expand_as(y) + eps)

    return y


def kl_div(y_target, y_pred, eps=1e-8):
    '''
    calculate the KL divergence given two distributions
    :param y_target: torch.Size([n_sample, n_class])
    :param y_pred: torch.Size([n_sample, n_class])
    :param eps: epsilon, default=1e-8
    :return: torch.Size([]) scalar
    '''
    # return (- y_target * torch.log(y_pred + eps)).sum() # use cross entropy
    return (- y_target * torch.log(y_pred + eps) + y_target * torch.log(y_target + eps)).sum()


def is_concept_single(region_emb, category, class_direction, kappa_fn, threshold=0.4):
    emb = torch.from_numpy(region_emb).unsqueeze(0)
    l2 = np.linalg.norm(emb, axis=1)
    kappa = torch.from_numpy(kappa_fn(l2)).float()
    region_prob = softmax_class_vmf(emb / l2, class_direction, kappa)[0, category]
    return region_prob >= threshold


def get_region_cos_mat(region_embs, class_direction):
    normalized_region_emb = region_embs / np.linalg.norm(region_embs, axis=2, keepdims=True)
    region_cos_mat = np.matmul(normalized_region_emb, class_direction.numpy().T)
    return region_cos_mat


def get_region_prob_mat(region_embs, class_direction, kappa_fn):
    region_prob = []
    for sample_idx in range(region_embs.shape[0]):
        region_emb_sample = torch.from_numpy(region_embs[sample_idx])
        l2 = np.linalg.norm(region_emb_sample, axis=1)
        kappa = torch.from_numpy(kappa_fn(l2))
        region_prob_sample = softmax_class_vmf(region_emb_sample, class_direction, kappa)
        region_prob.append(region_prob_sample.clone())
    region_prob = torch.stack(region_prob).clone().numpy()
    return region_prob