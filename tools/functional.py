import torch
from scipy.interpolate import interp1d
from .lib import load_obj

def softmax_class(x_query, x_proto, weights=None):
    '''
    Given the embeddings of each sample and the mean embedding of each class, claculate the softmax result
    :param x_query: torch.Size([n_sample, output_dim]) n_query=n_sample, output_dim=d
    :param x_proto: torch.Size([n_class, output_dim])
    :param weights: torch.Size([n_class])
    :return: torch.Size([n_sample, n_class])
    '''
    n_query = x_query.size(0)
    n_class = x_proto.size(0)
    d = x_query.size(1)
    assert d == x_proto.size(1)
    # compute query distribution over class
    # TODO: check whether this is consistent with the paper (NO square root?) [OK!]
    y = torch.pow(x_proto.unsqueeze(0).expand(n_query, n_class, d) - x_query.unsqueeze(1).expand(n_query, n_class, d), 2).sum(2).squeeze()
    y = torch.exp(-y) # torch.Size([n_sample, n_class])

    if weights is not None:
        # apply class weights
        y = y * weights.unsqueeze(0).expand_as(y)

    y = y / y.sum(1, keepdim=True).expand_as(y)

    return y


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


def frobenius_distance(y_target, y_pred):
    '''
    calculate the frobenius distance (squared) of two matrices A and B
    :param y_target: torch.Size([n_sample, n_class])
    :param y_pred: torch.Size([n_sample, n_class])
    :return: torch.Size([]) scalar
    '''
    return torch.pow((y_target - y_pred), 2).sum()


if __name__ == '__main__':
    func = get_kappa_function("../DRPR/kappa/kappa_max=10.0_dim=3.bin")
    print(func(torch.randn(5, 3).norm(p=2, dim=1)))