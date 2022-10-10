import os
import os.path as osp
import torch
import argparse
from embeddings import RegionEmbedding
import numpy as np
from configs import CLASS_NAMES, PATHS
from tools.lib import makedirs, set_seed, get_target_probs, load_raw_data
set_seed(2020)

parser = argparse.ArgumentParser("train embedding")
parser.add_argument("--dataset", default="TinyImagenet", help="CIFAR10 | TinyImagenet")
parser.add_argument("--arch", type=str, default="vgg16")
parser.add_argument("--feature-layer", type=str, default="conv_53")
parser.add_argument("--gpu-id", default="2")
parser.add_argument("--save-dir", default="./saved-results/region-embedding")
parser.add_argument("--sample-emb-dir", default="./saved-results/sample-embedding")
parser.add_argument("--importance-dir", default="./saved-results/importance-weight")
# hparams in training the regional embeddings
parser.add_argument("--output-dim", type=int, default=3)
parser.add_argument("--temperature", type=float, default=4.0)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--n-step", type=int, default=20)
parser.add_argument("--lambda-kl", type=float, default=1.0)
parser.add_argument("--lambda-mi", type=float, default=0.1)
parser.add_argument("--kl-sample-num", type=int, default=100)
parser.add_argument("--kappa-m", type=float, default=1000.0)
parser.add_argument("--kappa-p", type=float, default=5.0)
parser.add_argument("--r-controller", type=float, default=40.0)
args = parser.parse_args()


args.save_dir = osp.join(args.save_dir, f"{args.dataset}_{args.arch}", args.feature_layer)
makedirs(args.save_dir)
args.sample_emb_dir = osp.join(args.sample_emb_dir, f"{args.dataset}_{args.arch}")
args.importance_dir = osp.join(args.importance_dir, f"{args.dataset}_{args.arch}", args.feature_layer)
device = torch.device("cpu" if args.gpu_id == "cpu" else f"cuda:{args.gpu_id}")

class_names = CLASS_NAMES[args.dataset]
regional_features, logits = load_raw_data(
    PATHS=PATHS, dataset=args.dataset, arch=args.arch, device=device,
    data_names=[f"{args.feature_layer}_features", "logits"]
)
probs = get_target_probs(logits, args.temperature)

weights = torch.load(osp.join(args.importance_dir, "w.pth"))
region_weights = weights['W_region.weight'].abs()
kernel_weights = weights['W_kernel.weight'].abs()
sample_emb = torch.from_numpy(np.load(osp.join(args.sample_emb_dir, "emb.npy"))).float()
sample_emb = sample_emb.to(device)

N, K, H, W = regional_features.shape

# ############### set up the modules ################
region_embedding = RegionEmbedding(
    in_dim=K, out_dim=args.output_dim, device=device,
    kappa_m=args.kappa_m, kappa_p=args.kappa_p,
    controller=args.r_controller,
)
region_embedding.fit(
    data={"target_probs": probs, "regional_features": regional_features,
          "sample_embs": sample_emb, "w_k": kernel_weights, "w_r": region_weights},
    params={"lambda_kl": args.lambda_kl, "lambda_mi": args.lambda_mi,
            "kl_sample_num": args.kl_sample_num, "lr": args.lr,
            "momentum": 0.9, "wd": 5e-4, "n_step": args.n_step},
    verbose_dir=args.save_dir
)

region_emb = region_embedding.get_emb(regional_features).cpu().numpy()
np.save(osp.join(args.save_dir, f"region_emb.npy"), region_emb)
torch.save(region_embedding.cpu().state_dict(), osp.join(args.save_dir, f"region_transform.pth"))
