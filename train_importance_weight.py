import os
import os.path as osp
import torch
torch.autograd.set_detect_anomaly(True)
import argparse
from embeddings import ImportanceWeight
from tools.lib import makedirs, set_seed, get_target_probs, load_raw_data
set_seed(2020)
from configs import PATHS, CLASS_NAMES


parser = argparse.ArgumentParser("Importance Weight train")
parser.add_argument("--dataset", type=str, default="TinyImagenet")
parser.add_argument("--arch", type=str, default="vgg16")
parser.add_argument("--feature-layer", type=str, default="conv_53")
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--save-dir", default="./saved-results/importance-weight")
# hparams in training the importance weights
parser.add_argument("--temperature", type=float, default=4.0)
parser.add_argument("--kappa-m", type=float, default=1000.0)
parser.add_argument("--kappa-p", type=float, default=5.0)
parser.add_argument("--kl-sample-num", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--n-step", type=int, default=20)
args = parser.parse_args()

args.save_dir = osp.join(args.save_dir, f"{args.dataset}_{args.arch}", f"{args.feature_layer}")
makedirs(args.save_dir)
device = torch.device("cpu" if args.gpu_id == "cpu" else f"cuda:{args.gpu_id}")

# =====================================
#      First, load the source data
# =====================================
class_names = CLASS_NAMES[args.dataset]
images, labels, regional_features, logits = load_raw_data(
    PATHS=PATHS, dataset=args.dataset, arch=args.arch, device=device,
    data_names=["images", "labels", f"{args.feature_layer}_features", "logits"]
)
probs = get_target_probs(logits, args.temperature)

# =======================================
#   Next, estimate regional importance
# =======================================
importance_weight = ImportanceWeight(*regional_features.shape, kappa_m=args.kappa_m, kappa_p=args.kappa_p, device=device)
importance_weight.estimate(
    data={"target_probs": probs, "regional_features": regional_features, "images": images},
    params={"kl_sample_num": args.kl_sample_num, "lr": args.lr, "momentum": 0.9, "wd": 5e-4, "n_step": args.n_step},
    verbose_dir=args.save_dir,
)
torch.save(importance_weight.cpu().state_dict(), osp.join(args.save_dir, "w.pth"))
