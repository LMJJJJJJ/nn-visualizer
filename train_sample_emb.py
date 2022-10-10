import os
import os.path as osp
import argparse
import numpy as np
import torch
from embeddings import SampleEmbedding
from tools.lib import set_seed, makedirs, load_raw_data, get_target_probs
set_seed(2020)
from configs import PATHS, CLASS_NAMES

parser = argparse.ArgumentParser("train sample embeddings")
parser.add_argument("--dataset", default="TinyImagenet")
parser.add_argument("--arch", default="vgg16")
parser.add_argument("--gpu-id", type=int, default=1)
parser.add_argument("--save-dir", default="./saved-results/sample-embedding")
# hparams in training the sample embeddings
parser.add_argument("--temperature", type=float, default=4.0)
parser.add_argument("--output-dim", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--n-step", type=int, default=2000)
parser.add_argument("--s-controller", type=float, default=40.0)
args = parser.parse_args()

args.save_dir = osp.join(args.save_dir, f"{args.dataset}_{args.arch}")
makedirs(args.save_dir)
device = torch.device("cpu" if args.gpu_id == "cpu" else f"cuda:{args.gpu_id}")

# =====================================
#      First, load the raw data
# =====================================
class_names = CLASS_NAMES[args.dataset]
features, logits, labels = load_raw_data(
    PATHS=PATHS, dataset=args.dataset, arch=args.arch,
    data_names=["sample_features", "logits", "labels"],
    device=device
)
probs = get_target_probs(logits, args.temperature)

# =====================================
#   Next, project raw sample features
#     into a low dimensional space.
# =====================================
in_dim = features.shape[1]
out_dim = args.output_dim

sample_emb = SampleEmbedding(in_dim=in_dim, out_dim=out_dim, controller=args.s_controller, device=device)
sample_emb.fit(
    data={"sample_features": features, "target_probs": probs},
    params={"lr": args.lr, "momentum": 0.9, "wd": 5e-4, "n_step": args.n_step},
    verbose_dir=args.save_dir
)

# =====================================
#    Visualize and save the results
# =====================================
sample_emb.visualize(
    features, probs, labels,
    class_names=class_names,
    save_path=osp.join(args.save_dir, f"scatter.png")
)
embs = sample_emb(features).detach().cpu().numpy()
np.save(osp.join(args.save_dir, f"emb.npy"), embs)
class_directions = sample_emb.mu(features, probs).detach().cpu().numpy()
np.save(osp.join(args.save_dir, f"mu.npy"), class_directions)
torch.save(sample_emb.cpu().state_dict(), osp.join(args.save_dir, f"g.pth"))