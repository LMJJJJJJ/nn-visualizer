import os
import os.path as osp
import numpy as np
import torch
import argparse
from embeddings import SampleEmbedding
from tools.lib import WeightBuffer, create_gif, makedirs
from configs import *


parser = argparse.ArgumentParser("train sample embeddings (dynamics)")
parser.add_argument("--dataset", default="TinyImagenet")
parser.add_argument("--arch", default="vgg16")
parser.add_argument("--gpu-id", default=1)
parser.add_argument("--save-dir", default="./saved-results/sample-embedding")
parser.add_argument("--verbose-dir", default="./embeddings/verbose/sample")
args = parser.parse_args()

args.save_dir = osp.join(args.save_dir, args.dataset)
args.verbose_dir = osp.join(args.verbose_dir, args.dataset)
makedirs(osp.join(args.save_dir, "checkpoints"))


class_names = CLASS_NAMES[args.dataset]
features_ckpts = PATHS[args.dataset][args.arch]["sample_features_ckpts"]
logits_ckpts = PATHS[args.dataset][args.arch]["logits_ckpts"]
assert len(features_ckpts) == len(logits_ckpts)
labels = PATHS[args.dataset][args.arch]["labels"]
labels = torch.from_numpy(np.load(labels)).long()

temperature = 4.0

buffer = WeightBuffer(max_elements=3)

vis_save_paths = []

for features, logits in zip(features_ckpts, logits_ckpts):
    iteration = features.split(".")[0].split("_")[-1]
    print("="*10 + f" iteration {iteration} " + "="*10)
    features = torch.from_numpy(np.load(features)).float()
    logits = torch.from_numpy(np.load(logits)).float()
    features = features.to("cuda")
    logits = logits.to("cuda")

    probs = torch.exp(logits / temperature)
    probs = probs / torch.sum(probs, dim=1, keepdim=True)

    in_dim = features.shape[1]
    out_dim = 3

    sample_emb = SampleEmbedding(in_dim=in_dim, out_dim=out_dim)
    state_dict = buffer.get_mean()
    if state_dict is not None:
        sample_emb.load_state_dict(state_dict)
        align = sample_emb(features)
    else:
        align = None

    sample_emb.fit(
        features, probs, align=align,
        lr=0.0001, verbose=1,
        verbose_dir=osp.join(osp.join(args.verbose_dir, "checkpoints", f"iter_{iteration}"))
    )

    vis_save_path = osp.join(args.save_dir, "checkpoints", f"{args.arch}_{iteration}.png")
    sample_emb.visualize(
        features, None, labels,
        class_names=class_names,
        save_path=vis_save_path
    )
    vis_save_paths.append(vis_save_path)

    buffer.enqueue(sample_emb.state_dict())

    class_directions = sample_emb.mu(features, probs).detach().cpu().numpy()
    embs = sample_emb(features).detach().cpu().numpy()
    np.save(osp.join(args.save_dir, "checkpoints", f"{args.arch}_mu_{iteration}.npy"), class_directions)
    np.save(osp.join(args.save_dir, "checkpoints", f"{args.arch}_emb_{iteration}.npy"), embs)

    torch.save(sample_emb.cpu().state_dict(), osp.join(args.save_dir, "checkpoints", f"{args.arch}_g_{iteration}.pth"))

create_gif(vis_save_paths, osp.join(args.save_dir, f"{args.arch}_dynamics.gif"))
