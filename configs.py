import os
import os.path as osp


_feature_root = lambda dataset: f"/data1/limingjie/Visualization/transform/{dataset}"
_feature_ckpts_root = lambda dataset: f"/data1/limingjie/Visualization/transform/{dataset}/checkpoints"
_adv_feature_root = lambda dataset: f"/data1/limingjie/Visualization/transform-adversarial/{dataset}"


PATHS = {
    "TinyImagenet": {
        "vgg16": {
            "images": osp.join(_feature_root("TinyImagenet"), "images_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "sample_features": osp.join(_feature_root("TinyImagenet"), "final_features_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "labels": osp.join(_feature_root("TinyImagenet"), "labels_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "logits": osp.join(_feature_root("TinyImagenet"), "logits_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "sample_features_ckpts": sorted([
                osp.join(_feature_ckpts_root("TinyImagenet"), file)
                for file in os.listdir(_feature_ckpts_root("TinyImagenet"))
                if file.startswith("final_features_50_vgg16_bn_-4_-5_32_200_2020")
            ]),
            "logits_ckpts": sorted([
                osp.join(_feature_ckpts_root("TinyImagenet"), file)
                for file in os.listdir(_feature_ckpts_root("TinyImagenet"))
                if file.startswith("logits_50_vgg16_bn_-4_-5_32_200_2020")
            ]),
            # regional features
            "conv_53_features": osp.join(_feature_root("TinyImagenet"), "features_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "conv_43_features": osp.join(_feature_root("TinyImagenet"), "conv_43_features_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "conv_33_features": osp.join(_feature_root("TinyImagenet"), "conv_33_features_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "conv_22_features": osp.join(_feature_root("TinyImagenet"), "conv_22_features_50_vgg16_bn_-4_-5_32_200_2020.npy"),
            "conv_21_features": osp.join(_feature_root("TinyImagenet"), "conv_12_features_50_vgg16_bn_-4_-5_32_200_2020.npy"),
        }
    }
}

ADV_PATHS = {
    "TinyImagenet": {
        "vgg16": {
            "logits": {
                "pgd_inf_0.1_1_50_adv_all": {adv_iter: osp.join(_adv_feature_root("TinyImagenet"),
                                                                "vgg16_bn_-4_-5_32_200_2020",
                                                                "pgd_inf_0.1_1_50_adv_all",
                                                                f"logits_50_iter_{adv_iter}.npy")
                                             for adv_iter in range(21)},
            },
            "conv_53_features": {
                "pgd_inf_0.1_1_50_adv_all": {adv_iter: osp.join(_adv_feature_root("TinyImagenet"),
                                                                "vgg16_bn_-4_-5_32_200_2020",
                                                                "pgd_inf_0.1_1_50_adv_all",
                                                                f"features_50_iter_{adv_iter}.npy")
                                             for adv_iter in range(21)},
            },
        }
    }
}


CLASS_NAMES = {
    "TinyImagenet": ['bridge', 'bus', 'car', 'cat', 'desk', 'dog', 'frog', 'ipod', 'lifeboat', 'orange']
}