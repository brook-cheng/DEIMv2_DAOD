import os
import torch
import numpy as np
import cv2
import sklearn
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_feature(feature, dst_shape=(640, 640), rate=1.0):
    # (b,c,w,h)
    feature = torch.sigmoid(feature)

    # (c,wxh)
    w = feature.shape[2]
    h = feature.shape[3]
    feature = feature[0].reshape(feature.shape[1], -1).permute(1, 0)
    feature = feature.detach().cpu().numpy()

    if rate == 1.0:
        pca = PCA()
    else:
        pca = PCA(n_components=rate)

    pca_feature = pca.fit_transform(feature)
    pca_feature_mean = pca_feature.mean(axis=1).reshape(-1, 1)
    pca_feature_max = pca_feature.max(axis=1).reshape(-1, 1)
    pca_feature_min = pca_feature.min(axis=1).reshape(-1, 1)
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    pca_feature_mean_scaled = scaler.fit_transform(pca_feature_mean)
    pca_feature_max_scaled = scaler.fit_transform(pca_feature_max)
    pca_feature_min_scaled = scaler.fit_transform(pca_feature_min)
    pca_feature_scaled_mean_16bit = pca_feature_mean_scaled.reshape(h, w)
    pca_feature_scaled_mean_16bit = cv2.resize(
        pca_feature_scaled_mean_16bit, dst_shape
    ).reshape(dst_shape[0], dst_shape[1], 1)
    pca_feature_max_scaled_16bit = pca_feature_max_scaled.reshape(h, w)
    pca_feature_max_scaled_16bit = cv2.resize(
        pca_feature_max_scaled_16bit, dst_shape
    ).reshape(dst_shape[0], dst_shape[1], 1)
    pca_feature_min_scaled_16bit = pca_feature_min_scaled.reshape(h, w)
    pca_feature_min_scaled_16bit = cv2.resize(
        pca_feature_min_scaled_16bit, dst_shape
    ).reshape(dst_shape[0], dst_shape[1], 1)

    return (
        pca_feature_min_scaled_16bit,
        pca_feature_max_scaled_16bit,
        pca_feature_scaled_mean_16bit,
    )


def save_pca_features(img, features, dir_path="./test/data/output/PCA/"):
    feats_num = len(features)
    plt.figure(figsize=(3, feats_num), dpi=640)
    plt.subplots_adjust(
        wspace=0.05, hspace=0.22, top=0.95, bottom=0.05, left=0.05, right=0.95
    )
    print("feats_num:", feats_num)
    for idx, feature in enumerate(features):
        img_resized = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        pca_feature_all = pca_feature(feature, rate=1.0)
        print(f"draw feature {idx}")
        alpha = 0.5
        beta = 1 - alpha

        plt.subplot(feats_num, 3, 3 * idx + 1)
        pca_feature_ch3 = (pca_feature_all[0].repeat(3, axis=2) * 255).astype(np.uint8)
        pca_feature_ch3 = cv2.applyColorMap(pca_feature_ch3, cv2.COLORMAP_JET)
        pca_img_merge = (img_resized * alpha + pca_feature_ch3 * beta).astype(np.uint8)
        plt.imshow(pca_img_merge)
        plt.title(f"min{idx}", fontsize=8)
        plt.axis("off")

        plt.subplot(feats_num, 3, 3 * idx + 2)
        pca_feature_ch3 = (pca_feature_all[1].repeat(3, axis=2) * 255).astype(np.uint8)
        pca_feature_ch3 = cv2.applyColorMap(pca_feature_ch3, cv2.COLORMAP_JET)
        pca_img_merge = (img_resized * alpha + pca_feature_ch3 * beta).astype(np.uint8)
        plt.imshow(pca_img_merge)
        plt.title(f"max{idx}", fontsize=8)
        plt.axis("off")

        plt.subplot(feats_num, 3, 3 * idx + 3)
        pca_feature_ch3 = (pca_feature_all[2].repeat(3, axis=2) * 255).astype(np.uint8)
        pca_feature_ch3 = cv2.applyColorMap(pca_feature_ch3, cv2.COLORMAP_JET)
        pca_img_merge = (img_resized * alpha + pca_feature_ch3 * beta).astype(np.uint8)
        plt.imshow(pca_img_merge)
        plt.title(f"mean{idx}", fontsize=8)
        plt.axis("off")
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, "pca_features.png")
    plt.savefig(save_path, bbox_inches="tight")
