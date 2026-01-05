import os
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
import sklearn
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def save_pca_features(img, features, dir_path="./test/output/"):
    # (b,c,w,h)
    features = torch.sigmoid(features)

    def pca_features(feature, w, h, dst_shape=(640, 640), rate=1.0):
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

    # (c,wxh)
    w = features.shape[2]
    h = features.shape[3]

    img_resized = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pca_features_prepared = features[0].reshape(features.shape[1], -1).permute(1, 0)
    pca_features_prepared = pca_features_prepared.cpu().numpy()

    pca_feature_all = pca_features(pca_features_prepared, w, h, rate=1.0)

    alpha = 0.5
    beta = 1 - alpha
    plt.figure()
    plt.subplot(1, 3, 1)
    pca_feature_ch3 = (pca_feature_all[0].repeat(3, axis=2) * 255).astype(np.uint8)
    pca_feature_ch3 = cv2.applyColorMap(pca_feature_ch3, cv2.COLORMAP_JET)
    pca_img_merge = (img_resized * alpha + pca_feature_ch3 * beta).astype(np.uint8)
    plt.imshow(pca_img_merge)
    plt.title("all_min")

    plt.subplot(1, 3, 2)
    pca_feature_ch3 = (pca_feature_all[1].repeat(3, axis=2) * 255).astype(np.uint8)
    pca_feature_ch3 = cv2.applyColorMap(pca_feature_ch3, cv2.COLORMAP_JET)
    pca_img_merge = (img_resized * alpha + pca_feature_ch3 * beta).astype(np.uint8)
    plt.imshow(pca_img_merge)
    plt.title("all_max")
    plt.subplot(1, 3, 3)

    pca_feature_ch3 = (pca_feature_all[2].repeat(3, axis=2) * 255).astype(np.uint8)
    pca_feature_ch3 = cv2.applyColorMap(pca_feature_ch3, cv2.COLORMAP_JET)
    pca_img_merge = (img_resized * alpha + pca_feature_ch3 * beta).astype(np.uint8)
    plt.imshow(pca_img_merge)
    plt.title("all_mean")
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(f"{dir_path}_pca_.png")
