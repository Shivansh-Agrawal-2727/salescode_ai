from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import color, filters, feature


_fft_names = []
for r_name in ["b1", "b2", "b3"]:
    for s_name in ["s1", "s2", "s3", "s4"]:
        _fft_names.extend([
            f"fft_{r_name}_{s_name}_mean",
            f"fft_{r_name}_{s_name}_max_to_mean",
            f"fft_{r_name}_{s_name}_std_to_mean"
        ])

FEATURE_NAMES = [
    "mean_red",
    "mean_green",
    "mean_blue",
    "std_red",
    "std_green",
    "std_blue",
    "mean_hue",
    "mean_saturation",
    "std_saturation",
    "mean_value",
    "std_value",
    "contrast_p2_p98",
    "edge_density",
    "canny_density",
    "laplacian_variance",
    "sobel_mean",
    "gray_entropy",
    "overexposed_ratio",
    "underexposed_ratio",
    *_fft_names,
    *[f"lbp_bin_{i}" for i in range(10)],
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_ASM",
    "r_g_corr",
    "g_b_corr",
    "b_r_corr",
    *[f"r_hist_{i}" for i in range(8)],
    *[f"g_hist_{i}" for i in range(8)],
    *[f"b_hist_{i}" for i in range(8)],
    *[f"grad_hist_{i}" for i in range(8)],
]


def extract_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = color.rgb2gray(rgb)
    hsv = color.rgb2hsv(rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)

    # 1. Color stats
    channels_mean = rgb.reshape(-1, 3).mean(axis=0)
    channels_std = rgb.reshape(-1, 3).std(axis=0)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    # 2. Brightness contrast and limits
    p2, p98 = np.percentile(gray, [2, 98])
    sobel = filters.sobel(gray)
    laplacian = ndimage.laplace(gray)

    # 3. FFT Radial & Angular features
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fft)
    height, width = gray.shape
    cy, cx = height / 2, width / 2
    y, x = np.ogrid[:height, :width]
    
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    theta = np.arctan2(y - cy, x - cx)
    
    min_dim = min(height, width)
    band1_mask = (r >= min_dim * 0.02) & (r < min_dim * 0.10)
    band2_mask = (r >= min_dim * 0.10) & (r < min_dim * 0.25)
    band3_mask = (r >= min_dim * 0.25)
    
    theta_folded = theta % np.pi
    
    sec1_mask = (theta_folded >= 0) & (theta_folded < np.pi/4)
    sec2_mask = (theta_folded >= np.pi/4) & (theta_folded < np.pi/2)
    sec3_mask = (theta_folded >= np.pi/2) & (theta_folded < 3*np.pi/4)
    sec4_mask = (theta_folded >= 3*np.pi/4) & (theta_folded <= np.pi)
    
    fft_features = []
    for r_mask in [band1_mask, band2_mask, band3_mask]:
        for s_mask in [sec1_mask, sec2_mask, sec3_mask, sec4_mask]:
            mask = r_mask & s_mask
            if np.any(mask):
                vals = magnitude[mask]
                mean_v = vals.mean() + 1e-8
                max_v = vals.max()
                std_v = vals.std()
                fft_features.extend([
                    float(mean_v),
                    float(max_v / mean_v),
                    float(std_v / mean_v)
                ])
            else:
                fft_features.extend([0.0, 0.0, 0.0])

    # 4. LBP features on uint8 image (uniform method -> 10 bins)
    lbp = feature.local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    # 5. GLCM features (quantized to 16 levels to make it fast)
    gray_q = (gray * 15).astype(np.uint8)
    glcm = feature.graycomatrix(gray_q, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
    glcm_contrast = feature.graycoprops(glcm, 'contrast').mean()
    glcm_dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
    glcm_homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
    glcm_energy = feature.graycoprops(glcm, 'energy').mean()
    glcm_correlation = feature.graycoprops(glcm, 'correlation').mean()
    glcm_ASM = feature.graycoprops(glcm, 'ASM').mean()

    # 6. Color channel correlation features
    r_flat = rgb[..., 0].ravel()
    g_flat = rgb[..., 1].ravel()
    b_flat = rgb[..., 2].ravel()
    r_g_corr = np.corrcoef(r_flat, g_flat)[0, 1]
    g_b_corr = np.corrcoef(g_flat, b_flat)[0, 1]
    b_r_corr = np.corrcoef(b_flat, r_flat)[0, 1]

    # 7. RGB Color histograms (8 bins per channel -> 24 features)
    r_hist, _ = np.histogram(r_flat, bins=8, range=(0, 1), density=True)
    g_hist, _ = np.histogram(g_flat, bins=8, range=(0, 1), density=True)
    b_hist, _ = np.histogram(b_flat, bins=8, range=(0, 1), density=True)

    # 8. Gradient orientation histogram
    dx = ndimage.sobel(gray, axis=1)
    dy = ndimage.sobel(gray, axis=0)
    magnitudes = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx)
    angles_deg = np.rad2deg(angles) % 180
    grad_hist, _ = np.histogram(angles_deg, bins=8, range=(0, 180), weights=magnitudes)
    grad_hist = grad_hist / (grad_hist.sum() + 1e-8)

    # Basic stats
    hist_gray, _ = np.histogram(gray, bins=64, range=(0.0, 1.0), density=True)
    hist_gray = hist_gray + 1e-12
    entropy = -np.sum(hist_gray * np.log2(hist_gray)) / 64.0

    edge_density = 0.0
    if np.any(sobel):
        threshold = filters.threshold_otsu(sobel)
        edge_density = float((sobel > threshold).mean())

    features = np.array(
        [
            *channels_mean,
            *channels_std,
            float(hsv[..., 0].mean()),
            float(saturation.mean()),
            float(saturation.std()),
            float(value.mean()),
            float(value.std()),
            float(p98 - p2),
            edge_density,
            float(feature.canny(gray, sigma=1.2).mean()),
            float(laplacian.var()),
            float(sobel.mean()),
            float(entropy),
            float((gray > 0.96).mean()),
            float((gray < 0.04).mean()),
            
            # FFT features (36 features)
            *fft_features,
            
            # LBP hist (10 features)
            *lbp_hist.tolist(),
            
            # GLCM (6 features)
            float(glcm_contrast),
            float(glcm_dissimilarity),
            float(glcm_homogeneity),
            float(glcm_energy),
            float(glcm_correlation),
            float(glcm_ASM),
            
            # Channel correlations (3 features)
            float(r_g_corr),
            float(g_b_corr),
            float(b_r_corr),
            
            # Color histograms (24 features)
            *r_hist.tolist(),
            *g_hist.tolist(),
            *b_hist.tolist(),
            
            # Gradient histogram (8 features)
            *grad_hist.tolist(),
        ],
        dtype=np.float32,
    )

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

