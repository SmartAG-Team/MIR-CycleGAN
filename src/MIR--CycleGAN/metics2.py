import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from math import log10, sqrt

# Set global font size for Matplotlib
plt.rcParams.update({'font.size': 10})

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    rmse = sqrt(mse)
    return rmse

def calculate_ssim(img1, img2):
    s, _ = ssim(img1, img2, full=True)
    return s

# Define paths for all image pairs
image_pairs = [
    (r'C:\lzy\MIR-CycleGAN\mir-images\Deadline-groundtuth.png', r'C:\lzy\MIR-CycleGAN\mir-images\Deadline-MIR.png'),
    (r'C:\lzy\MIR-CycleGAN\mir-images\ETM-ground.png', r'C:\lzy\MIR-CycleGAN\mir-images\ETM-MIR.png'),
    (r'C:\lzy\MIR-CycleGAN\mir-images\Thin-groundtruth.png', r'C:\lzy\MIR-CycleGAN\mir-images\Thin-MIR.png'),
    (r'C:\lzy\MIR-CycleGAN\mir-images\Thick-groundtruth.png', r'C:\lzy\MIR-CycleGAN\mir-images\Thick-MIR.png')
]

# Initialize Matplotlib figure for subplots
fig, axs = plt.subplots(2, 2, figsize=(9, 9))

for i, (img1_path, img2_path) in enumerate(image_pairs):
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute metrics
    psnr_value = calculate_psnr(img1_gray, img2_gray)
    ssim_value = calculate_ssim(img1_gray, img2_gray)
    rmse_value = calculate_rmse(img1_gray, img2_gray)
    r2_value = r2_score(img1_gray.flatten(), img2_gray.flatten())
    mae_value = mean_absolute_error(img1_gray.flatten(), img2_gray.flatten())
 
    # Plot images in subplots
    ax = axs[i // 2, i % 2]
    ax.scatter(img1_gray.flatten(), img2_gray.flatten(), alpha=0.5, color='blue', s=1)
    ax.plot([0, 255], [0, 255], 'r--')  # y=x reference line

    if i == 0:
        ax.set_title(f'Dead detectors in MODIS')
        ax.set_xticks([])
    if i == 1:
        ax.set_title(f'ETM+ SLC-off')
        ax.set_xticks([])
        ax.set_yticks([])
    if i == 2:
        ax.set_title(f'Thin cloud')
    if i == 3:
        ax.set_title(f'Thick cloud')  
        ax.set_yticks([])


    if i == 0 or i == 2:
        ax.set_ylabel('Reconstructed Image Pixel Values')
    if i == 2 or i == 3:
        ax.set_xlabel('Ground truth Image Pixel Values')

    ax.grid(False)

    # Add metrics text with adjusted font size
    textstr = '\n'.join((
        f'PSNR: {psnr_value:.2f}',
        f'SSIM: {ssim_value:.4f}',
        f'RMSE: {rmse_value:.2f}',
        f'R²: {r2_value:.4f}',
        f'MAE: {mae_value:.2f}',
    ))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()