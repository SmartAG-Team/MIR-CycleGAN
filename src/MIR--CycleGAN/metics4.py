import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, r2_score
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
    (r'C:\lzy\MIR-CycleGAN\Deadlines\7.png', r'C:\lzy\MIR-CycleGAN\Deadlines\2.png'),
    (r'C:\lzy\MIR-CycleGAN\Deadlines\7.png', r'C:\lzy\MIR-CycleGAN\Deadlines\3.png'),
    (r'C:\lzy\MIR-CycleGAN\Deadlines\7.png', r'C:\lzy\MIR-CycleGAN\Deadlines\4.png'),
    (r'C:\lzy\MIR-CycleGAN\Deadlines\7.png', r'C:\lzy\MIR-CycleGAN\Deadlines\5.png'),
    (r'C:\lzy\MIR-CycleGAN\Deadlines\7.png', r'C:\lzy\MIR-CycleGAN\Deadlines\6.png'),
    (r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\7.png', r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\2.png'),
    (r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\7.png', r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\3.png'),
    (r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\7.png', r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\4.png'),
    (r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\7.png', r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\5.png'),
    (r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\7.png', r'C:\lzy\MIR-CycleGAN\ETM+SLC-off\6.png'),
    (r'C:\lzy\MIR-CycleGAN\Thick\7.png', r'C:\lzy\MIR-CycleGAN\Thick\2.png'),
    (r'C:\lzy\MIR-CycleGAN\Thick\7.png', r'C:\lzy\MIR-CycleGAN\Thick\3.png'),
    (r'C:\lzy\MIR-CycleGAN\Thick\7.png', r'C:\lzy\MIR-CycleGAN\Thick\4.png'),
    (r'C:\lzy\MIR-CycleGAN\Thick\7.png', r'C:\lzy\MIR-CycleGAN\Thick\5.png'),
    (r'C:\lzy\MIR-CycleGAN\Thick\7.png', r'C:\lzy\MIR-CycleGAN\Thick\6.png'),
    (r'C:\lzy\MIR-CycleGAN\Thin\7.png', r'C:\lzy\MIR-CycleGAN\Thin\2.png'),
    (r'C:\lzy\MIR-CycleGAN\Thin\7.png', r'C:\lzy\MIR-CycleGAN\Thin\3.png'),
    (r'C:\lzy\MIR-CycleGAN\Thin\7.png', r'C:\lzy\MIR-CycleGAN\Thin\4.png'),
    (r'C:\lzy\MIR-CycleGAN\Thin\7.png', r'C:\lzy\MIR-CycleGAN\Thin\5.png'),
    (r'C:\lzy\MIR-CycleGAN\Thin\7.png', r'C:\lzy\MIR-CycleGAN\Thin\6.png')
]

fig, axs = plt.subplots(4, 5, figsize=(12, 9))

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
    # mae_value = mean_absolute_error(img1_gray.flatten(), img2_gray.flatten())
 
    # Plot images in subplots
    row = i // 5  # Determine row index in subplot grid
    col = i % 5   # Determine column index in subplot grid
    ax = axs[row, col]  # Select the correct subplot

    ax.scatter(img1_gray.flatten(), img2_gray.flatten(), alpha=0.5, color='blue', s=1)
    ax.plot([0, 255], [0, 255], 'r--')  # y=x reference line
    if row == 0 or row == 1 or row == 2:
        ax.set_xticks([])
    else:
        ax.set_xticks(range(0, 256, 50))  # Set X-axis ticks at intervals of 50

    if col == 1 or col == 2 or col == 3 or col == 4:
        ax.set_yticks([])
    else:
        ax.set_yticks(range(0, 256, 50))  # Set Y-axis ticks at intervals of 50

    ax.grid(False)

    # Add metrics text with adjusted font size
    textstr = '\n'.join((
        f'PSNR: {psnr_value:.2f}',
        f'SSIM: {ssim_value:.4f}',
        f'RMSE: {rmse_value:.2f}',
        f'R²: {r2_value:.4f}',
        # f'MAE: {mae_value:.2f}',
    ))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top')

# Add titles above each column
title_positions = [0.17, 0.35, 0.54, 0.720, 0.910]
for col, (title, pos) in enumerate(zip(['SpA GAN', 'ST-GAN', 'PMAA', 'STS-CNN', 'MIR-CycleGAN'], title_positions)):
    fig.text(pos, 0.92, title, ha='center', fontsize=12)

row_labels = ['Deadlines', 'ETM+SLC-off', 'Thick cloud', 'Thin cloud']
row_label_positions = [0.80, 0.60, 0.40, 0.20]
for label, pos in zip(row_labels, row_label_positions):
    fig.text(0.04, pos, label, va='center', rotation='vertical', fontsize=12)

# Add a single X-axis label and Y-axis label for the entire figure
fig.text(0.5, 0.04, 'Ground truth Image Pixel Values', ha='center', fontsize=12)
fig.text(0.015, 0.5, 'Reconstructed Image Pixel Values', va='center', rotation='vertical', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0.05, 0.05, 1, 0.92])

# Show plot
plt.show()
