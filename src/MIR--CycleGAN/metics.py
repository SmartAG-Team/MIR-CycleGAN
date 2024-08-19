import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from math import log10, sqrt

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

# img1 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\Deadline-groundtuth.png')
# img2 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\Deadline-MIR.png')

# img1 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\ETM-ground.png')
# img2 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\ETM-MIR.png')

img1 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\Thick-groundtruth.png')
img2 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\Thick-MIR.png') 

# img1 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\Thin-groundtruth.png')
# img2 = cv2.imread(r'C:\lzy\MIR-CycleGAN\mir-images\Thin-MIR.png')



# 将图片转换为灰度图
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 检查图片是否成功读取
if img1 is None or img2 is None:
    print("Error: One or both images not found or could not be opened.")
    exit()

# 确保两张图片大小相同
if img1.shape != img2.shape:
    print("Error: Images must have the same dimensions.")
    exit()

# 计算PSNR
psnr_value = calculate_psnr(img1, img2)
print(f'PSNR: {psnr_value}')

# 计算SSIM
ssim_value = calculate_ssim(img1, img2)
print(f'SSIM: {ssim_value}')

# 计算RMSE
rmse_value = calculate_rmse(img1, img2)
print(f'RMSE: {rmse_value}')

# 计算R²
r2_value = r2_score(img1.flatten(), img2.flatten())
print(f'R²: {r2_value}')

# 计算MAE
mae_value = mean_absolute_error(img1.flatten(), img2.flatten())
print(f'MAE: {mae_value}')

# 绘制重建图像与真实图像之间的散点图
plt.figure(figsize=(7, 7))
plt.scatter(img1.flatten(), img2.flatten(), alpha=0.5, color='blue', s=1)
# plt.title('Scatter Plot between Original and Reconstructed Images')
plt.xlabel('Ground truth Image Pixel Values')
plt.ylabel('Reconstructed Image Pixel Values')
plt.grid(True)

# 添加y=x的参考线
min_val = min(img1.min(), img2.min())
max_val = max(img1.max(), img2.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # y=x 参考线

# 在左上角添加文本信息
textstr = '\n'.join((
    f'PSNR: {psnr_value:.2f}',
    f'SSIM: {ssim_value:.4f}',
    f'RMSE: {rmse_value:.2f}',
    f'R²: {r2_value:.4f}',
    f'MAE: {mae_value:.2f}',
))

# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top')

# 取消网格
plt.grid(False)

plt.show()
