from skimage.metrics import peak_signal_noise_ratio as Psnr
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import numpy as np
import math

# SSIM PSNR CC SAM

def sam(src, dst):
    # src, dst均为numpy数组
    val = np.dot(src.T, dst)/(np.linalg.norm(src)*np.linalg.norm(dst))
    sam = np.arccos(val)
    return sam

# def psnr(target, ref):
# 　　#将图像格式转为float64
# 　　target_data = np.array(target, dtype=np.float64)
# 　　ref_data = np.array(ref,dtype=np.float64)
# 　　# 直接相减，求差值
# 　　diff = ref_data - target_data
# 　　# 按第三个通道顺序把三维矩阵拉平
# 　　diff = diff.flatten('C')
# 　　# 计算MSE值
# 　　rmse = math.sqrt(np.mean(diff ** 2.))
# 　　# 精度
# 　　eps = np.finfo(np.float64).eps
# 　　if(rmse == 0):
# 　　rmse = eps
# 　　return 20*math.log10(255.0/rmse)

def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if rmse == 0:
        rmse = eps
    return 20 * math.log10(255.0 / rmse)

# path = r"/root/autodl-tmp/SLCoff/results/multi_model_SLCoff_test_51/test_13/images"
path = r"C:\lzy\MIR-CycleGAN\root\autodl-tmp\results\SLCoff\mir_cyclegan\test_latest\images"

list_psnr_1 = []
list_ssim_1 = []
list_psnr_2 = []
list_ssim_2 = []
list_psnr_3 = []
list_cc = []
list_cc_1 = []
list_cc_2 =[]
list_SAM = []
list_MAE = []
list_RMSE = []
f_nums = len(os.listdir(path))
list = os.listdir(path + "/")

# print(list)

list1 = []
for i in range(len(list)):
    if(i % 7 == 0):
        list1.append(i)

count = 0
sam_1 = 0
sam_2 = 0.0

for i in list1:
    if count == 1002:
        break
    count = count + 1
    print("正在计算第 " + str(count) + " 组图片的评价指标~")
    img1 = cv2.imread(path + "/" + list[i+1])
    print(path + "/" + list[i+1])
    print(path + "/" + list[i+4])
    img2 = cv2.imread(path + "/" + list[i+4])
    # 转为灰度图像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # if psnr(img1, img2) == float("inf"):
    #     list_psnr_1.append(40)
    # else:
    #     list_psnr_1.append(psnr(img1, img2))
    if Psnr(img1, img2) != float("inf"):
        list_psnr_1.append(Psnr(img1, img2))
    list_ssim_1.append(ssim(img1, img2, channel_axis=-1))
    #
    # # 计算直方图
    img1_h1 = cv2.calcHist([img1_gray], [0], None, [256], [0.0, 255.0])
    img2_h2 = cv2.calcHist([img2_gray], [0], None, [256], [0.0, 255.0])
    list_cc_1.append(cv2.compareHist(img1_h1, img2_h2, method=cv2.HISTCMP_CORREL))

    if Psnr(img1_gray, img2_gray, data_range=255) != float("inf"):
        list_psnr_2.append(Psnr(img1_gray, img2_gray, data_range=255))
    list_ssim_2.append(ssim(img1_gray, img2_gray, win_size=11, data_range=255))

    list_psnr_3.append(psnr(img1, img2))

    list_MAE.append(np.mean(np.abs(img1 - img2)))
    list_RMSE.append(np.sqrt(np.mean(np.square(img1 - img2))))

print("彩色平均PSNR:", np.mean(list_psnr_1))
print("彩色平均SSIM:", np.mean(list_ssim_1))
print("灰度平均PSNR:", np.mean(list_psnr_2))
print("灰度平均SSIM:", np.mean(list_ssim_2))
print("PSNR3:", np.mean(list_psnr_3))
print("CC_1:", np.mean(list_cc_1))
# print("CC_2:", np.mean(list_cc_2))
# print("MSAM:", np.mean(list_SAM))
