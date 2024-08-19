from skimage.metrics import peak_signal_noise_ratio as Psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import log10, sqrt
import math
from torch import nn
from PIL import Image
from torchvision import transforms
import os
# SSIM PSNR CC SAM


path = r"C:\lzy\MIR-CycleGAN\root\autodl-tmp\results\latest-old\SLCoff\mir_cyclegan\test_latest\images"


list_psnr_1 = []
list_ssim_1 = []
list_psnr_2 = []
list_ssim_2 = []
list_psnr_3 = []
list_MAE = []
list_RMSE = []
list_MSE = []

list_mse=[]
list_rmse=[]

f_nums = len(os.listdir(path))
list = os.listdir(path + "/")
print(len(list))

filename_list = []
for i in range(len(list)):
    # print(list[i])
    parts = list[i].split("_")[:2]
    # print(parts)
    result = "_".join(parts[:2])
    # print(result)
    if result not in filename_list:
        print(result)
        filename_list.append(result)

print(len(filename_list))

sam_1 = 0
sam_2 = 0.0

for i in range(len(filename_list)):
    print("正在计算第 " + str(i) + " 组图片的评价指标~")
    # if os.path.exists(path + "/" + filename_list[i] + '_fake_B.tif') and os.path.exists(path + "/" + filename_list[i] + '_real_B.tif'):
    #     img1 = cv2.imread(path + "/" + filename_list[i] + '_fake_B.tif')
    #     img2 = cv2.imread(path + "/" + filename_list[i] + '_real_B.tif')
    if os.path.exists(path + "/" + filename_list[i] + '_fake_B.png') and os.path.exists(path + "/" + filename_list[i] + '_real_B.png'):
        img1 = cv2.imread(path + "/" + filename_list[i] + '_fake_B.png')
        img2 = cv2.imread(path + "/" + filename_list[i] + '_real_B.png')
        print(path + "/" + filename_list[i] + '_fake_B.png')
        print(path + "/" + filename_list[i] + '_real_B.png')
    else:
        print("图片不存在！")
        continue
    criterionMSE = nn.MSELoss()
    criterionMSE = criterionMSE.cuda()
    to_tensor = transforms.ToTensor()
    real_image = to_tensor(img1)
    rec_image = to_tensor(img2)

    mse12 = criterionMSE(real_image, rec_image)
    rmse = math.sqrt(mse12.item())
    list_mse.append(mse12)
    list_rmse.append(np.sqrt(mse12))

    # 转为灰度图像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if Psnr(img1, img2) != float("inf"):
        list_psnr_1.append(Psnr(img1, img2))
    list_ssim_1.append(ssim(img1, img2, channel_axis=-1,data_range = 255))   # channel_axis=-1  data_range=255

    if Psnr(img1_gray, img2_gray, data_range=255) != float("inf"):
        list_psnr_2.append(Psnr(img1_gray, img2_gray,data_range=255))
    list_ssim_2.append(ssim(img1_gray, img2_gray, channel_axis=-1,data_range=255))



print("彩色平均PSNR:", np.mean(list_psnr_1))
print("彩色平均SSIM:", np.mean(list_ssim_1))
print("灰度平均PSNR:", np.mean(list_psnr_2))
print("灰度平均SSIM:", np.mean(list_ssim_2))
print("RMSE: ", np.mean(list_rmse))
print("MSE: ", np.mean(list_mse))
