from skimage.metrics import peak_signal_noise_ratio as Psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


def mse1(imageA, imageB):
    # 计算两张图片的MSE相似度
    # 注意：两张图片必须具有相同的维度，因为是基于图像中的对应像素操作的
    # 对应像素相减并将结果累加起来
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # 进行误差归一化
    err /= float(imageA.shape[0] * imageA.shape[1])

    # 返回结果，该值越小越好，越小说明两张图像越相似
    return err


# path = r"/root/autodl-tmp/SLCoff/results/multi_model_SLCoff_test_51/test_13/images"
path = r"C:\Users\lenovo\Desktop\cloudthick_percent_cut\results\3\multi_model_end_coverpercent\test_latest\images"

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
list_MSE = []
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
        # print(path + "/" + filename_list[i] + '_fake_B.png')
        # print(path + "/" + filename_list[i] + '_real_B.png')
    else:
        print("图片不存在！")
        continue

    # 转为灰度图像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if Psnr(img1, img2) != float("inf"):
        list_psnr_1.append(Psnr(img1, img2))
    list_ssim_1.append(ssim(img1, img2, channel_axis=-1))

    # # 计算直方图
    img1_h1 = cv2.calcHist([img1_gray], [0], None, [256], [0.0, 255.0])
    img2_h2 = cv2.calcHist([img2_gray], [0], None, [256], [0.0, 255.0])
    list_cc_1.append(cv2.compareHist(img1_h1, img2_h2, method=cv2.HISTCMP_CORREL))

    if Psnr(img1_gray, img2_gray, data_range=255) != float("inf"):
        list_psnr_2.append(Psnr(img1_gray, img2_gray, data_range=255))
    list_ssim_2.append(ssim(img1_gray, img2_gray, win_size=11, data_range=255))

    list_psnr_3.append(psnr(img1, img2))
    # list_MAE.append(mean_absolute_error(img1, img2))
    list_MAE.append(np.abs(img1_gray - img2_gray).mean())
    list_RMSE.append(np.sqrt(np.mean(np.square(img1_gray - img2_gray))))
    list_MSE.append(mse(img1, img2))


print("彩色平均PSNR:", np.mean(list_psnr_1))
print("彩色平均SSIM:", np.mean(list_ssim_1))
print("灰度平均PSNR:", np.mean(list_psnr_2))
print("灰度平均SSIM:", np.mean(list_ssim_2))
print("PSNR3:", np.mean(list_psnr_3))
print("CC_1:", np.mean(list_cc_1))
print("MAE: ", np.mean(list_MAE))
print("RMSE: ", np.mean(list_RMSE))
print("MSE: ", np.mean(list_MSE))
