from torch import nn
import numpy as np
from skimage.metrics import structural_similarity as SSIM
import os
from PIL import Image
from torchvision import transforms
import math

import warnings
warnings.filterwarnings("ignore")

criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()

path = r"C:\lzy\MIR-CycleGAN\root\autodl-tmp\results\latest-old\SLCoff\mir_cyclegan\test_latest\images"

list_mse = []
list_rmse = []
list_psnr = []
list_ssim = []

f_nums = len(os.listdir(path))
list = os.listdir(path + "/")
print(len(list))



filename_list = []
for i in range(len(list)):
    parts = list[i].split("_")[:2]
    result = "_".join(parts[:2])
    if result not in filename_list:
        print(result)
        filename_list.append(result)

for i in range(len(filename_list)):
    print("正在计算第 " + str(i) + " 组图片的评价指标~")
    # if os.path.exists(path + "/" + filename_list[i] + '_fake_B.tif') and os.path.exists(path + "/" + filename_list[i] + '_real_B.tif'):
    #     rec_image = Image.open(path + "/" + filename_list[i] + '_fake_B.tif')
    #     real_image = Image.open(path + "/" + filename_list[i] + '_real_B.tif')
    if os.path.exists(path + "/" + filename_list[i] + '_fake_B.png') and os.path.exists(path + "/" + filename_list[i] + '_real_B.png'):
        rec_image = Image.open(path + "/" + filename_list[i] + '_fake_B.png')
        real_image = Image.open(path + "/" + filename_list[i] + '_real_B.png')
    else:
        print("图片不存在！")
        continue

    
    to_tensor = transforms.ToTensor()
    real_image = to_tensor(real_image).unsqueeze(0)  # 增加一个批次维度
    rec_image = to_tensor(rec_image).unsqueeze(0)  # 增加一个批次维度

    # 计算 MSE
    mse = criterionMSE(real_image, rec_image)
    rmse = math.sqrt(mse.item())

    # 计算 PSNR，注意图像范围在 [0, 1]，所以这里的最大值为 1.0
    psnr = 20 * np.log10(1.0 / rmse)

    # 将图像转换为 NumPy 数组，用于 SSIM 计算
    # img1 = real_image.squeeze().permute(1, 2, 0).numpy()  # 去掉批次维度并转换为 NumPy
    # img2 = rec_image.squeeze().permute(1, 2, 0).numpy()
    img1 = np.tensordot(real_image.cpu().numpy()[0, :3], [0.298912, 0.586611, 0.114478], axes=0)
    img2 = np.tensordot(rec_image.cpu().numpy()[0, :3], [0.298912, 0.586611, 0.114478], axes=0)

    ssim = SSIM(img1, img2, data_range=1,channel_axis=-1,win_size=3)

    list_mse.append(mse)
    list_rmse.append(np.sqrt(mse))
    list_psnr.append(psnr)
    list_ssim.append(ssim)

print("===> Avg. MSE: {:.4f}".format(np.mean(list_mse)))
print("===> Avg. RMSE: {:.4f}".format(np.mean(list_rmse)))
print("===> Avg. PSNR: {:.4f} dB".format(np.mean(list_psnr)))
print("===> Avg. SSIM: {:.4f} dB".format(np.mean(list_ssim)))



























