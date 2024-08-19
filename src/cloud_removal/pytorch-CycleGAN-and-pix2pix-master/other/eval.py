from torch import nn
import numpy as np
from skimage.metrics import structural_similarity as SSIM
import os
from PIL import Image
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()

path = r"C:\Users\lenovo\Desktop\cloudthick_percent_cut\results\3\multi_model_end_coverpercent\test_latest\images"

list_mse = []
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
    real_image = to_tensor(real_image)
    rec_image = to_tensor(rec_image)

    mse = criterionMSE(real_image, rec_image)
    psnr = 10 * np.log(1 / mse.item())
    img1 = np.tensordot(real_image.cpu().numpy()[0, :3], [0.298912, 0.586611, 0.114478], axes=0)
    img2 = np.tensordot(rec_image.cpu().numpy()[0, :3], [0.298912, 0.586611, 0.114478], axes=0)

    ssim = SSIM(img1, img2, data_range=1, win_size=3)

    list_mse.append(mse)
    list_psnr.append(psnr)
    list_ssim.append(ssim)

print("===> Avg. MSE: {:.4f}".format(np.mean(list_mse)))
print("===> Avg. PSNR: {:.4f} dB".format(np.mean(list_psnr)))
print("===> Avg. SSIM: {:.4f} dB".format(np.mean(list_ssim)))



























