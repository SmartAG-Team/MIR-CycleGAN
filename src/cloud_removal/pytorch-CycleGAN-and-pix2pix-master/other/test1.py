from skimage.metrics import peak_signal_noise_ratio as Psnr
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import numpy as np
import math

path = r"C:\Users\lenovo\Desktop\test_latest\images"

f_nums = len(os.listdir(path))
list = os.listdir(path + "/")
print(len(list))

filename_list = []
for i in range(len(list)):
    # print(list[i])
    parts = list[i].split("_")[:2]
    print(parts)
    # print(parts)
    result = "_".join(parts[:2])
    # print(result)
    if result not in filename_list:
        print(result)
        filename_list.append(result)