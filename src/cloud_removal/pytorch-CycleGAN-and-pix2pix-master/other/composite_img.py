import cv2 as cv
# from osgeo import gdal
import matplotlib.pyplot as plt
import pylab
import numpy as np
from time import *
from PIL import Image
import os
from collections import defaultdict
from random import choice
# import splitfolders

# 将条带添加到原始图像中去
# 将mask的背景透明化
def transPNG(srcImageName):
    img = Image.open(srcImageName)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] > 220 and item[1] > 220 and item[2] >220:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

# 将所有的遮挡背景透明化并保存
# for i in range(1, 10):
#     mask = transPNG(r"D:\Code\8-16-cycle_GAN\pytorch-CycleGAN-and-pix2pix-master\masks\Mask" + str(i) + ".tif")
#     mask = mask.resize((256, 256))
#     mask.save(r"D:\Code\8-16-cycle_GAN\pytorch-CycleGAN-and-pix2pix-master\trans_masks\Mask" + str(i) + ".png")

# 图片融合
def mix(img1, img2, coordinator):
    """
    将遮挡与原图融合为代修复遥感图像
    :param img1: 原始图像
    :param img2: 背景透明的mask
    :param coordinator: 开始融合的坐标，默认为（0,0），从左上角开始
    :return:
    """
    im = img1
    mark = img2
    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    layer.paste(mark, coordinator)
    out = Image.composite(layer, im, layer)
    return out

mask_list = []
for i in range(0, 3):
    mask_list.append(Image.open(r'D:\Code\ziji\SLC_off-dead_pixel-model\cloud\云层' + str(i+1) + '.png'))

#存放所有文件的字典key是目录,value是多个文件
dictFile = defaultdict(list)
# fpathe:表示文件夹地址
# f:表示文件名
# fs:表示图片列表
for fpathe, dirs, fs in os.walk(r"D:\cloud_dataset\moni\cloud_thin_2\A_ref\train"):
    for f in fs:
        dictFile[fpathe+'/'].append(f)

for key in dictFile:
    for value in dictFile[key]:
        # 从mask的list中随机选取一张图片，与原图进行融合
        choosedMask = choice(mask_list)
        img = Image.open(key + value)
        img_mix = mix(img, choosedMask, (0, 0))
        img_mix.save("D:\cloud_dataset\moni\cloud_thin_2\A_miss/train/" + value)

# train:validation:test=8:1:1
# splitfolders.ratio(input='D:/Code/10_25/stgan-master/SLC-off', output='D:/Code/10_25/stgan-master/dataset/ETM', seed=1337, ratio=(0.8, 0.1, 0.1))

