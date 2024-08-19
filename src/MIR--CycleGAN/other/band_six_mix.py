import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import defaultdict
from random import choice

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

mask = Image.open(r'D:\Code\ziji\SLC_off-dead_pixel-model\trans_mask_1\mask1.png')

#存放所有文件的字典key是目录,value是多个文件
dictFile = defaultdict(list)
# fpathe:表示文件夹地址
# f:表示文件名
# fs:表示图片列表
for fpathe, dirs, fs in os.walk(r"D:\cloud_dataset\moni\single_image\clear"):
    for f in fs:
        dictFile[fpathe+'/'].append(f)

count = 1

for key in dictFile:
    for value in dictFile[key]:
        print("正在处理第 " + str(count) + "张图片~")
        img = Image.open(key + value)
        # B_real
        img_g = img.convert("L")
        transf_1 = transforms.ToTensor()
        img_tensor_1 = transf_1(img_g)
        # 创建一个转换函数来将图像张量转换为PIL图像
        to_pil_1 = transforms.ToPILImage()
        # 将图像张量转换为PIL图像
        pil_image_1 = to_pil_1(img_tensor_1)
        # 保存PIL图像
        pil_image_1.save(r"D:\cloud_dataset\moni\AquaMODIS\B_real/" + value)

        # 融合图像
        img_mix = mix(img, mask, (0, 0))
        img_gray = img_mix.convert("L")
        transf = transforms.ToTensor()
        img_tensor = transf(img_gray)  # tensor数据格式是torch(C,H,W)
        # 创建一个转换函数来将图像张量转换为PIL图像
        to_pil = transforms.ToPILImage()
        # 将图像张量转换为PIL图像
        pil_image = to_pil(img_tensor)
        # 保存PIL图像
        pil_image.save(r"D:\cloud_dataset\moni\AquaMODIS\A_miss/" + value)
        count = count + 1

