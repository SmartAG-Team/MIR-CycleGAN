import torch.nn as nn
import torch

class MultiSourceInfoFusion(nn.Module):
    def __init__(self, input_nc):
        super(MultiSourceInfoFusion, self).__init__()
        self.input_nc = input_nc
        if input_nc == 1:
            input_nc = 2
        self.fusion = nn.Conv2d(input_nc // 2, 30, kernel_size=3, stride=1, padding=1)
        # self.fusion = nn.Conv2d(input_nc, 30, kernel_size=3, stride=1, padding=1)
        # self.fusion = nn.Conv2d(input_nc, 60, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        # 该x表示real_A0和real_A1拼接在一起
        # 前一半通道表示real_A0，后一半通道表示real_A1
        # x.shape [1,3,256,256] y.shape [1,3,256,256]
        x = self.fusion(x)
        y = self.fusion(y)
        # input.shape [1,60,256,256]
        input = torch.cat((x, y), dim=1)
        return input

# 测试使用几次迭代
class MultiSourceInfoFusion_1(nn.Module):
    def __init__(self, input_nc, count):
        super(MultiSourceInfoFusion_1, self).__init__()
        self.input_nc = input_nc
        self.count = count
        # print(input_nc)
        if input_nc == 1:
            input_nc = 2
        self.fusion = nn.Conv2d(input_nc // 2, 30, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv2d(60, input_nc // 2, kernel_size=3, stride=1, padding=1)


    def forward(self, x, y):
        # 该x表示real_A0和real_A1拼接在一起
        # 前一半通道表示real_A0，后一半通道表示real_A1
        # x.shape [1,3,256,256] y.shape [1,3,256,256]
        if self.count == 1:
            x = self.fusion(x)
            y = self.fusion(y)
            input = torch.cat((x, y), dim=1)
        else:
            for i in range(0, self.count):
                # print(x.shape)
                x_1 = self.fusion(x)
                y_1 = self.fusion(y)
                input_1 = torch.cat((x_1, y_1), dim=1)
                input_2 = self.downsample(input_1)
                x = input_2
            input = torch.cat((x_1, y_1), dim=1)

        # input.shape [1,60,256,256]
        return input

# 不使用下采样
class MultiSourceInfoFusion_2(nn.Module):
    def __init__(self, input_nc, count):
        super(MultiSourceInfoFusion_2, self).__init__()
        self.input_nc = input_nc
        self.count = count
        # print(input_nc)
        if input_nc == 1:
            input_nc = 2
        self.fusion_1 = nn.Sequential(nn.Conv2d(input_nc // 2, 30, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(30), nn.PReLU())
        self.fusion_2 = nn.Sequential(nn.Conv2d(60, 30, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(30), nn.PReLU())
        # self.fusion_1 = nn.Conv2d(input_nc // 2, 30, kernel_size=3, stride=1, padding=1)
        # self.fusion_2 = nn.Conv2d(60, 30, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv2d(60, input_nc // 2, kernel_size=3, stride=1, padding=1)


    def forward(self, x, y):
        # 该x表示real_A0和real_A1拼接在一起
        # 前一半通道表示real_A0，后一半通道表示real_A1
        # x.shape [1,3,256,256] y.shape [1,3,256,256]

        x_1 = self.fusion_1(x)
        y = self.fusion_1(y)
        # channels=60
        input_1 = torch.cat((x_1, y), dim=1)
        x_2 = self.fusion_2(input_1)
        input_2 = torch.cat((x_2, y), dim=1)
        x_3 = self.fusion_2(input_2)
        input = torch.cat((x_3, y), dim=1)
        # x_4 = self.fusion_2(input)
        # input_4 = torch.cat((x_4, y), dim=1)
        # x_5 = self.fusion_2(input_4)
        # input_5 = torch.cat((x_5, y), dim=1)

        # input.shape [1,60,256,256]
        # return input, input_1, input_2, input_4, input_5
        return input

