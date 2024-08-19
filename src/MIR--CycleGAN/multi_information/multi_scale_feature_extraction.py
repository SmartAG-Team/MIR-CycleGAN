import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Softmax
# from torch.nn.functional import conv2d as Conv2d

class ASPP(nn.Module): # deeplab

    def __init__(self, dim, in_dim):
        super(ASPP, self).__init__()

        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        # down_dim = in_dim // 2
        down_dim = 1

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=1, padding=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=5, padding=5), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim),  nn.PReLU())
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU())

    def forward(self, x):
        x = self.down_conv(x)
        # print("down_conv:", x.shape)
        # [1,3,256,256]
        conv1 = self.conv1(x)
        # print("conv1:", conv1.shape)
        # [1,1,256,256]
        conv2 = self.conv2(x)
        # print("conv2:", conv2.shape)
        # [1,1,256,256]
        conv3 = self.conv3(x)
        # print("conv3:", conv3.shape)
        # [1,1,256,256]
        conv4 = self.conv4(x)
        # print("conv4:", conv4.shape)
        # [1,1,256,256]
        # print("自适应平均池化：", F.adaptive_avg_pool2d(x, 1).shape)
        # [1,3,1,1]
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 128)), size=x.size()[2:], mode='bilinear', align_corners=True)
        # print("conv5:", conv5.shape)
        # [1,1,256,256]
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
        # [1,3,256,256]

class MSFE(nn.Module): # deeplab

    def __init__(self, dim, in_dim):
        super(MSFE, self).__init__()
        # dim=60  in_dim=60
        self.input_nc = dim

        # self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        in_dim = dim // 3
        down_dim = dim // 3

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=5, padding=5), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=7, padding=7), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim),  nn.PReLU())
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU())

    def forward(self, x):
        input_x, input_y, input_z = x.split(self.input_nc // 3, dim=1)
        conv2 = self.conv2(input_x)
        conv3 = self.conv3(input_y)
        conv4 = self.conv4(input_z)
        return torch.cat((conv2, conv3, conv4), 1)


class PAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(PAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2
        # print("down_dim: ", down_dim)
        # print("down_dim // 8: ", down_dim // 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        # self.query_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        # self.key_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        # self.value_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)

        # 先注释掉
        # self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=1, kernel_size=1)
        # self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=1, kernel_size=1)
        # self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        # self.gamma2 = Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        # self.query_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        # self.key_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        # self.value_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)

        # 先注释掉
        # self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=1, kernel_size=1)
        # self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=1, kernel_size=1)
        # self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        # self.gamma3 = Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        # self.query_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        # self.key_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        # self.value_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)

        # 先注释掉
        # self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=1, kernel_size=1)
        # self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=1, kernel_size=1)
        # self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        # self.gamma4 = Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim),  nn.PReLU()  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        # print("conv1 shape: ", conv1.shape)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        # print("m_batch_size: ", m_batchsize)
        # print("conv2 shape: ", conv2.shape)
        # print(self.query_conv2(conv2).shape)
        # print(self.query_conv2(conv2).view(m_batchsize, -1, width * height).shape)
        # print(self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1).shape)
        # [1,1,256,256] [1,1,65536] [1,65536,1]

        # 先注释掉
        # proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        # energy2 = torch.bmm(proj_query2, proj_key2)
        # attention2 = self.softmax(energy2)
        # proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        # out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        # out2 = out2.view(m_batchsize, C, height, width)
        # out2 = self.gamma2 * out2 + conv2

        conv3 = self.conv3(x)

        # 先注释掉
        # m_batchsize, C, height, width = conv3.size()
        # proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        # energy3 = torch.bmm(proj_query3, proj_key3)
        # attention3 = self.softmax(energy3)
        # proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        # out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        # out3 = out3.view(m_batchsize, C, height, width)
        # out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)

        # 先注释掉
        # m_batchsize, C, height, width = conv4.size()
        # proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        # energy4 = torch.bmm(proj_query4, proj_key4)
        # attention4 = self.softmax(energy4)
        # proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        # out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        # out4 = out4.view(m_batchsize, C, height, width)
        # out4 = self.gamma4 * out4 + conv4
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 128)), size=x.size()[2:], mode='bilinear',
                              align_corners=True)

        # conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True) # 如果batch设为1，这里就会有问题。

        # conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], align_corners=False)

        # 先注释掉
        # return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))



