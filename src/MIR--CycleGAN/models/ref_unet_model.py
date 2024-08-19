import torch
from .base_model import BaseModel
from . import networks
import random
import numpy as np
import cv2

class RefUnetModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):  # 修改命令行选项，返回修改后的参数选项
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.  #默认我们使用GAN损失，使用批量归一化的UNet和对齐的数据集
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # 设置默认选项 生成器UNet-256，数据集格式是组合的
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.set_defaults(norm='batch', netG='munet_wo_cbam', dataset_mode='aligned')
        # 打乱输入图像的顺序
        parser.add_argument('--scramble', type=bool, default=False, help='scramble order of input images?')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')  # vanilla表示原始的未经修改的模型
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')  # L1损失的权重

        return parser

    def __init__(self, opt):  # 初始化pix2pix类，存储所有的实验标记，需要baseoptions的子类
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # 基础模型初始化
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # 指定你想要打印的训练损失，训练/测试脚本命名为：BaseModel.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # 指定想要存储或者展示的图像，训练/测试脚本命名为：BaseModel.get_current_visuals
        # real_A_0表示待修复影像，real_A_1表示参考影像，fake_B表示重建影像，real_B表示原始完整影像
        self.visual_names = ['real_A_miss', 'real_A_ref', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # 指定想要存储到磁盘的模型，训练/测试脚本命名为：BaseModel.save_networks、BaseModel.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # 定义网络结构：包含生成器和判别器
        self.netG = networks.define_G(2*opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # 如果是训练过程，定义判别器，条件GAN需要输入两张图像，所以通道数为input_nc+output_nc
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(2*opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # L1损失
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # 初始化优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.generator_epochs = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.  输入包括数据本身以及它的元数据信息

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        self.real_A_miss = input['A_miss'].to(self.device)
        self.real_A_ref = input['A_ref'].to(self.device)

        # 将两张图像拼接成一张
        self.real_A = torch.cat((self.real_A_miss, self.real_A_ref), 1).to(self.device)
        self.real_B = input['B_real'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # 通过生成网络生成修复的影像fake_B
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # 反向传播，计算判别器的GAN损失
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.generator_epochs % 5 == 0:
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        self.generator_epochs = self.generator_epochs + 1
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
