import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from data_manager import TrainDataset
from models.gen.SPANet import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport


def train(config):
    gpu_manage(config)  # 配置 GPU 设置

    ### DATASET LOAD ###
    print('===> Loading datasets')

    dataset = TrainDataset(config)  # 加载训练数据集
    print('dataset:', len(dataset))  # 打印数据集的总长度
    train_size = int((1 - config.validation_size) * len(dataset))  # 计算训练集大小
    validation_size = len(dataset) - train_size  # 计算验证集大小
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])  # 划分训练集和验证集
    print('train dataset:', len(train_dataset))  # 打印训练集大小
    print('validation dataset:', len(validation_dataset))  # 打印验证集大小
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)  # 加载训练集数据
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)  # 加载验证集数据
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)  # 初始化生成器模型

    if config.gen_init is not None:
        param = torch.load(config.gen_init)  # 加载预训练生成器模型参数
        gen.load_state_dict(param)  # 加载预训练模型参数到生成器
        print('load {} as pretrained model'.format(config.gen_init))  # 打印加载的预训练模型信息

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)  # 初始化判别器模型

    if config.dis_init is not None:
        param = torch.load(config.dis_init)  # 加载预训练判别器模型参数
        dis.load_state_dict(param)  # 加载预训练模型参数到判别器
        print('load {} as pretrained model'.format(config.dis_init))  # 打印加载的预训练模型信息

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)  # 配置生成器的优化器
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)  # 配置判别器的优化器

    real_a = torch.FloatTensor(config.batchsize, config.in_ch, config.width, config.height)  # 定义输入图像的张量
    real_b = torch.FloatTensor(config.batchsize, config.out_ch, config.width, config.height)  # 定义目标图像的张量
    M = torch.FloatTensor(config.batchsize, config.width, config.height)  # 定义掩码张量

    criterionL1 = nn.L1Loss()  # 定义 L1 损失
    criterionMSE = nn.MSELoss()  # 定义均方误差损失
    criterionSoftplus = nn.Softplus()  # 定义 Softplus 损失

    if config.cuda:
        gen = gen.cuda()  # 将生成器模型移到 GPU
        dis = dis.cuda()  # 将判别器模型移到 GPU
        criterionL1 = criterionL1.cuda()  # 将 L1 损失移到 GPU
        criterionMSE = criterionMSE.cuda()  # 将 MSE 损失移到 GPU
        criterionSoftplus = criterionSoftplus.cuda()  # 将 Softplus 损失移到 GPU
        real_a = real_a.cuda()  # 将输入图像张量移到 GPU
        real_b = real_b.cuda()  # 将目标图像张量移到 GPU
        M = M.cuda()  # 将掩码张量移到 GPU

    real_a = Variable(real_a)  # 将输入图像张量转换为 Variable 类型
    real_b = Variable(real_b)  # 将目标图像张量转换为 Variable 类型

    logreport = LogReport(log_dir=config.out_dir)  # 初始化日志报告
    validationreport = TestReport(log_dir=config.out_dir)  # 初始化验证报告

    print('===> begin')
    start_time = time.time()  # 记录训练开始时间

    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()  # 记录每个 epoch 的开始时间
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]  # 从批次数据中获取输入图像、目标图像和掩码
            with torch.no_grad():  # 禁用梯度计算
                real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)  # 将输入图像数据复制到 GPU 张量
                real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)  # 将目标图像数据复制到 GPU 张量
                M.resize_(M_cpu.size()).copy_(M_cpu)  # 将掩码数据复制到 GPU 张量

            # 前向传播，生成假图像和注意力图
            att, fake_b = gen.forward(real_a)

            ################
            ### Update D ###
            ################
            
            opt_dis.zero_grad()  # 清空判别器的梯度

            # 使用假图像更新判别器
            fake_ab = torch.cat((real_a, fake_b), 1)  # 拼接输入图像和生成的假图像
            pred_fake = dis.forward(fake_ab.detach())  # 计算假图像的判别结果
            batchsize, _, w, h = pred_fake.size()

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h  # 计算判别器对假图像的损失

            # 使用真实图像更新判别器
            real_ab = torch.cat((real_a, real_b), 1)  # 拼接输入图像和真实目标图像
            pred_real = dis.forward(real_ab)  # 计算真实图像的判别结果
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h  # 计算判别器对真实图像的损失

            # 计算判别器总损失
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()  # 反向传播判别器损失

            if epoch % config.minimax == 0:
                opt_dis.step()  # 更新判别器参数

            ################
            ### Update G ###
            ################
            
            opt_gen.zero_grad()  # 清空生成器的梯度

            # 首先，生成器生成的假图像应欺骗判别器
            fake_ab = torch.cat((real_a, fake_b), 1)  # 拼接输入图像和生成的假图像
            pred_fake = dis.forward(fake_ab)  # 计算假图像的判别结果
            loss_g_gan = torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h  # 计算生成器的 GAN 损失

            # 其次，生成的假图像应与真实图像相匹配
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb  # 计算生成器的 L1 损失
            loss_g_att = criterionMSE(att[:,0,:,:], M)  # 计算生成器的注意力损失
            loss_g = loss_g_gan + loss_g_l1 + loss_g_att  # 计算生成器的总损失

            loss_g.backward()  # 反向传播生成器损失

            opt_gen.step()  # 更新生成器参数

            # log
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d_fake.item(), loss_d_real.item(), loss_g_gan.item(), loss_g_l1.item()))
                
                log = {}
                log['epoch'] = epoch  # 记录当前 epoch
                log['iteration'] = len(training_data_loader) * (epoch-1) + iteration  # 记录当前迭代次数
                log['gen/loss'] = loss_g.item()  # 记录生成
                log['dis/loss'] = loss_d.item() # 记录判别器的总损失
 
                logreport(log) # 更新日志报告

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        
        print('Saving model at epoch {}...'.format(epoch))
        if epoch % config.snapshot_interval == 0:
            checkpoint(config, epoch, gen, dis)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)
