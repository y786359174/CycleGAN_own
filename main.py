import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloader, get_img_shape
from network import GeneratorResNet,  Discriminator
from utils import *


import cv2
import einops
import numpy as np

def train(genG:GeneratorResNet, genF:GeneratorResNet, disG:Discriminator, disF:Discriminator, ckpt_path, device = 'cuda'):
    
    n_epochs = 5000
    batch_size = 1
    lr = 2e-4
    beta1 = 0.5
    k_max = 5

    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_Cycle = torch.nn.L1Loss().to(device)
    criterion_Identity = torch.nn.L1Loss().to(device)

    Imgx_dir_name = '../face2face/dataset/train/C'
    Imgy_dir_name = '../face2face/dataset/train/A'
    dataloaderx = get_dataloader(batch_size,Imgx_dir_name)
    dataloadery = get_dataloader(batch_size,Imgy_dir_name)
    len_x = len(dataloaderx.dataset)
    len_y = len(dataloadery.dataset)

    if(len_x>len_y):
        len_dataset = len_y
    else:
        len_dataset = len_x

    gen_params = list(genG.parameters()) + list(genF.parameters())
    dis_params = list(disG.parameters()) + list(disF.parameters())
    optim_gen = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr, betas=(beta1, 0.999), weight_decay=0.0001)# betas 是一个优化参数，简单来说就是考虑近期的梯度信息的程度
    optim_dis = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                            0.5*lr, betas=(beta1, 0.999), weight_decay=0.0001)

    genG = genG.train()
    genF = genF.train()
    disG = disG.train()
    disF = disF.train()
    label_fake = torch.full((batch_size,1,16,16), 0., dtype=torch.float, device=device)       # 真实图是1，虚假是0,需要注意,这里用的时候是计算loss，是小数，要用1. 和0.
    label_real = torch.full((batch_size,1,16,16), 1., dtype=torch.float, device=device)       # 还有一件事，这里不能随意用batch_size，因为最后可能不满512，但是还是会继续算。得实时读取x的bn大小
                                                                    # 一般四个维度分别是bn c h w
                                                                    # 我把他从循环中挪出来并且检测x的bn不是batch_size就跳过
    xy_buffer = ReplayBuffer()
    yx_buffer = ReplayBuffer()

    k = k_max
    for epoch_i in tqdm(range(n_epochs)):
        tic = time.time()
        if(epoch_i%100==0 and k < k_max):          # 让k缓慢的增大到k_max
            k+=1
        datax_iter = iter(dataloaderx)
        datay_iter = iter(dataloadery)
        for _ in tqdm(range(len_dataset)):
            x,_  = next(datax_iter)
            y,_  = next(datay_iter)
            x = x.to(device)
            y = y.to(device)

            if(x.shape[0]!=batch_size):
                continue
            
            # 训练Gen

            xy = genG(x)
            yx = genF(y)
            g_loss_gan = criterion_GAN(disG(xy), label_real) + criterion_GAN(disF(yx), label_real)

            xyx = genF(xy)
            yxy = genG(yx)
            g_loss_cycle = criterion_Cycle(xyx, x) + criterion_Cycle(yxy, y)

            xx = genF(x)
            yy = genG(y)
            g_loss_identity = criterion_Identity(xx, x) + criterion_Identity(yy, y)

            g_loss = g_loss_gan * 100 + g_loss_cycle*10 + g_loss_identity*5
            optim_gen.zero_grad()
            g_loss.backward()
            optim_gen.step()

            # 训练Dis
            xy_ = xy_buffer.push_and_pop(xy).to(device)
            yx_ = yx_buffer.push_and_pop(yx).to(device)
            d_loss_G_real = criterion_GAN(disG(y), label_real)
            d_loss_G_fake = criterion_GAN(disG(xy_), label_fake)
            d_loss_F_real = criterion_GAN(disF(x), label_real)
            d_loss_F_fake = criterion_GAN(disF(yx_), label_fake)

            d_loss = d_loss_G_real + d_loss_G_fake + d_loss_F_real + d_loss_F_fake
            optim_dis.zero_grad()
            d_loss.backward()
            optim_dis.step()
            
            
        toc = time.time()    
        gan_weights = {'genG': genG.state_dict(), 'genF': genF.state_dict(), 'disG': disG.state_dict(), 'disF': disF.state_dict()}
        torch.save(gan_weights, ckpt_path)
        print(f'epoch {epoch_i} g_loss: {g_loss.item():.4e} d_loss: {d_loss.item():.4e} time: {(toc - tic):.2f}s')
        print(f'g_loss_gan {g_loss_gan.item():.4e} g_loss_cycle: {10*g_loss_cycle.item():.4e} g_loss_identity: {5*g_loss_identity.item():.4e} ')
        print(f'd_loss_G_real {d_loss_G_real.item():.4e} d_loss_G_fake: {d_loss_G_fake.item():.4e} d_loss_F_real: {d_loss_F_real.item():.4e} d_loss_F_fake: {d_loss_F_fake.item():.4e}')
        if(epoch_i%10==0):
            sample(genG, genF, device=device)

sample_time = 0

def sample(genG:GeneratorResNet, genF:GeneratorResNet, device='cuda'):
    global sample_time
    sample_time += 1
    i_n = 10
    # for i in range(i_n*i_n):
    genG = genG.to(device)
    genG = genG.eval()
    genF = genF.to(device)
    genF = genF.eval()
    with torch.no_grad():
        Imgx_dir_name = '../face2face/dataset/train/C'
        dataloaderx = get_dataloader(i_n, Imgx_dir_name)
        x,_  = next(iter(dataloaderx))
        x = x.to(device)
        xy = genG(x)
        xyx = genF(xy)
        x_stack = torch.stack((x, xy, xyx), dim = 0)
        x_new = einops.rearrange(x_stack, 'x n c h w -> (x h) (n w) c')
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite('cyclegan_sample_%d.jpg' % (sample_time), x_new)
    genG = genG.train()
    genF = genF.train()

## 定义参数初始化函数
def weights_init_normal(m):                                    
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字. 
    if classname.find("Conv") != -1:                        ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.


if __name__ == '__main__':

    ckpt_path = './model_cyclegan.pth'
    device = 'cuda'
    image_shape = get_img_shape()
    genG = GeneratorResNet(image_shape, 9).to(device)    # x->y
    genF = GeneratorResNet(image_shape, 9).to(device)    # y->x
    disG = Discriminator(image_shape).to(device)
    disF = Discriminator(image_shape).to(device)
    genG.apply(weights_init_normal)
    genF.apply(weights_init_normal)
    disG.apply(weights_init_normal)
    disF.apply(weights_init_normal)
    # sample(genG, genF, device=device)
    train(genG, genF, disG, disF, ckpt_path, device=device)
    gan_weights = torch.load(ckpt_path)
    genG.load_state_dict(gan_weights['genG'])
    genF.load_state_dict(gan_weights['genF'])
    disG.load_state_dict(gan_weights['disG'])
    disF.load_state_dict(gan_weights['disF'])

    sample(genG, genF, device=device)
