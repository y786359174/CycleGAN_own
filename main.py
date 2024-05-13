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
    batch_size = 256
    lr = 1e-4
    beta1 = 0.5
    # k_max = 1

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
                                            lr*0.1, betas=(beta1, 0.999), weight_decay=0.0001)

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

    # k = k_max
    for epoch_i in tqdm(range(n_epochs)):
        tic = time.time()
        # if(epoch_i%100==0 and k < k_max):          # 让k缓慢的增大到k_max
        #     k+=1
        datax_iter = iter(dataloaderx)
        datay_iter = iter(dataloadery)
        for _ in range(len_dataset):
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

            g_loss = g_loss_gan + g_loss_cycle*10 + g_loss_identity*5
            optim_gen.zero_grad()
            g_loss.backward()
            optim_gen.step()

            # 训练Dis
            xy_ = xy_buffer.push_and_pop(xy).to(device)
            yx_ = yx_buffer.push_and_pop(yx).to(device)
            d_loss_G = criterion_GAN(disG(xy_), label_fake) + criterion_GAN(disG(y), label_real)
            d_loss_F = criterion_GAN(disF(yx_), label_fake) + criterion_GAN(disF(x), label_real)

            d_loss = d_loss_G + d_loss_F
            optim_dis.zero_grad()
            d_loss.backward()
            optim_dis.step()
            
            
        toc = time.time()    
        if(epoch_i%20==0):
            sample(genG, device=device)
            print(f'epoch {epoch_i} g_loss: {g_loss.item():.4e} d_loss: {d_loss.item():.4e} time: {(toc - tic):.2f}s')
        gan_weights = {'genG': genG.state_dict(), 'genF': genF.state_dict(), 'disG': disG.state_dict(), 'disF': disF.state_dict()}
        torch.save(gan_weights, ckpt_path)


sample_time = 0

def sample(gen:GeneratorResNet, device='cuda'):
    global sample_time
    sample_time += 1
    i_n = 10
    # for i in range(i_n*i_n):
    gen = gen.to(device)
    gen = gen.eval()
    with torch.no_grad():
        Imgx_dir_name = '../face2face/dataset/train/C'
        dataloaderx = get_dataloader(i_n, Imgx_dir_name)
        x,_  = next(iter(dataloaderx))
        x = x.to(device)
        xy = gen(x)

        x_new = einops.rearrange(torch.stack((x, xy), dim=-1), 'x n c h w -> (x h) (n w) c', n1 = i_n)
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite('cyclegan_sample_%d.jpg' % (sample_time), x_new)
    gen = gen.train()

def set_layer_init_weight(layer):
    if 'Conv' in layer.__class__.__name__:
        nn.init.normal_(layer.weight.data, 0, 0.02)
    elif 'BatchNorm' in layer.__class__.__name__:
        nn.init.normal_(layer.weight.data, 1, 0.02)
        nn.init.constant_(layer.bias.data, 0)

if __name__ == '__main__':

    ckpt_path = './model_cyclegan.pth'
    device = 'cpu'
    image_shape = get_img_shape()
    genG = GeneratorResNet(image_shape, 9).to(device)    # x->y
    genF = GeneratorResNet(image_shape, 9).to(device)    # y->x
    disG = Discriminator(image_shape).to(device)
    disF = Discriminator(image_shape).to(device)
    genG.apply(set_layer_init_weight)
    genF.apply(set_layer_init_weight)
    disG.apply(set_layer_init_weight)
    disF.apply(set_layer_init_weight)
    train(genG, genF, disG, disF, ckpt_path, device=device)
    gan_weights = torch.load(ckpt_path)
    genG.load_state_dict(gan_weights['genG'])
    genF.load_state_dict(gan_weights['genF'])
    disG.load_state_dict(gan_weights['disG'])
    disF.load_state_dict(gan_weights['disF'])

    sample(genG, device=device)
