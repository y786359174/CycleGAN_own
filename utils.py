import random
import time
import datetime
import sys
from torch.autograd import Variable
import torch
import numpy as np
from torchvision.utils import save_image



class ReplayBuffer:
# 用来保存先前生成的样本，缓存区
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):                       ## 放入一张图像，再从buffer里取一张出来
        to_return = []                                  ## 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:          ## 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:          ## 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
    
def kl_reg_loss(mu):    # 正则
    # # 计算隐空间向量的均值和方差
    # latent_mean = torch.mean(latent_vector, dim=0)
    # latent_var = torch.var(latent_vector, dim=0)      #分布式中计算方差会出错，不知道为啥
    
    # # 计算 KL 散度
    # kl_div = 0.5 * (latent_var + latent_mean**2 - 1 - torch.log(latent_var))        # 只有看成正态分布，才可以这么计算
    # kl_div = torch.mean(kl_div)
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    
    return encoding_loss
