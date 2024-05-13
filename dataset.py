import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Lambda, ToTensor, Pad, Resize
import einops
import numpy as np
import cv2

import os
import pandas as pd
from tqdm import tqdm
# def get_dataloader(batch_size: int):
#     transform = Compose([ToTensor(), 
#                          Pad(padding=2, fill=0),
#                          Lambda(lambda x: (x - 0.5) * 2)
#                          ])
#     dataset = torchvision.datasets.MNIST(root='./data/mnist',
#                                          transform=transform, download=True)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# def get_img_shape():
#     return (1, 32, 32)

# def get_dataloader(batch_size: int):
#     transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
#     data_dir = '../faces'
#     dataset = torchvision.datasets.ImageFolder(root=data_dir,
#                                          transform=transform)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# def get_img_shape():
#     return (3, 96, 96)


def get_dataloader(batch_size: int, data_dir):
    transform = Compose([ToTensor(), 
                        #  Resize(256),
                        #  transforms.CenterCrop(256),
                        #  Resize(32),
                         Lambda(lambda x: (x - 0.5) * 2)
                         ])
    dataset = torchvision.datasets.ImageFolder(root=data_dir,
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_img_shape():
    return (3, 256, 256)    # 虽然这么写很蠢。但是好像还真挺好用
    # return (3, 32, 32)    # 虽然这么写很蠢。但是好像还真挺好用


def module_test():
    batch_size = 64
    dataloader = get_dataloader(batch_size)
    data_iter = iter(dataloader)                # 测试，只提取一次dataloader，把它放入迭代器，next读取一个
    x,_ = next(data_iter)
    image_shape = x.shape
    print("x.shape =",image_shape)
    x = (x + 1) / 2 * 255
    x = x.clamp(0, 255)
    x_new = einops.rearrange(x,                                     # 新库，对张量维度重新变化
                            '(b1 b2) c h w -> (b1 h) (b2 w) c',     # 还可以自己分解和重组，括号里的就是分解/重组的部分
                            b1=int(image_shape[0]**0.5))
    x_new = x_new.numpy().astype(np.uint8)
    print(x_new.shape)
    if(x_new.shape[2]==3):
        x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
    cv2.imwrite('minst88.jpg', x_new)
    

def Img_process():          # 我现在的图片大小都不一样，我要重新裁剪一下看看效果，正常train时不需要重新裁剪，transformer就能直接处理
    transform = Compose([ToTensor(), 
                         Resize(256),
                         transforms.CenterCrop(256)
                         ])
    data_dir = '../face2face/dataset/train/B'
    dataset = torchvision.datasets.ImageFolder(root=data_dir,
                                         transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
    save_dir = '../face2face/dataset/train/C/C'  # 指定保存图像的目录
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

    for i, (x, _) in tqdm(enumerate(dataloader)):
        image = transforms.ToPILImage()(x.squeeze())  # 将张量转换为PIL图像
        save_path = os.path.join(save_dir, f"image_{i}.jpg")  # 构建保存路径
        
        os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
        
        image.save(save_path)  # 保存图像到指定路径
        

if __name__ == '__main__':
    # module_test()
    Img_process()