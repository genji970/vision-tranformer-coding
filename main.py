import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from input_processing import data_projection
from Data_loading import data_loader
from multihead_attention import Multi_Head_Attention
from Encoder import Encoder_func

epochs = 4
batch_size = 32
patch_size = 8
cnt = 0
latent_size = patch_size * patch_size * 3

first_stage_cnt = 0
second_stage_cnt = 0
third_stage_cnt = 0

if __name__ == "__main__":
    train_dataloader , test_loader = data_loader() # 데이터 가져오기 train_dataloader , test_dataloader

    for epoch in range(epochs):
        for x , labels in train_dataloader:
            image_projecting = data_projection(x , batch_size , patch_size)

            input = image_projecting.image_to_patches()
            z = image_projecting.preprocessing(input)
            first_stage_cnt += 1
            print(first_stage_cnt)
            # linear_projection의 반환값 : z

            multi_head_attention = Multi_Head_Attention(latent_size)
            attention = multi_head_attention.attention(z)

            second_stage_cnt += 1
            print(second_stage_cnt)
            # attention result
            encoder = Encoder_func(int(latent_size/8))
            encoder(attention)

            third_stage_cnt += 1
            print(third_stage_cnt)

