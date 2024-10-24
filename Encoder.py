import torch.nn as nn

class Encoder_func(nn.Module):
    def __init__(self , latent_dim):
        # x.shape : batch_num , head_num=8, N + 1, D / head_num
        super().__init__()
        self.l1 = nn.Linear(latent_dim , latent_dim)
        self.l2 = nn.Linear(latent_dim , latent_dim)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(latent_dim)

    def forward(self , x):

        x = self.layernorm(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layernorm(x)
        x = self.l2(x)
        x = self.relu(x)

        return x