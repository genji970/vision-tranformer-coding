import torch
import torch.nn as nn

class Multi_Head_Attention(nn.Module):
    def __init__(self , latent_dim , head = 8):
        super().__init__()
        self.head = head
        self.key = nn.Linear(int(latent_dim / head), int(latent_dim / head))
        self.query = nn.Linear(int(latent_dim / head) , int(latent_dim / head))
        self.value = nn.Linear(int(latent_dim / head) , int(latent_dim / head))

        self.softmax = nn.Softmax(dim= -1)
    def attention(self, x):
        B, N_prime, embedding_size = x.shape
        x = x.view(B, N_prime, int(embedding_size/self.head) , self.head).permute(0,3,1,2) # batch_soze , self.head=8 , N_prime , embedding_size/head

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_score = Q @ K.permute(0,1,3,2)
        attention_score = attention_score / torch.sqrt(torch.tensor(embedding_size))
        attention_score = self.softmax(attention_score)
        attention = attention_score @ V
        return attention



