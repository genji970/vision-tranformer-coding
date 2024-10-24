import torch
import torch.nn as nn

class data_projection(nn.Module):
    def __init__(self, image , batch_num , patch_size,channel_num=3):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1 , 32*32))
        self.channel = channel_num
        self.image = image
        self.B , self.C , self.H, self.W = image.shape
        self.patch_size = patch_size
        self.cls_tocken = nn.Parameter(torch.randn(1,1,patch_size*patch_size*channel_num))
        self.positional_embedding = nn.Parameter(torch.randn(1 , int((image.shape[2] * image.shape[3])/(patch_size*patch_size)) + 1 , patch_size*patch_size*channel_num))

    def image_to_patches(self):
        # 패치로 나누기
        patches = self.image.unfold(2, self.patch_size, self.patch_size)  # Height 방향 패치 생성
        patches = patches.unfold(3, self.patch_size, self.patch_size)  # Width 방향 패치 생성
        # B , C , H / Patch_size , W / Patch_size , patch_size , patch_size
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # B , H/Patch_size , W/Patch_size , C , patch_size , patch_size
        patches = patches.view(self.B , int(self.H/self.patch_size) , int(self.W/self.patch_size) , self.C * self.patch_size * self.patch_size)
        patches = patches.view(self.B , int((self.H/self.patch_size)*(self.W/self.patch_size)) , -1)

        return patches # shape : B , N , 1차원 벡터화 = patch_size^2 * channel_num

    def preprocessing(self,x): # x는 image_to_patches의 output
        cls_tocken = self.cls_tocken.expand(x.shape[0] , 1 , self.patch_size*self.patch_size*self.channel)
        x = torch.cat([cls_tocken , x],dim = 1) # x.shape : B , N+1 , patch_size^2 * channel_num
        # positional_embedding.shape : 1 , N + 1 , patch_size^2 * channel_num
        positional_embedding = self.positional_embedding.expand(self.B,int((self.image.shape[2] * self.image.shape[3])/(self.patch_size*self.patch_size)) + 1 , self.patch_size*self.patch_size*self.channel)
        return x + positional_embedding
        # batch_num , N + 1 , patch_size^2*channel_num=3