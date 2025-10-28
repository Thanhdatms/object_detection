import numpy as np
import torch.nn as nn
import torch


# x: input tensor vi du (4,512,38,38)
# out: la tensor sau khi chuan hoa va scale co gia tri la 20
# out*x: tensor sau khi chuan hoa tung vector theo chieu channels va nhan voi weight scale

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__() # khoi tao de ke thua cac func trong nn.Module
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.esp = 1e-10
    
    def reset_parameters(self): # reset_parameters la 1 func co san tron _ConvNd(Module) base cua tat ca class Conv dung de khoi tao lai weight
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        # x.size() = (batch_size, channels, heights, widths) (0,1,2,3)

        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.esp
        x = torch.div(x, norm) # chuan hoa tung vector theo chieu channels
        
        # luc nay weight co dang (512,) can chuyen ve (1,512,1,1) de thuc hien phep nhan
        # tao out co dang (batch_size, channels, heights, widths)
        # expand_as de copy gia tri trong weight ra toan bo out
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return out*x


