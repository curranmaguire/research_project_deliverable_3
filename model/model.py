#https://github.com/cookielee77/DAST/blob/master/network/DAST.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import 

class StyleTransferModel(nn.module):
    def __init__(self,args,vocab):
        super(StyleTransferModel, self).__init__()
        self.args = args
        self.vocab = vocab

    def forward(self, )