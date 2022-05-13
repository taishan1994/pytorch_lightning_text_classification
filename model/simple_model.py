# coding=utf-8
"""
这里存放的是模型，继承原始的nn.module
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Bilstm(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 biFlag,
                 device=None,
                 dropout=0.5):
        # input_dim 输入特征维度d_input
        # hidden_dim 隐藏层的大小
        # output_dim 输出层的大小（分类的类别数）
        # num_layers LSTM隐藏层的层数
        # biFlag 是否使用双向
        super(Bilstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if (biFlag):
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = biFlag
        # 根据需要修改device
        # self.device = device

        # 定义LSTM网络的输入，输出，层数，是否batch_first，dropout比例，是否双向
        self.layer1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout, bidirectional=biFlag)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim * self.bi_num, output_dim),
        )

        # self.to(self.device)

    def init_hidden(self, batch_size):
        # 定义初始的hidden state
        # return (torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device),
        #         torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device))
        return (torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim))


    def forward(self, x, length):
        # 输入原始数据x，标签y，以及长度length
        # 准备
        batch_size = x.size(0)
        # pack sequence
        x = pack_padded_sequence(x, length, batch_first=True)

        # run the network
        hidden1 = self.init_hidden(batch_size)
        out, hidden1 = self.layer1(x, hidden1)
        # out,_=self.layerLSTM(x) is also ok if you don't want to refer to hidden state
        # unpack sequence
        out, length = pad_packed_sequence(out, batch_first=True)
        out = torch.sum(out, dim=1)
        out = self.layer2(out)
        # 返回正确的标签，预测标签，以及长度向量
        return out, length


class SimpleModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 biFlag,
                 device,
                 dropout=0.5
                 ):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.bilstm = Bilstm(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            biFlag,
            device,
            dropout
        )
        self.to(device)

    def forward(self, train_x, length):
        train_x = self.embedding(train_x)
        out, out_length = self.bilstm(train_x, length)
        return out, out_length