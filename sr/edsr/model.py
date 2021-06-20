from . import common

import torch
import torch.nn as nn


class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, aspp=False,
                 dilation=False, act='relu', conv=common.default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        if act == "relu":
            act = nn.ReLU(True)
        elif act == "leakyrelu":
            act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == "elu":
            act = nn.ELU(alpha=1.0)
        elif act == "prelu":
            act = nn.PReLU()
        self.dilation = dilation

        # define head module
        m_head = [conv(1, n_feats, kernel_size, dilation=1)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1, dilation=self.dilation
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        tail_feats = n_feats

        m_tail = [
            common.Upsampler(conv, scale, tail_feats, act=False),
            conv(tail_feats, 1, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
