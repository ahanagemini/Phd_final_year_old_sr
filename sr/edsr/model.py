from . import common

import torch
import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, aspp=False,
                 dilation=False, act='relu', conv=common.default_conv):
        super(EDSR, self).__init__()

        # n_resblocks = 16 * 2
        # n_feats = 64 * 4
        kernel_size = 3
        if act == "relu":
            act = nn.ReLU(True)
        elif act == "leakyrelu":
            act = nn.LeakyReLU(0.1)
        self.dilation = dilation
        self.aspp = aspp
        #self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]
        #self.sub_mean = common.MeanShift(1)
        #self.add_mean = common.MeanShift(1, sign=1)

        # define head module
        if self.dilation:
            m_head = [conv(1, n_feats, kernel_size, dilation=1)]
            m_head1 = [conv(n_feats, n_feats, kernel_size, dilation=2)]
            m_head2 = [conv(n_feats, n_feats, kernel_size, dilation=4)]
        else:
            m_head = [conv(1, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        tail_feats = n_feats
        if self.aspp:
            m_aspp1 = [conv(n_feats, n_feats, kernel_size, dilation=1)]
            m_aspp2 = [conv(n_feats, n_feats, kernel_size, dilation=2)]
            m_aspp3 = [conv(n_feats, n_feats, kernel_size, dilation=4)]
            tail_feats = n_feats * 3

        m_tail = [
            common.Upsampler(conv, scale, tail_feats, act=False),
            common.ResBlock(conv, tail_feats, kernel_size, act=act),
            conv(tail_feats, 1, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        if self.dilation:
            self.head1 = nn.Sequential(*m_head1)
            self.head2 = nn.Sequential(*m_head2)
        self.body = nn.Sequential(*m_body)
        if self.aspp:
            self.aspp1 = nn.Sequential(*m_aspp1)
            self.aspp2 = nn.Sequential(*m_aspp2)
            self.aspp3 = nn.Sequential(*m_aspp3)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)
        if self.dilation:
            x = self.head1(x)
            x = self.head2(x)
        res = self.body(x)
        res += x
        if self.aspp:
            aspp1 = self.aspp1(res)
            aspp2 = self.aspp2(res)
            aspp3 = self.aspp3(res)
            res = torch.cat((aspp1, aspp2, aspp3), dim=1)
        x = self.tail(res)
        #x = self.add_mean(x)

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
