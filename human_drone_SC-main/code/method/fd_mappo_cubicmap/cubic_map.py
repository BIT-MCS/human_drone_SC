from .conf import *


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class CubicMap(torch.nn.Module):
    def __init__(self):
        super(CubicMap, self).__init__()
        self.M_size_x = CONF['M_size'][1]
        self.M_size_y = CONF['M_size'][2]
        self.M_feature = CONF['M_size'][0]
        self.r_feature = CONF['M_size'][0] * 2
        self.c_feature = CONF['M_size'][0] * 2
        self.w_feature = CONF['M_size'][0]
        self.x_shape = [CONF['M_size'][0] * 2, CONF['mtx_size'], CONF['mtx_size']]

        init_ = lambda m: init(m,
                               torch.nn.init.orthogonal_,
                               lambda x: torch.nn.init.constant_(x, 0),
                               torch.nn.init.calculate_gain('relu'))

        self.read_cnn_layer = nn.Sequential(
            init_(torch.nn.Conv2d(
                in_channels=self.M_feature,
                out_channels=self.M_feature * 2,
                kernel_size=4,
                stride=2,
            )),
            nn.ReLU(),
            init_(torch.nn.Conv2d(
                in_channels=self.M_feature * 2,
                out_channels=self.r_feature,
                kernel_size=4,
                stride=2,
                padding=1,
            )),
            nn.ReLU(),
        )
        self.write_trans_layer = nn.Sequential(
            init_(nn.Conv2d(self.x_shape[0], self.w_feature,
                            kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            init_(nn.Conv2d(self.r_feature + self.x_shape[0] + self.r_feature, CONF['hidden_size'], kernel_size=3)),
            nn.ReLU(),
        )
        self.q_trans_layer = nn.Sequential(
            init_(nn.Conv2d(self.x_shape[0] + self.M_feature * 2, self.M_feature,
                            kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        )
        self.context_read_cnn_layer = nn.Sequential(
            init_(torch.nn.Conv2d(
                in_channels=self.M_feature,
                out_channels=self.M_feature * 2,
                kernel_size=4,
                stride=2,
            )),
            torch.nn.ReLU(),
            init_(torch.nn.Conv2d(
                in_channels=self.M_feature * 2,
                out_channels=self.c_feature,
                kernel_size=4,
                stride=2,
                padding=1,
            )),
            torch.nn.ReLU(),
        )
        self.w_i_r = init_(
            nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1)))
        self.w_h_r = init_(
            nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1)))

        self.w_i_z = init_(
            nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1)))
        self.w_h_z = init_(
            nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1)))

        self.w_i_n = init_(
            nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1)))
        self.w_h_n = init_(
            nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1)))

    def read(self, M):
        r = self.read_cnn_layer(M)
        return r

    def write(self, x, M, p_msk, n_msk):
        dim_len = x.shape[1]

        B, N = x.shape[:2]
        x = self.write_trans_layer(x.view(B * N, *x.shape[2:]))
        x = x.view(B, N, *x.shape[1:])
        x = x.view(*x.shape[:-2], -1).permute([0, 1, 3, 2])
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        p_msk_x = x * p_msk[:, :dim_len]
        p_msk_x_sum = torch.sum(p_msk_x, dim=2)
        p_msk_x_sum = torch.sum(p_msk_x_sum, dim=1)

        su_p_msk = -1 * n_msk + 1
        su_old_x = su_p_msk * M
        su_h = self.soft_update(p_msk_x_sum, su_old_x)
        su_new_x = su_p_msk * su_h

        n_msk_M = n_msk * M
        new_M = n_msk_M + su_new_x
        return new_M

    def cal_output(self, x, r, c):
        o = self.output_layer(torch.cat([x, r, c], dim=1))
        o = o.squeeze(-1).squeeze(-1)
        return o

    def match_corr(self, embed_ref, embed_srch):
        b, c, h, w = embed_srch.shape
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w),
                             embed_ref, groups=b, padding=1)
        match_map = match_map.permute(1, 0, 2, 3)
        return match_map

    def context_read(self, x, r, M):
        q = self.q_trans_layer(torch.cat([x, r], dim=1))
        score_map = self.match_corr(embed_ref=q, embed_srch=M)
        score_map_M = score_map * M
        # score_map_M = torch.sigmoid(score_map) * M
        c = self.context_read_cnn_layer(score_map_M)
        return c

    def soft_update(self, x, h):
        r = torch.sigmoid(self.w_i_r(x) + self.w_h_r(h))
        z = torch.sigmoid(self.w_i_z(x) + self.w_h_z(h))
        n = torch.tanh(self.w_i_n(x) + r * (self.w_h_n(h)))
        h_ = (1 - z) * n + z * h

        return h_

    def forward(self, x, M, p_msk, n_msk):
        x_ = x.unsqueeze(dim=1)
        r = self.read(M=M)
        c = self.context_read(x=x, r=r, M=M)
        new_M = self.write(x=x_, M=M, p_msk=p_msk, n_msk=n_msk)
        output = self.cal_output(x=x, r=r, c=c)
        return output, new_M
