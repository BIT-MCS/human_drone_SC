from .distributions import *
from .cubic_map import *


class Policy(nn.Module):
    def __init__(self, uid):
        super(Policy, self).__init__()
        self.hidden_size = CONF['hidden_size']
        self.base = CNNBase(CONF['obs_shape'][0], self.hidden_size, uid)
        self.dist_dia = DiagGaussian(self.hidden_size, 2)

    def get_action_s(self, obs_s, h, p_msk, n_msk):
        value_s, actor_feature_s, rhs_h_s = self.base(obs_s, h, p_msk, n_msk)
        dist_dia = self.dist_dia(actor_feature_s)
        action_dia = dist_dia.sample()
        action_log_probs_dia = dist_dia.log_probs(action_dia)

        return value_s, \
               action_dia, \
               action_log_probs_dia, \
               rhs_h_s

    def get_value_s(self, obs_s, h, p_msk, n_msk):
        value_s, _, rhs_hc_s = self.base(obs_s, h, p_msk, n_msk)
        return value_s

    def evaluate_action_s(self, obs_s, action_s, h, p_msk, n_msk):
        value_s, actor_feature_s, rhs_h_s = self.base(obs_s, h, p_msk, n_msk)
        dist_dia = self.dist_dia(actor_feature_s)
        action_log_probs_dia = dist_dia.log_probs(action_s)
        dist_entropy_dia = dist_dia.entropy().mean()
        return value_s, \
               dist_entropy_dia, \
               action_log_probs_dia, \
               rhs_h_s


class CNNBase(nn.Module):
    def __init__(self, input_channel_num, hidden_size, uid):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.uid = uid
        self.main = nn.Sequential(
            init_(nn.Conv2d(input_channel_num, 32, 8, stride=4, padding=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 5, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 4, stride=1, padding=1)),
            nn.ReLU(),

        )

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.cubic_map = CubicMap()
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, obs_s, rhs_h_s, p_msk, n_msk):
        actor_feature_s = self.main(obs_s)
        output, rhs_hc_s = self._forward_rnn(actor_feature_s, rhs_h_s, p_msk, n_msk)
        actor_feature_s = output
        value_s = self.critic_linear(actor_feature_s)
        return value_s, actor_feature_s, rhs_hc_s

    def _forward_rnn(self, x, rhs_h_s, p_msk, n_msk):
        if x.size(0) == rhs_h_s.size(0):
            outputs, rhs_h_s_s = self.cubic_map(x, rhs_h_s, p_msk, n_msk)
        else:
            N = rhs_h_s.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, *x.shape[1:])
            p_msk = p_msk.view(T, N, *p_msk.shape[1:])
            n_msk = n_msk.view(T, N, *n_msk.shape[1:])
            outputs = []
            rhs_h_s_s = []
            for i in range(T):
                output, rhs_h_s = self.cubic_map(x[i], rhs_h_s, p_msk[i], n_msk[i])
                outputs.append(output)
                rhs_h_s_s.append(rhs_h_s)

            outputs = torch.stack(outputs, dim=0)
            outputs = outputs.view(T * N, -1)
            rhs_h_s_s = torch.stack(rhs_h_s_s, dim=0)
            rhs_h_s_s = rhs_h_s_s.view(T * N, *CONF['M_size'])

        return outputs, rhs_h_s_s
