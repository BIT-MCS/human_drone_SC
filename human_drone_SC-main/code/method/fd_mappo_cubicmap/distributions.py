from util import *

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))
        self.fc_mean = nn.Sequential(
            init_(nn.Linear(input_size, output_size)),
        )
        self.logstd = AddBias(torch.zeros(output_size))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean = torch.tanh(action_mean)
        zeros = torch.zeros(action_mean.size(), device=x.device)
        action_logstd = self.logstd(zeros)
        action_logstd = torch.tanh(action_logstd)
        return FixedNormal(action_mean, action_logstd.exp())
