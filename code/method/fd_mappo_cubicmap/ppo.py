from .conf import *


def adjust_learning_rate(optimizer, lr, iter_id):
    lr = CONF['lr'] * CONF['decay_rate'] ** max(0, iter_id - CONF['decay_start_iter_id'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class PPO:
    def __init__(self, ac_list):
        self.uav_num = len(ac_list)
        self.ac_list = ac_list
        self.lr_list = [CONF['lr'] for _ in range(self.uav_num)]
        self.optimizer_list = [optim.Adam(ac.parameters(), lr=CONF['lr'], eps=CONF['eps'], weight_decay=1e-6) for ac in
                               self.ac_list]

    def update(self, rollouts, iter_id):
        advantage_s = rollouts.return_s[:-1] - rollouts.value_s[:-1]
        advantage_s = (advantage_s - advantage_s.mean()) / (advantage_s.std() + 1e-5)

        value_loss_total = 0
        action_loss_total = 0
        dist_entropy_total = 0
        loss_total = 0
        sample_num = 0

        for _ in range(CONF['buffer_replay_time']):
            for uid in range(self.uav_num):
                data_generator = rollouts.minibatch_generator(advantage_s, uid)
                for sample_mini_batch in data_generator:
                    obs_mini_batch, action_mini_batch, value_mini_batch, return_mini_batch, \
                    old_action_s_log_prob_mini_batch, adv_targ_mini_batch, h_mini_batch, p_msk_mini_batch, n_msk_mini_batch = sample_mini_batch
                    sample_num += action_mini_batch.size(0)
                    evl_value_s, dist_entropy_s, action_s_log_prob, h = self.ac_list[uid].evaluate_action_s(
                        obs_mini_batch, action_mini_batch, h_mini_batch, p_msk_mini_batch,
                        n_msk_mini_batch)
                    ratio = torch.exp(action_s_log_prob - old_action_s_log_prob_mini_batch)
                    surr1 = ratio * adv_targ_mini_batch
                    surr2 = torch.clamp(ratio, 1.0 - CONF['clip_param'],
                                        1.0 + CONF['clip_param']) * adv_targ_mini_batch
                    action_loss = -torch.min(surr1, surr2).mean()

                    if CONF['use_clipped_value_loss']:
                        value_pred_clipped = value_mini_batch + \
                                             (evl_value_s - value_mini_batch).clamp(-CONF['clip_param'],
                                                                                    CONF['clip_param'])
                        value_losses = (evl_value_s - return_mini_batch).pow(2)
                        value_losses_clipped = (value_pred_clipped - return_mini_batch).pow(2)
                        value_loss = .5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(return_mini_batch, evl_value_s)

                    self.optimizer_list[uid].zero_grad()
                    loss = value_loss * CONF['value_loss_coef'] + action_loss - dist_entropy_s * CONF['entropy_coef']
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac_list[uid].parameters(), CONF['max_grad_norm'])
                    self.optimizer_list[uid].step()
                    value_loss_total += value_loss.item()
                    action_loss_total += action_loss.item()
                    dist_entropy_total += dist_entropy_s.item()
                    loss_total += loss.item()

        value_loss_per_sample = value_loss_total / sample_num
        action_loss_per_sample = action_loss_total / sample_num
        dist_entropy_per_sample = dist_entropy_total / sample_num
        loss_per_sample = loss_total / sample_num
        for uid in range(self.uav_num):
            self.lr_list[uid] = adjust_learning_rate(optimizer=self.optimizer_list[uid], lr=self.lr_list[uid],
                                                     iter_id=iter_id)
        return value_loss_per_sample, action_loss_per_sample, dist_entropy_per_sample, loss_per_sample
