from .conf import *


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage:
    def __init__(self, ENV_CONF):
        self.ENV_CONF = ENV_CONF
        self.obs_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], *CONF['obs_shape']],
            dtype=torch.float32)
        self.value_s = torch.zeros([self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                   dtype=torch.float32)
        self.return_s = torch.zeros([self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                    dtype=torch.float32)

        self.reward_s = torch.zeros([self.ENV_CONF['max_step'], CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                    dtype=torch.float32)
        self.action_s_log_prob = torch.zeros([self.ENV_CONF['max_step'], CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                             dtype=torch.float32)
        self.action_s = torch.zeros(
            [self.ENV_CONF['max_step'], CONF['env_num'], self.ENV_CONF['uav_num'], CONF['action_space']],
            dtype=torch.float32)
        self.recurrent_hidden_states_s = torch.zeros(self.ENV_CONF['max_step'] + 1, CONF['env_num'],
                                                     self.ENV_CONF['uav_num'],
                                                     *CONF['M_size'], dtype=torch.float32)

        self.p_msk_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], 1,
             CONF['mtx_size'] * CONF['mtx_size'], *CONF['M_size']],
            dtype=torch.float32)
        self.n_msk_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], *CONF['M_size']],
            dtype=torch.float32)
        self._to_device()

    def _to_device(self):
        self.obs_s = self.obs_s.to(CONF['device'])
        self.value_s = self.value_s.to(CONF['device'])
        self.return_s = self.return_s.to(CONF['device'])

        self.reward_s = self.reward_s.to(CONF['device'])
        self.action_s_log_prob = self.action_s_log_prob.to(CONF['device'])
        self.action_s = self.action_s.to(CONF['device'])

        self.recurrent_hidden_states_s = self.recurrent_hidden_states_s.to(CONF['device'])

        self.p_msk_s = self.p_msk_s.to(CONF['device'])
        self.n_msk_s = self.n_msk_s.to(CONF['device'])

    def insert(self, shared_rollout, env_num):
        self.obs_s[:, env_num].copy_(shared_rollout.obs_s[:, 0])
        self.value_s[:, env_num].copy_(shared_rollout.value_s[:, 0])

        self.action_s[:, env_num].copy_(shared_rollout.action_s[:, 0])
        self.action_s_log_prob[:, env_num].copy_(shared_rollout.action_s_log_prob[:, 0])

        self.reward_s[:, env_num].copy_(shared_rollout.reward_s[:, 0])
        self.return_s[:, env_num].copy_(shared_rollout.return_s[:, 0])
        self.recurrent_hidden_states_s[:, env_num].copy_(shared_rollout.recurrent_hidden_states_s[:, 0])
        self.p_msk_s[:, env_num].copy_(shared_rollout.p_msk_s[:, 0])
        self.n_msk_s[:, env_num].copy_(shared_rollout.n_msk_s[:, 0])

    def minibatch_generator(self, advantage_s, uid):
        T, N = CONF['seq_len'], (self.ENV_CONF['max_step'] - CONF['seq_len'] + 1) * CONF['env_num']
        sampler = BatchSampler(SubsetRandomSampler(range(N)), CONF['mini_batch_size'], drop_last=False)
        obs_batch = []
        rhs_batch = []
        action_batch = []
        value_batch = []
        return_batch = []
        old_action_s_log_prob_batch = []
        adv_targ_batch = []
        p_msk_batch = []
        n_msk_batch = []
        for start_ind in range(self.ENV_CONF['max_step'] - CONF['seq_len'] + 1):
            start = start_ind
            end = start + CONF['seq_len']

            obs_batch.append(self.obs_s[start:end, :, uid])
            action_batch.append(self.action_s[start:end, :, uid])
            value_batch.append(self.value_s[start:end, :, uid])
            return_batch.append(self.return_s[start:end, :, uid])
            old_action_s_log_prob_batch.append(self.action_s_log_prob[start:end, :, uid])
            adv_targ_batch.append(advantage_s[start:end, :, uid])
            rhs_batch.append(self.recurrent_hidden_states_s[start:start + 1, :, uid])
            p_msk_batch.append(self.p_msk_s[start:end, :, uid])
            n_msk_batch.append(self.n_msk_s[start:end, :, uid])

        obs_batch = torch.cat(obs_batch, 1)
        action_batch = torch.cat(action_batch, 1)
        value_batch = torch.cat(value_batch, 1)
        return_batch = torch.cat(return_batch, 1)
        old_action_s_log_prob_batch = torch.cat(old_action_s_log_prob_batch, 1)
        adv_targ_batch = torch.cat(adv_targ_batch, 1)
        rhs_batch = torch.cat(rhs_batch, 1).view(N, *CONF['M_size'])
        p_msk_batch = torch.cat(p_msk_batch, 1)
        n_msk_batch = torch.cat(n_msk_batch, 1)

        for indices in sampler:
            cur_mini_batch_size = len(indices)
            obs_mini_batch = _flatten_helper(T, cur_mini_batch_size, obs_batch[:, indices])
            action_mini_batch = _flatten_helper(T, cur_mini_batch_size, action_batch[:, indices])
            value_mini_batch = _flatten_helper(T, cur_mini_batch_size, value_batch[:, indices])
            return_mini_batch = _flatten_helper(T, cur_mini_batch_size, return_batch[:, indices])

            old_action_s_log_prob_mini_batch = _flatten_helper(T, cur_mini_batch_size,
                                                               old_action_s_log_prob_batch[:, indices])
            adv_targ_mini_batch = _flatten_helper(T, cur_mini_batch_size, adv_targ_batch[:, indices])
            rhs_mini_batch = rhs_batch[indices]
            p_msk_mini_batch = _flatten_helper(T, cur_mini_batch_size, p_msk_batch[:, indices])
            n_msk_mini_batch = _flatten_helper(T, cur_mini_batch_size, n_msk_batch[:, indices])
            yield obs_mini_batch, action_mini_batch, value_mini_batch, return_mini_batch, \
                  old_action_s_log_prob_mini_batch, adv_targ_mini_batch, rhs_mini_batch, p_msk_mini_batch, n_msk_mini_batch
