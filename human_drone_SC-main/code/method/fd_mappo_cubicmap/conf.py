from util import *

log_root_path = '../' + __file__.split('/')[-4] + '_log'
method_name = __file__.split('/')[-2]
CONF = {
    'root_path': log_root_path,
    'lr': 2.5e-4,
    'eps': 1e-5,
    'alpha': 0.99,
    'gamma': 0.99,
    'tau': 0.95,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.1,
    'max_grad_norm': 0.5,
    'env_num': 8,
    'mini_batch_size': 400,
    'buffer_replay_time': 4,
    'clip_param': 0.1,
    'train_iter': 10000,
    'test_num': 50,
    'use_clipped_value_loss': True,
    'decay_rate': 0.9995,
    'decay_start_iter_id': 3000,
    'obs_shape': [5, 20, 20],
    'obs_range': 20,
    'hr_shape': [400, 400],
    'action_space': 2,
    'hidden_size': 256,
    'device': 'cuda:0',
    'method_name': method_name,
    'dvd_size': 1 / 1,
    'dvd_num': 1,
    'seq_len': 5,
    'M_size': [16, 16, 16],  # Z, X, Y
    'mtx_size': 3,  # X' (Y')
}
