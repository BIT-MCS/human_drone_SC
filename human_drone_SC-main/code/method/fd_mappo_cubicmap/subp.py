from .model import *
from .sublog import *


@jit(nopython=True)
def fast_clip(a, min_v, max_v):
    l = a.shape[0]
    for i in range(l):
        if a[i] < min_v:
            a[i] = min_v
        elif a[i] > max_v:
            a[i] = max_v

    return a


@jit(nopython=True)
def fast_clip_val(a, min_v, max_v):
    if a < min_v:
        a = min_v
    elif a > max_v:
        a = max_v
    return a


@jit(fastmath=True)
def fast_int(a):
    a = np.round(a)
    a = int(a)
    return a


@jit(nopython=True)
def resize_img(scrH, scrW, dstH, dstW, obs, retimg):
    for i in range(dstH):
        for j in range(dstW):
            scrx = fast_int((i + 1) * (scrH * 1.0 / dstH))
            scry = fast_int((j + 1) * (scrW * 1.0 / dstW))
            new_x = fast_clip_val(scrx - 1, 0, obs.shape[2] - 1)
            new_y = fast_clip_val(scry - 1, 0, obs.shape[3] - 1)

            retimg[..., i, j] = obs[..., new_x, new_y]

    return retimg


@jit(nopython=True, )
def gen_obs(hr_obs, obs, ux, uy, blk_hr_grid_range, hr_shape, uav_num,
            cur_poi2hr_grid_range_channel, obs_shape, px, py, retimg,
            uav_types, org_ux, org_uy, uav_e, uid_pos, field_length,
            M_size, p_msk, n_msk, mtx_size, charge_hr_grid_range):
    # -----------------------------------------------
    ux = fast_clip(ux, 0, hr_shape[0] - 1)
    uy = fast_clip(uy, 0, hr_shape[1] - 1)
    px = fast_clip(px, 0, hr_shape[0] - 1)
    py = fast_clip(py, 0, hr_shape[0] - 1)

    for uid in range(uav_num):
        hr_obs[uid, :-4] = cur_poi2hr_grid_range_channel
        hr_obs[uid, -4, ...] = charge_hr_grid_range
        for i in range(blk_hr_grid_range[0].shape[0]):
            hr_obs[uid, -1, blk_hr_grid_range[0][i], blk_hr_grid_range[1][i]] = -1
        hr_obs[uid, -2, ux[uid], uy[uid]] = uav_e[uid]

    for i in range(len(px)):
        hr_obs[:, -3, px[i], py[i]] = 1

    half_shape_x = int(obs.shape[2] / 2)
    half_shape_y = int(obs.shape[3] / 2)

    hr_x_min = ux - half_shape_x
    hr_y_min = uy - half_shape_y
    hr_x_max = ux - half_shape_x + obs.shape[2]
    hr_y_max = uy - half_shape_y + obs.shape[2]

    hr_x_min = fast_clip(hr_x_min, 0, hr_shape[0] - 1)
    hr_y_min = fast_clip(hr_y_min, 0, hr_shape[1] - 1)
    hr_x_max = fast_clip(hr_x_max, 1, hr_shape[0])
    hr_y_max = fast_clip(hr_y_max, 1, hr_shape[1])

    x_min = half_shape_x - ux
    y_min = half_shape_y - uy
    x_min = fast_clip(x_min, 0, obs.shape[2] - 1)
    y_min = fast_clip(y_min, 0, obs.shape[3] - 1)
    x_max = x_min + hr_x_max - hr_x_min
    y_max = y_min + hr_y_max - hr_y_min
    x_max = fast_clip(x_max, 1, obs.shape[2])
    y_max = fast_clip(y_max, 1, obs.shape[3])

    for uid in range(uav_num):
        obs[uid, 0,
        x_min[uid]:x_max[uid], y_min[uid]:y_max[uid]] = hr_obs[uid, uav_types[uid],
                                                        hr_x_min[uid]:hr_x_max[uid], hr_y_min[uid]:hr_y_max[uid]]
        obs[uid, 1:,
        x_min[uid]:x_max[uid], y_min[uid]:y_max[uid]] = hr_obs[uid, -4:,
                                                        hr_x_min[uid]:hr_x_max[uid], hr_y_min[uid]:hr_y_max[uid]]

    w_size = mtx_size // 2
    for i in range(uav_num):
        uid_pos[i, 0] = fast_clip_val(int(org_ux[i] / field_length[0] * (M_size[1] - w_size * 2)) + 1, 1, M_size[1] - 2)
        uid_pos[i, 1] = fast_clip_val(int(org_uy[i] / field_length[1] * (M_size[2] - w_size * 2)) + 1, 1, M_size[2] - 2)

    for i in range(uav_num):
        c = 0
        for x in range(uid_pos[i, 0] - w_size, uid_pos[i, 0] + w_size + 1):
            for y in range(uid_pos[i, 1] - w_size, uid_pos[i, 1] + w_size + 1):
                p_msk[i, 0, c, :, x, y] = 1
                c += 1
    for i in range(uav_num):
        n_msk[i, :, uid_pos[i, 0] - w_size:uid_pos[i, 0] + w_size + 1,
        uid_pos[i, 1] - w_size:uid_pos[i, 1] + w_size + 1] = 0

    if obs_shape[1] == obs.shape[2]:
        return obs, uid_pos, p_msk, n_msk

    scrH, scrW = obs.shape[2], obs.shape[3]
    dstH, dstW = obs_shape[1], obs_shape[2]

    obs_new = resize_img(scrH, scrW, dstH, dstW, obs, retimg)

    return obs_new, uid_pos, p_msk, n_msk


def process_action(action_list):
    action_s = np.zeros([len(action_list), 2], dtype=np.float32)
    for i, action in enumerate(action_list):
        action_s[i, :] = action[0].numpy()
    return action_s


def subp(process_id,
         log_path,
         shared_rollout,
         shared_ifdone,
         init_poi_value_s,
         ENV_CONF, Env,
         ):
    with torch.no_grad():
        np.random.seed(seed + process_id)
        torch.manual_seed(seed + process_id)
        sub_iter_counter = 0
        print("---------------------------->", process_id, "subp")
        sublog = SubLog(ENV_CONF, log_path=log_path, process_id=process_id)
        local_ac_list = [Policy(uid) for uid in range(ENV_CONF['uav_num'])]

        for uavid in range(ENV_CONF['uav_num']):
            local_ac_list[uavid].eval()

        env = Env(CONF['obs_shape'][1:], CONF['hr_shape'])
        while sub_iter_counter < CONF['train_iter']:
            while True:
                if not shared_ifdone.value:
                    # sync shared model to local
                    for uavid in range(ENV_CONF['uav_num']):
                        local_ac_list[uavid].load_state_dict(torch.load(log_path + '/tmp_model_' + str(uavid) + '.pth',
                                                                        map_location=torch.device('cpu')))
                    ################################## feed in sharestorage ####################################
                    st = env.reset(init_poi_value_s)

                    hr_obs = np.zeros(
                        [ENV_CONF['uav_num'], CONF['obs_shape'][0] - 1 + ENV_CONF['data_type_num'], *CONF['hr_shape']],
                        dtype=np.float32)
                    obs = np.zeros([ENV_CONF['uav_num'], CONF['obs_shape'][0], CONF['obs_range'], CONF['obs_range']],
                                   dtype=np.float32)
                    ux = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 0] / ENV_CONF['field_length'][0] *
                                  CONF['hr_shape'][0], dtype=np.long)
                    uy = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 1] / ENV_CONF['field_length'][1] *
                                  CONF['hr_shape'][1], dtype=np.long)
                    px = np.array(st['cur_peo_pos_value'][:, 0] / ENV_CONF['field_length'][0] *
                                  CONF['hr_shape'][0], dtype=np.long)
                    py = np.array(st['cur_peo_pos_value'][:, 1] / ENV_CONF['field_length'][1] *
                                  CONF['hr_shape'][1], dtype=np.long)
                    retimg = np.zeros([ENV_CONF['uav_num'], *CONF['obs_shape']], dtype=np.float32)

                    uid_pos = np.zeros([ENV_CONF['uav_num'], 2], dtype=np.long)
                    p_msk = np.zeros(
                        [ENV_CONF['uav_num'], 1, CONF['mtx_size'] * CONF['mtx_size'], *CONF['M_size']],
                        dtype=np.float32)
                    n_msk = np.ones([ENV_CONF['uav_num'], *CONF['M_size']])
                    obs, uid_pos, p_msk, n_msk = gen_obs(hr_obs, obs, ux, uy, env.blk_hr_grid_range,
                                                         CONF['hr_shape'], ENV_CONF['uav_num'],
                                                         st['cur_poi2hr_grid_range_channel'],
                                                         CONF['obs_shape'], px, py, retimg,
                                                         st['uav_types'],
                                                         st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                         :, 0],
                                                         st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                         :, 1],
                                                         st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                         :, 2],
                                                         uid_pos,
                                                         ENV_CONF['field_length'], CONF['M_size'],
                                                         p_msk, n_msk, CONF['mtx_size'],
                                                         st[
                                                             'cur_charge_station2hr_grid_range_channel']
                                                         )
                    sublog.episode_info = []
                    sublog.episode_info.append([st, obs])
                    shared_rollout.reset()
                    shared_rollout.obs_s[0, 0].copy_(torch.tensor(obs, dtype=torch.float32))
                    shared_rollout.p_msk_s[0, 0].copy_(torch.tensor(p_msk, dtype=torch.float32))
                    shared_rollout.n_msk_s[0, 0].copy_(torch.tensor(n_msk, dtype=torch.float32))
                    for step_id in range(ENV_CONF['max_step']):
                        value_list = []
                        action_list = []
                        action_log_prob_list = []
                        h_list = []

                        for uid in range(ENV_CONF['uav_num']):
                            local_ac = local_ac_list[uid]

                            value, action, action_log_prob, h = local_ac.get_action_s(
                                shared_rollout.obs_s[step_id, :, uid],
                                shared_rollout.recurrent_hidden_states_s[step_id, :, uid],
                                shared_rollout.p_msk_s[step_id, :, uid],
                                shared_rollout.n_msk_s[step_id, :, uid],
                            )

                            value_list.append(torch.unsqueeze(value, 1))
                            action_list.append(action)
                            action_log_prob_list.append(torch.unsqueeze(action_log_prob, 1))
                            h_list.append(torch.unsqueeze(h, 1))

                        value_s = torch.cat(value_list, 1)
                        action_s = process_action(action_list)
                        action_s_log_prob = torch.cat(action_log_prob_list, 1)
                        h = torch.cat(h_list, 1)
                        st, done = env.step(action_s)
                        hr_obs = np.zeros(
                            [ENV_CONF['uav_num'], CONF['obs_shape'][0] - 1 + ENV_CONF['data_type_num'],
                             *CONF['hr_shape']],
                            dtype=np.float32)
                        obs = np.zeros(
                            [ENV_CONF['uav_num'], CONF['obs_shape'][0], CONF['obs_range'], CONF['obs_range']],
                            dtype=np.float32)
                        ux = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 0] / ENV_CONF['field_length'][0] *
                                      CONF['hr_shape'][0], dtype=np.long)
                        uy = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 1] / ENV_CONF['field_length'][1] *
                                      CONF['hr_shape'][1], dtype=np.long)
                        px = np.array(st['cur_peo_pos_value'][:, 0] / ENV_CONF['field_length'][0] *
                                      CONF['hr_shape'][0], dtype=np.long)
                        py = np.array(st['cur_peo_pos_value'][:, 1] / ENV_CONF['field_length'][1] *
                                      CONF['hr_shape'][1], dtype=np.long)
                        retimg = np.zeros([ENV_CONF['uav_num'], *CONF['obs_shape']], dtype=np.float32)
                        uid_pos = np.zeros([ENV_CONF['uav_num'], 2], dtype=np.long)
                        p_msk = np.zeros([ENV_CONF['uav_num'], 1, CONF['mtx_size'] * CONF['mtx_size'],
                                          *CONF['M_size']],
                                         dtype=np.float32)
                        n_msk = np.ones([ENV_CONF['uav_num'], *CONF['M_size']])
                        obs, uid_pos, p_msk, n_msk = gen_obs(hr_obs, obs, ux, uy,
                                                             env.blk_hr_grid_range,
                                                             CONF['hr_shape'], ENV_CONF['uav_num'],
                                                             st['cur_poi2hr_grid_range_channel'],
                                                             CONF['obs_shape'], px, py, retimg,
                                                             st['uav_types'],
                                                             st[
                                                                 'cur_uav_pos_energy_hit_mec_cec_charge'][
                                                             :, 0],
                                                             st[
                                                                 'cur_uav_pos_energy_hit_mec_cec_charge'][
                                                             :, 1],
                                                             st[
                                                                 'cur_uav_pos_energy_hit_mec_cec_charge'][
                                                             :, 2],
                                                             uid_pos,
                                                             ENV_CONF['field_length'],
                                                             CONF['M_size'],
                                                             p_msk, n_msk, CONF['mtx_size'],
                                                             st[
                                                                 'cur_charge_station2hr_grid_range_channel']
                                                             )
                        sublog.episode_info.append([st, obs, done, action_s])
                        shared_rollout.insert(torch.tensor(obs, dtype=torch.float32),
                                              value_s,
                                              action_s_log_prob,
                                              torch.tensor(action_s, dtype=torch.float32),
                                              h, torch.tensor(p_msk, dtype=torch.float32),
                                              torch.tensor(n_msk, dtype=torch.float32))
                    next_value_list = []
                    for uid in range(ENV_CONF['uav_num']):
                        local_ac = local_ac_list[uid]
                        next_value = local_ac.get_value_s(
                            shared_rollout.obs_s[-1, :, uid],
                            shared_rollout.recurrent_hidden_states_s[-1, :, uid],
                            shared_rollout.p_msk_s[-1, :, uid],
                            shared_rollout.n_msk_s[-1, :, uid],
                        )
                        next_value_list.append(torch.unsqueeze(next_value, 1))
                    next_value_s = torch.cat(next_value_list, 1)
                    shared_rollout.gen_reward(sublog.episode_info)

                    shared_rollout.compute_returns(next_value_s, use_gae=True, gamma=CONF['gamma'], tau=CONF['tau'])
                    ################################## sublog work ####################################
                    sublog.gen_metrics_result(sub_iter_counter)
                    sublog.record_metrics_result()
                    if process_id == 0 and sub_iter_counter % 10 == 0:
                        sublog.record_trace_se(sub_iter_counter, env)
                    if sub_iter_counter % 50 == 0:
                        sublog.draw_ana(sub_iter_counter, shared_rollout.reward_s.numpy())
                    shared_ifdone.value = True
                    sub_iter_counter += 1
                    break
