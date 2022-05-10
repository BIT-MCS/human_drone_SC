from .conf import *


class Env:
    env_conf = {}
    peo_dict = {}
    blk_dict = {}
    peo_pos_value = 0
    peo_visit_poi_time = 0
    pure_peo_dc = 0
    coordx_offset = 0
    coordy_offset = 0
    state_info = {}
    poi_pos_value = 0
    step_counter = 0
    reset_counter = 0

    def __init__(self, grid_size, hr_grid_size):
        super(Env, self).__init__()
        self.if_regen_pure_peo_dc = True
        self.env_conf = ENV_CONF
        self.grid_size = grid_size
        self.hr_grid_size = [ENV_CONF['data_type_num'], *hr_grid_size]
        self.coordx_offset = self._lon2coordx(self.env_conf['core_lon_min'])
        self.coordy_offset = self._lat2coordy(self.env_conf['core_lat_min'])
        self._load_peo_dict()
        self._load_blk_dict()
        self.blk_hr_grid_range = [[], []]
        self._gen_blk_grid_dict()
        self.poi_pos_value = np.zeros([self.env_conf['poi_num'], 2 + ENV_CONF['data_type_num']], dtype=np.float32)
        self.poi2hr_grid_range = {}
        self._load_poi_pos()
        self._gen_poi_grid_dict()
        self.charge_station_pos = np.zeros([ENV_CONF['charge_station_num'], 2], dtype=np.float32)
        self.charge_station2hr_grid_range = {}
        self._load_charge_station_pos()
        self._gen_charge_station_grid_dict_p()
        self._gen_peo_pos_visit_info()
        self.pre_reset()
        self._blk_dict2array()
        self._poi_dict2array()
        self._charge_station_dict2array()
        self._uav_types = gen_uav_sensor_type()

    def _blk_dict2array(self):
        max_len = 0
        blk_len_list = []
        for blk in self.blk_dict:
            max_len = max(max_len, len(self.blk_dict[blk][1]))
            blk_len_list.append(len(self.blk_dict[blk][1]))
        self.blk_len_array = np.array(blk_len_list)
        self.blk_array = np.zeros([len(self.blk_dict), max_len, 2], dtype=np.float32)
        self.blk_name_array = np.zeros(len(self.blk_dict), dtype=np.long)
        for idx, blk in enumerate(self.blk_dict):
            self.blk_array[idx, :self.blk_len_array[idx]] = np.array(self.blk_dict[blk][1])
            if self.blk_dict[blk][1] == 'r':
                print('error!!!!')

    def _poi_dict2array(self):
        max_len = 0
        len_list = []
        for poi in self.poi2hr_grid_range:
            max_len = max(max_len, len(self.poi2hr_grid_range[poi][0]))
            len_list.append(len(self.poi2hr_grid_range[poi][0]))
        self.poi_len_array = np.array(len_list)
        self.poi_array = np.zeros([len(self.poi2hr_grid_range), max_len, 2], dtype=np.long)
        self.poi2idx = {}
        for idx, poi in enumerate(self.poi2hr_grid_range):
            self.poi_array[idx, :self.poi_len_array[idx], 0] = self.poi2hr_grid_range[poi][0]
            self.poi_array[idx, :self.poi_len_array[idx], 1] = self.poi2hr_grid_range[poi][1]
            self.poi2idx[poi] = idx

    def _charge_station_dict2array(self):
        max_len = 0
        len_list = []
        for charge_station in self.charge_station2hr_grid_range:
            max_len = max(max_len, len(self.charge_station2hr_grid_range[charge_station][0]))
            len_list.append(len(self.charge_station2hr_grid_range[charge_station][0]))
        self.charge_station_len_array = np.array(len_list)
        self.charge_station_array = np.zeros([len(self.charge_station2hr_grid_range), max_len, 2], dtype=np.long)
        self.charge_station2idx = {}
        for idx, charge_station in enumerate(self.charge_station2hr_grid_range):
            self.charge_station_array[idx, :self.charge_station_len_array[idx], 0] = \
                self.charge_station2hr_grid_range[charge_station][0]
            self.charge_station_array[idx, :self.charge_station_len_array[idx], 1] = \
                self.charge_station2hr_grid_range[charge_station][1]
            self.charge_station2idx[charge_station] = idx

    def _lon2coordx(self, lon):
        coordx = self.env_conf['ref_coordx'] + (lon - self.env_conf['ref_lon']) * self.env_conf['coordx_per_lon']
        return coordx

    def _lat2coordy(self, lat):
        coordy = self.env_conf['ref_coordy'] + (lat - self.env_conf['ref_lat']) * self.env_conf['coordy_per_lat']
        return coordy

    def _load_peo_dict(self):
        self.peo_dict = np.load(self.env_conf['peo_dict_path'], allow_pickle=True)[()]
        for peo in self.peo_dict:
            record_num = len(self.peo_dict[peo])
            for i in range(record_num):
                self.peo_dict[peo][i][1] -= self.coordx_offset
                self.peo_dict[peo][i][2] -= self.coordy_offset

    def _load_blk_dict(self):
        self.blk_dict = np.load(self.env_conf['blk_dict_path'], allow_pickle=True)[()]
        for blk in self.blk_dict:
            blk_att = self.blk_dict[blk][0]
            if blk_att == 'p':
                key_num = len(self.blk_dict[blk][1])
                for i in range(key_num):
                    self.blk_dict[blk][1][i][0] -= self.coordx_offset
                    self.blk_dict[blk][1][i][1] -= self.coordy_offset
            elif blk_att == 'r':
                self.blk_dict[blk][1][0][0] -= self.coordx_offset
                self.blk_dict[blk][1][0][1] -= self.coordy_offset

    def _gen_blk_grid_dict(self):
        if not os.path.exists('./environment/' + ENV_CONF['dataset_name'] + '/blk_hr_grid_range.npy'):
            self.hr_p_is_in_blk_dict = {}
            for i in range(self.hr_grid_size[1] + 1):
                for j in range(self.hr_grid_size[2] + 1):
                    key = str(i) + '_' + str(j)
                    self.hr_p_is_in_blk_dict[key] = self._is_into_blk2((
                        i * self.env_conf['field_length'][0] / self.hr_grid_size[1],
                        j * self.env_conf['field_length'][1] / self.hr_grid_size[2]))
            for i in range(self.hr_grid_size[1]):
                for j in range(self.hr_grid_size[2]):
                    for dx, dy in zip([0, 1, 0, 1], [0, 0, 1, 1]):
                        x = i + dx
                        y = j + dy
                        key = str(x) + '_' + str(y)
                        if self.hr_p_is_in_blk_dict[key]:
                            self.blk_hr_grid_range[0].append(i)
                            self.blk_hr_grid_range[1].append(j)
                            break
            np.save('./environment/' + ENV_CONF['dataset_name'] + '/blk_hr_grid_range.npy', self.blk_hr_grid_range)
        self.blk_hr_grid_range = np.load('./environment/' + ENV_CONF['dataset_name'] + '/blk_hr_grid_range.npy',
                                         allow_pickle=True)

    def _load_poi_pos(self):
        poi_dict = np.load(self.env_conf['poi_dict_path'], allow_pickle=True)[()]
        counter = 0
        for poi in poi_dict:
            self.poi_pos_value[counter, :] = [poi_dict[poi][0] - self.coordx_offset,
                                              poi_dict[poi][1] - self.coordy_offset] + [0] * ENV_CONF['data_type_num']
            counter += 1

    def _load_charge_station_pos(self):
        all_charge_station_dict = np.load(self.env_conf['charge_station_dict_path'], allow_pickle=True)[()]
        charge_path = './environment/' + dataset_name + '/charge_station_set.npy'
        used_name = list(np.load(charge_path, allow_pickle=True)[()])
        for idx in range(len(used_name)):
            self.charge_station_pos[idx, :] = [
                all_charge_station_dict[used_name[idx]][0] - self.coordx_offset,
                all_charge_station_dict[used_name[idx]][1] - self.coordy_offset]

    def _gen_poi_grid_dict(self):
        if not os.path.exists('./environment/' + ENV_CONF['dataset_name'] + '/poi2hr_grid_range.npy'):
            for poi in range(self.env_conf['poi_num']):
                pos_x = self.poi_pos_value[poi, 0]
                pos_y = self.poi_pos_value[poi, 1]

                self.poi2hr_grid_range[poi] = [[], []]
                for x in range(self.hr_grid_size[1]):
                    for y in range(self.hr_grid_size[2]):
                        true_x = (x + 0.5) * self.env_conf['field_length'][0] / self.hr_grid_size[1]
                        true_y = (y + 0.5) * self.env_conf['field_length'][1] / self.hr_grid_size[2]
                        dis = ((true_x - pos_x) ** 2 + (true_y - pos_y) ** 2) ** 0.5
                        if dis < self.env_conf['uav_sensing_range']:
                            self.poi2hr_grid_range[poi][0].append(x)
                            self.poi2hr_grid_range[poi][1].append(y)
            np.save('./environment/' + ENV_CONF['dataset_name'] + '/poi2hr_grid_range.npy', self.poi2hr_grid_range)
        self.poi2hr_grid_range = \
            np.load('./environment/' + ENV_CONF['dataset_name'] + '/poi2hr_grid_range.npy', allow_pickle=True)[()]

    def _gen_charge_station_grid_dict_p(self):
        for charge_station_idx in range(ENV_CONF['charge_station_num']):
            pos_x = self.charge_station_pos[charge_station_idx, 0]
            pos_y = self.charge_station_pos[charge_station_idx, 1]
            self.charge_station2hr_grid_range[charge_station_idx] = [[], []]
            hr_pos_x = pos_x * self.hr_grid_size[1] / self.env_conf['field_length'][0]
            hr_pos_y = pos_y * self.hr_grid_size[2] / self.env_conf['field_length'][1]
            self.charge_station2hr_grid_range[charge_station_idx][0].append(hr_pos_x)
            self.charge_station2hr_grid_range[charge_station_idx][1].append(hr_pos_y)

    def _gen_peo_pos_visit_info(self):
        if not os.path.exists('./environment/' + ENV_CONF['dataset_name'] + '/peo_pos_value.npy'):
            self.peo_pos_value = np.zeros([self.env_conf['max_step'], self.env_conf['peo_num'], 3])
            self.peo_visit_poi_time = np.zeros([self.env_conf['max_step'], self.env_conf['poi_num']])

            cur_peo_pos_value = np.zeros([self.env_conf['peo_num'], 3])
            peo_list = list(self.peo_dict.keys())[: self.env_conf['peo_num']]
            for i, peo in enumerate(peo_list):
                cur_peo_pos_value[i, 0] = self.peo_dict[peo][0][1]
                cur_peo_pos_value[i, 1] = self.peo_dict[peo][0][2]
                cur_peo_pos_value[i, 2] = self.env_conf['peo_value']
            for pre_step in range(self.env_conf['max_step']):
                cur_visit_poi = np.zeros(self.env_conf['poi_num'])
                if pre_step == 0:
                    visit_poi = np.zeros(self.env_conf['poi_num'])
                else:
                    visit_poi = copy.deepcopy(self.peo_visit_poi_time[pre_step - 1])
                time = pre_step * self.env_conf['epoch_time_range']
                future_time = (pre_step + 1) * self.env_conf['epoch_time_range']
                start_id = max(time // self.env_conf['record_time_interval'] - 1, 0)
                for peoid, peo in enumerate(peo_list):
                    record_list = self.peo_dict[peo]
                    if start_id > len(record_list) - 1:
                        cur_peo_pos_value[peoid, 0] = 1e8
                        cur_peo_pos_value[peoid, 1] = 1e8
                        continue
                    for record in record_list[start_id:]:
                        record_time = record[0]
                        if record_time > future_time + self.env_conf['min_value']:
                            break
                        if time < record_time <= future_time + self.env_conf['min_value']:
                            peo_x = record[1]
                            peo_y = record[2]
                            cur_peo_pos_value[peoid, 0] = peo_x
                            cur_peo_pos_value[peoid, 1] = peo_y
                            for pid in range(self.env_conf['poi_num']):
                                if cur_visit_poi[pid] == 0:
                                    px = self.poi_pos_value[pid, 0]
                                    py = self.poi_pos_value[pid, 1]
                                    dis = self._dis((px, py), (peo_x, peo_y))
                                    if dis < self.env_conf['peo_sensing_range']:
                                        visit_poi[pid] += 1
                                        cur_visit_poi[pid] = 1
                self.peo_pos_value[pre_step] = copy.deepcopy(cur_peo_pos_value)
                self.peo_visit_poi_time[pre_step] = copy.deepcopy(visit_poi)
            np.save('./environment/' + ENV_CONF['dataset_name'] + '/peo_pos_value.npy', self.peo_pos_value)
            np.save('./environment/' + ENV_CONF['dataset_name'] + '/peo_visit_poi_time.npy', self.peo_visit_poi_time)
        self.peo_pos_value = np.load('./environment/' + ENV_CONF['dataset_name'] + '/peo_pos_value.npy',
                                     allow_pickle=True)
        self.peo_visit_poi_time = \
            np.load('./environment/' + ENV_CONF['dataset_name'] + '/peo_visit_poi_time.npy', allow_pickle=True)

    def pre_reset(self):
        self._pre_cur_pid_visit = np.zeros([self.env_conf['poi_num'], 2], dtype=np.float32)

        # init uav
        self._pre_cur_uav_pos_energy_hit_mec_cec_charge = np.zeros([self.env_conf['uav_num'], 7], dtype=np.float32)
        uav_init_pos_coordx = 0
        uav_init_pos_coordy = 0
        if self.env_conf['uav_init_pos'] == 'center':
            uav_init_pos_coordx = self.env_conf['field_length'][0] / 2
            uav_init_pos_coordy = self.env_conf['field_length'][1] / 2
        self._pre_cur_uav_pos_energy_hit_mec_cec_charge[:, 0] = uav_init_pos_coordx
        self._pre_cur_uav_pos_energy_hit_mec_cec_charge[:, 1] = uav_init_pos_coordy
        self._pre_cur_uav_pos_energy_hit_mec_cec_charge[:, 2] = self.env_conf['uav_init_energy']

        # init peo
        self._pre_cur_peo_pos_value = np.zeros([self.env_conf['peo_num'], 3], dtype=np.float32)
        peo_list = list(self.peo_dict.keys())[: self.env_conf['peo_num']]
        for i, peo in enumerate(peo_list):
            self._pre_cur_peo_pos_value[i, 0] = self.peo_dict[peo][0][1]
            self._pre_cur_peo_pos_value[i, 1] = self.peo_dict[peo][0][2]

    def _is_into_blk2(self, p):
        flag = False
        x = p[0]
        y = p[1]
        for blk in self.blk_dict:
            blk_att = self.blk_dict[blk][0]
            blk_key_list = self.blk_dict[blk][1]
            if blk_att == 'p':
                path = mplpath.Path(blk_key_list)
                if path.contains_point(p):
                    flag = True
                    break
            elif blk_att == 'r':
                dis = self._dis([x, y], blk_key_list[0])
                if dis <= blk_key_list[1]:
                    flag = True
                    break
        return flag

    def _dis(self, xy1, xy2):
        dis = ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5
        return dis

    def reset(self, init_poi_value):
        self.step_counter = 0
        if self.reset_counter > 0 and np.sum(np.abs(init_poi_value - self.init_poi_value)) < ENV_CONF['min_value']:
            self.if_regen_pure_peo_dc = False
        self.init_poi_value = init_poi_value

        # init uav
        self._cur_uav_pos_energy_hit_mec_cec_charge = copy.deepcopy(self._pre_cur_uav_pos_energy_hit_mec_cec_charge)
        self._cur_uav_visit_poi_time = np.zeros([self.env_conf['uav_num'], self.env_conf['poi_num']], dtype=np.float32)
        self._cur_uav_dc = np.zeros([self.env_conf['uav_num'], self.env_conf['poi_num']], dtype=np.float32)

        # init peo
        self._cur_peo_pos_value = copy.deepcopy(self._pre_cur_peo_pos_value)
        self._cur_peo_dc = np.zeros(self.env_conf['poi_num'], dtype=np.float32)
        self._cur_peo_visit_poi_time = np.zeros(self.env_conf['poi_num'], dtype=np.float32)

        # init poi
        self.poi_pos_value[:, 2:] = self.init_poi_value
        self._cur_poi_pos_value = copy.deepcopy(self.poi_pos_value)
        if self.if_regen_pure_peo_dc:
            self._gen_pure_peo_dc()

        self.poi2hr_grid_range_channel = np.zeros(self.hr_grid_size, dtype=np.float32)
        self.poi2hr_grid_range_channel = go_fast2(self.poi2hr_grid_range_channel, ENV_CONF['poi_num'], self.poi_array,
                                                  self.poi_len_array,
                                                  ENV_CONF['data_type_num'], self._cur_poi_pos_value,
                                                  ENV_CONF['poi_value_max'] / ENV_CONF['data_type_num'])
        self.charge_station2hr_grid_range_channel = np.zeros(self.hr_grid_size[1:], dtype=np.float32)
        self.charge_station2hr_grid_range_channel = go_fast2_charge_station(self.charge_station2hr_grid_range_channel,
                                                                            ENV_CONF['charge_station_num'],
                                                                            self.charge_station_array,
                                                                            self.charge_station_len_array)

        self.is_uav_collect = np.zeros([ENV_CONF['uav_num']], dtype=np.bool)

        self._gen_state_info()
        self.reset_counter += 1

        return self.state_info

    def _gen_pure_peo_dc(self):
        self.pure_peo_dc = np.zeros([self.env_conf['poi_num'], ENV_CONF['data_type_num']],
                                    dtype=np.float32)
        poi_value = copy.deepcopy(self.poi_pos_value[:, 2:])
        for step in range(self.env_conf['max_step']):
            visit_poi = self.peo_visit_poi_time[step]
            if step == 0:
                last_visit_poi = 0
            else:
                last_visit_poi = self.peo_visit_poi_time[step - 1]
            inc_visit_poi = visit_poi - last_visit_poi
            tmp_inc_peo_dc = inc_visit_poi * self.env_conf['peo_collect_speed_per_poi'] / ENV_CONF['data_type_num']
            tmp_ones = np.ones_like(poi_value)
            tmp_inc_peo_dc = np.expand_dims(tmp_inc_peo_dc, axis=-1) * tmp_ones
            tmp_poi_value = np.maximum((poi_value - tmp_inc_peo_dc), 0)
            self.pure_peo_dc += poi_value - tmp_poi_value
            poi_value = copy.deepcopy(tmp_poi_value)

    def _gen_state_info(self):
        self.state_info = {}
        self.state_info['env_init_poi_value'] = self.init_poi_value

        self.state_info['pure_peo_dc'] = self.pure_peo_dc

        self.state_info['cur_peo_pos_value'] = copy.deepcopy(self._cur_peo_pos_value)
        self.state_info['cur_peo_dc'] = copy.deepcopy(self._cur_peo_dc)
        self.state_info['cur_peo_visit_poi_time'] = copy.deepcopy(self._cur_peo_visit_poi_time)

        self.state_info['cur_uav_pos_energy_hit_mec_cec_charge'] = copy.deepcopy(
            self._cur_uav_pos_energy_hit_mec_cec_charge)
        self.state_info['cur_uav_dc'] = copy.deepcopy(self._cur_uav_dc)
        self.state_info['cur_uav_visit_poi_time'] = copy.deepcopy(self._cur_uav_visit_poi_time)

        self.state_info['cur_poi_pos_value'] = copy.deepcopy(self._cur_poi_pos_value)

        self.state_info['cur_poi2hr_grid_range_channel'] = copy.deepcopy(self.poi2hr_grid_range_channel)
        self.state_info['cur_charge_station2hr_grid_range_channel'] = copy.deepcopy(
            self.charge_station2hr_grid_range_channel)

        self.state_info['uav_types'] = self._uav_types

        self.state_info['is_uav_collect'] = copy.deepcopy(self.is_uav_collect)

    def step(self, action):
        action_cp = np.zeros_like(action)
        action_cp[:, :2] = action[:, :2] * self.env_conf['uav_dis_max'] / (2 ** 0.5)
        action = self._clip_action(action_cp)
        hit_list, uav_move_dis = fast_uav_move(self.env_conf['uav_num'], action,
                                               self._cur_uav_pos_energy_hit_mec_cec_charge,
                                               self.env_conf['uav_move_energy_consume_ratio'],
                                               self.env_conf['min_value'],
                                               self.env_conf['field_length'][0], self.env_conf['field_length'][1],
                                               self.blk_array, self.blk_len_array, self.blk_name_array)

        for uid in range(self.env_conf['uav_num']):
            if not fast_is_out_of_energy(self._cur_uav_pos_energy_hit_mec_cec_charge, ENV_CONF['min_value'], uid):
                move_energy_consume = uav_move_dis[uid] * self.env_conf['uav_move_energy_consume_ratio']
                self._cur_uav_pos_energy_hit_mec_cec_charge[uid, 2] -= move_energy_consume
                self._cur_uav_pos_energy_hit_mec_cec_charge[uid, 4] += move_energy_consume

        uav_collected_data = np.zeros(
            shape=[self.env_conf['uav_num'], self.env_conf['poi_num'], ENV_CONF['data_type_num']], dtype=np.float32)
        hit_list.append(-1)
        uav_collect_array, uav_collected_data, _cur_poi_pos_value, charge_list = fast_collect_data_or_charge(
            uav_collected_data,
            self.env_conf[
                'poi_num'],
            self.env_conf[
                'uav_num'],
            self._cur_uav_pos_energy_hit_mec_cec_charge,
            self.env_conf[
                'min_value'],
            hit_list,
            self._cur_poi_pos_value,
            self.env_conf[
                'uav_sensing_range'],
            self._cur_uav_visit_poi_time,
            self.env_conf[
                'uav_collect_speed_per_poi'],
            self.env_conf[
                'uav_collect_energy_consume_ratio'],
            self._uav_types,
            action,
            self.env_conf[
                'uav_init_energy'],
            self.charge_station_pos,
            self.env_conf[
                'charge_sensing_range'])

        for uid in range(self.env_conf['uav_num']):
            self.is_uav_collect[uid] = not charge_list[uid]
        self._cur_poi_pos_value = _cur_poi_pos_value
        uav_collect_energy_consume = np.sum(np.sum(uav_collected_data, axis=1), axis=-1) * self.env_conf[
            'uav_collect_energy_consume_ratio']
        self._cur_uav_pos_energy_hit_mec_cec_charge[:, 2] -= uav_collect_energy_consume
        self._cur_uav_pos_energy_hit_mec_cec_charge[:, 5] += uav_collect_energy_consume

        self._cur_peo_pos_value = self.peo_pos_value[self.step_counter]
        self._cur_peo_visit_poi_time = self.peo_visit_poi_time[self.step_counter]
        last_peo_visit_poi_time = 0
        if self.step_counter != 0:
            last_peo_visit_poi_time = self.peo_visit_poi_time[self.step_counter - 1]
        inc_peo_visit_poi_time = self._cur_peo_visit_poi_time - last_peo_visit_poi_time
        inc_peo_visit_poi_time = inc_peo_visit_poi_time * uav_collect_array
        tmp_inc_peo_dc = inc_peo_visit_poi_time * self.env_conf['peo_collect_speed_per_poi'] / ENV_CONF['data_type_num']
        tmp_ones = np.ones_like(self._cur_poi_pos_value[:, 2:])
        tmp_inc_peo_dc = np.expand_dims(tmp_inc_peo_dc, axis=-1) * tmp_ones
        tmp_poi_value = np.maximum((self._cur_poi_pos_value[:, 2:] - tmp_inc_peo_dc), 0)
        self._cur_peo_dc += np.sum(self._cur_poi_pos_value[:, 2:] - tmp_poi_value, axis=-1)
        self._delta_dc = self._cur_poi_pos_value[:, 2:] - tmp_poi_value
        self._cur_poi_pos_value[:, 2:] = copy.deepcopy(tmp_poi_value)

        self._cur_uav_dc += np.sum(uav_collected_data, axis=-1)
        self._delta_dc += np.sum(uav_collected_data, axis=0)
        nonzero_idxs = np.nonzero(np.sum(self._delta_dc, axis=-1))
        nonzero_num = nonzero_idxs[0].size

        poi_idxs = []
        for nonzero_idxs_id in range(nonzero_num):
            poi = nonzero_idxs[0][nonzero_idxs_id]
            poi_idxs.append(self.poi2idx[poi])

        if len(poi_idxs) == 0:
            poi_idxs.append(-1)
        self.poi2hr_grid_range_channel = go_fast(poi_idxs, self.poi_array, self.poi_len_array,
                                                 self.poi2hr_grid_range_channel,
                                                 self._delta_dc,
                                                 ENV_CONF['poi_value_max'] / ENV_CONF['data_type_num'],
                                                 ENV_CONF['data_type_num']
                                                 )

        self._gen_state_info()
        done = False
        self.step_counter += 1
        if np.sum(self._cur_uav_pos_energy_hit_mec_cec_charge[:, 2]) < self.env_conf[
            'min_value'] or self.step_counter >= \
                self.env_conf['max_step']:
            done = True

        return self.state_info, done

    def _clip_action(self, action):
        for uid in range(self.env_conf['uav_num']):
            dx = action[uid, 0]
            dy = action[uid, 1]
            dis = (dx ** 2 + dy ** 2) ** 0.5
            if dis > self.env_conf['uav_dis_max']:
                dx = dx / dis * self.env_conf['uav_dis_max']
                dy = dy / dis * self.env_conf['uav_dis_max']
            action[uid, 0] = dx
            action[uid, 1] = dy
        return action

    @staticmethod
    def gen_whole_init_poi_value():
        if not os.path.exists('./environment/' + ENV_CONF['dataset_name'] + '/init_poi_value.npy'):
            range = (ENV_CONF['poi_value_max'] - ENV_CONF['poi_value_min']) / ENV_CONF['data_type_num']

            init_poi_value = np.random.random([ENV_CONF['poi_num'], ENV_CONF['data_type_num']])
            init_poi_value = init_poi_value * range + ENV_CONF['poi_value_min'] / ENV_CONF['data_type_num']
            np.save('./environment/' + ENV_CONF['dataset_name'] + '/init_poi_value.npy', init_poi_value)
        init_poi_value = np.load('./environment/' + ENV_CONF['dataset_name'] + '/init_poi_value.npy',
                                 allow_pickle=True)
        return init_poi_value


@jit(nopython=True)
def go_fast(poi_idxs, poi_array, poi_len_array, poi2hr_grid_range_channel, _delta_dc,
            poi_val_max, data_type_num):
    if poi_idxs[0] == -1:
        return poi2hr_grid_range_channel
    for idx in poi_idxs:

        xs = poi_array[idx, :poi_len_array[idx], 0]
        ys = poi_array[idx, :poi_len_array[idx], 1]

        for subidx in range(len(xs)):
            for subsubidx in range(data_type_num):
                poi2hr_grid_range_channel[subsubidx, xs[subidx], ys[subidx]] -= _delta_dc[idx, subsubidx] / poi_val_max

    return poi2hr_grid_range_channel


@jit(fastmath=True)
def fast_dis(xy1, xy2):
    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    dis = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    return dis


@jit(nopython=True)
def fast_is_out_of_energy(_cur_uav_pos_energy_hit_mec_cec_charge, min_value, uid):
    if _cur_uav_pos_energy_hit_mec_cec_charge[uid][2] > min_value:
        return False
    return True


@jit(nopython=True)
def fast_is_out_of_range(p, field_length_0, field_length_1):
    x, y = p
    if 0 <= x < field_length_0 and 0 <= y < field_length_1:
        return False
    return True


@jit(fastmath=True)
def fast_is_cross_circle(p1, p2, c, r):
    flag1 = fast_dis(p1, c) <= r
    flag2 = fast_dis(p2, c) <= r
    if flag1 and flag2:
        return False
    elif flag1 or flag2:
        return True
    else:
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0] * p2[1] - p2[0] * p1[1]
        dist = abs(A * c[0] + B * c[1] + C) / np.sqrt(A * A + B * B)
        if dist > r:
            return False
        angle1 = (c[0] - p1[0]) * (p2[0] - p1[0]) + (c[1] - p1[1]) * (p2[1] - p1[1])
        angle2 = (c[0] - p2[0]) * (p1[0] - p2[0]) + (c[1] - p2[1]) * (p1[1] - p2[1])
        if angle1 > 0 and angle2 > 0:
            return True
        else:
            return False


@jit(nopython=True)
def fast_cross_multiply(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]


@jit(nopython=True)
def fast_is_cross_segment(segment1, segment2):
    p1, p2 = segment1
    p3, p4 = segment2
    if max(p1[0], p2[0]) >= min(p3[0], p4[0]) and max(p3[0], p4[0]) >= min(p1[0], p2[0]) and max(p1[1],
                                                                                                 p2[1]) >= min(
        p3[1], p4[1]) and max(p3[1], p4[1]) >= min(p1[1], p2[1]):
        v_m = [p2[0] - p1[0], p2[1] - p1[1]]
        v_1 = [p3[0] - p1[0], p3[1] - p1[1]]
        v_2 = [p4[0] - p1[0], p4[1] - p1[1]]
        w_m = [p4[0] - p3[0], p4[1] - p3[1]]
        w_1 = [p1[0] - p3[0], p1[1] - p3[1]]
        w_2 = [p2[0] - p3[0], p2[1] - p3[1]]
        if fast_cross_multiply(v_1, v_m) * fast_cross_multiply(v_2, v_m) <= 0 and \
                fast_cross_multiply(w_1, w_m) * fast_cross_multiply(w_2, w_m) <= 0:
            return True
        else:
            return False
    else:
        return False


@jit(nopython=True)
def fast_is_hit_blk(p_s, p_e, blk_array, blk_len_array, blk_name_array):
    for idx in range(blk_array.shape[0]):
        blk_att = blk_name_array[idx]
        blk_key_list = blk_array[idx, :blk_len_array[idx]]
        if blk_att == 0:
            key_num = len(blk_key_list)
            for i in range(key_num):
                coord1 = blk_key_list[i]
                if i + 1 >= key_num:
                    coord2 = blk_key_list[0]
                else:
                    coord2 = blk_key_list[i + 1]
                if fast_is_cross_segment([coord1, coord2], [p_s, p_e]):
                    return True
        elif blk_att == 1:
            print('error')
    return False


@jit(nopython=True)
def fast_uav_move(uav_num, action, _cur_uav_pos_energy_hit_mec_cec_charge, uav_move_energy_consume_ratio, min_value,
                  field_length_0, field_length_1, blk_array, blk_len_array, blk_name_array):
    hit_list = []
    uav_move_dis = np.zeros(uav_num, dtype=np.float32)
    for uid in range(uav_num):
        if not fast_is_out_of_energy(_cur_uav_pos_energy_hit_mec_cec_charge, min_value, uid):
            dx, dy = action[uid][0], action[uid][1]
            new_x = _cur_uav_pos_energy_hit_mec_cec_charge[uid, 0] + dx
            new_y = _cur_uav_pos_energy_hit_mec_cec_charge[uid, 1] + dy

            if not fast_is_out_of_range([new_x, new_y], field_length_0, field_length_1) \
                    and not fast_is_hit_blk(
                [new_x - dx, new_y - dy],
                [new_x, new_y], blk_array, blk_len_array, blk_name_array):
                temp_dis = fast_dis([dx, dy], [0, 0])
                if temp_dis * uav_move_energy_consume_ratio <= \
                        _cur_uav_pos_energy_hit_mec_cec_charge[uid, 2]:
                    uav_move_dis[uid] += temp_dis
                    _cur_uav_pos_energy_hit_mec_cec_charge[uid, 0] = new_x
                    _cur_uav_pos_energy_hit_mec_cec_charge[uid, 1] = new_y
            else:
                _cur_uav_pos_energy_hit_mec_cec_charge[uid, 3] += 1
                hit_list.append(uid)
    return hit_list, uav_move_dis


@jit(nopython=True)
def fast_collect_data_or_charge(uav_collected_data, poi_num, uav_num, _cur_uav_pos_energy_hit_mec_cec_charge, min_value,
                                hit_list,
                                _cur_poi_pos_value,
                                uav_sensing_range, _cur_uav_visit_poi_time, uav_collect_speed_per_poi,
                                uav_collect_energy_consume_ratio, uav_types, action, uav_init_energy,
                                charge_station_pos, charge_sensing_range):
    uav_collect_array = np.ones(poi_num, dtype=np.float32)
    charge_list = []
    for uid in range(uav_num):
        is_charged = False
        if not fast_is_out_of_energy(_cur_uav_pos_energy_hit_mec_cec_charge, min_value, uid) and uid not in hit_list:
            ux = _cur_uav_pos_energy_hit_mec_cec_charge[uid, 0]
            uy = _cur_uav_pos_energy_hit_mec_cec_charge[uid, 1]
            for cid in range(charge_station_pos.shape[0]):
                dis = fast_dis(charge_station_pos[cid], [ux, uy])
                if dis < charge_sensing_range:
                    _cur_uav_pos_energy_hit_mec_cec_charge[uid, 6] += (uav_init_energy - \
                                                                       _cur_uav_pos_energy_hit_mec_cec_charge[
                                                                           uid, 2])
                    _cur_uav_pos_energy_hit_mec_cec_charge[uid, 2] = uav_init_energy
                    is_charged = True

            whether_break = False
            ux = _cur_uav_pos_energy_hit_mec_cec_charge[uid, 0]
            uy = _cur_uav_pos_energy_hit_mec_cec_charge[uid, 1]
            for pid in range(poi_num):
                if uav_collect_array[pid] == 1:
                    px = _cur_poi_pos_value[pid, 0]
                    py = _cur_poi_pos_value[pid, 1]
                    dis = fast_dis([px, py], [ux, uy])
                    if dis < uav_sensing_range:
                        _cur_uav_visit_poi_time[uid][pid] += 1
                        collected_data = min(_cur_poi_pos_value[pid, 2 + uav_types[uid]],
                                             uav_collect_speed_per_poi)
                        cec = (np.sum(uav_collected_data[uid]) + collected_data) * uav_collect_energy_consume_ratio
                        if cec > _cur_uav_pos_energy_hit_mec_cec_charge[uid, 2]:
                            collected_data = (_cur_uav_pos_energy_hit_mec_cec_charge[
                                                  uid, 2] / uav_collect_energy_consume_ratio) - np.sum(
                                uav_collected_data[uid])
                            whether_break = True
                        if collected_data > 0:
                            uav_collect_array[pid] = 0
                        uav_collected_data[uid, pid, uav_types[uid]] += collected_data
                        _cur_poi_pos_value[pid, 2 + uav_types[uid]] -= collected_data
                        if whether_break:
                            break
        charge_list.append(is_charged)
    return uav_collect_array, uav_collected_data, _cur_poi_pos_value, charge_list


@jit(nopython=True)
def go_fast2(poi2hr_grid_range_channel, poi_num, poi2hr_grid_range, poi_range_len, data_type_num, _cur_poi_pos_value,
             poi_value_max):
    for poi in range(poi_num):
        for l in range(poi_range_len[poi]):
            for d in range(data_type_num):
                poi2hr_grid_range_channel[d, poi2hr_grid_range[poi, l, 0], poi2hr_grid_range[poi, l, 1]] += \
                    _cur_poi_pos_value[poi, 2 + d] / poi_value_max
    return poi2hr_grid_range_channel


@jit(nopython=True)
def go_fast2_charge_station(charge_station2hr_grid_range_channel, charge_station_num, charge_station2hr_grid_range,
                            charge_station_range_len):
    for charge_station in range(charge_station_num):
        for l in range(charge_station_range_len[charge_station]):
            charge_station2hr_grid_range_channel[
                charge_station2hr_grid_range[charge_station, l, 0], charge_station2hr_grid_range[
                    charge_station, l, 1]] += 1
    return charge_station2hr_grid_range_channel


def gen_uav_sensor_type():
    _uav_types = np.zeros([ENV_CONF['uav_num']], dtype=np.long)
    for i in range(ENV_CONF['uav_num']):
        _uav_types[i] = i % ENV_CONF['data_type_num']
    return _uav_types
