from .conf import *


class SubLog:
    def __init__(self, ENV_CONF, log_path=None, process_id=0):
        self.ENV_CONF = ENV_CONF
        self.root_path = CONF['root_path']
        self.env_name = self.ENV_CONF['env_name']
        self.log_path = os.path.join(log_path, 'process_' + str(process_id))
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.method_name = CONF['method_name']
        self.episode_info = []

        self.eff_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.f_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.dcr_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.dcr_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.dcr_peo_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.ec_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.ec_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.mec_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.mec_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.cec_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.cec_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.hit_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.hit_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.co_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.co_uav_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.co_peo_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.uav_color = ['C' + str(_) for _ in range(self.ENV_CONF['uav_num'])]
        self.ecr_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.charge_list = np.zeros(CONF['train_iter'], dtype=np.float32)

    def gen_metrics_result(self, iter_id):
        final_st = self.episode_info[-1][0]
        env_init_poi_value = np.sum(final_st['env_init_poi_value'], axis=-1)
        pure_peo_dc = final_st['pure_peo_dc']
        peo_dc = final_st['cur_peo_dc']
        uav_pos_energy_hit_mec_cec_charge = final_st['cur_uav_pos_energy_hit_mec_cec_charge']
        uav_dc = final_st['cur_uav_dc']

        # f
        f = 0.0
        active_poi_id_list = np.nonzero(env_init_poi_value)
        uav_peo_collect_poi_ratio = (peo_dc + np.sum(uav_dc, axis=0))[active_poi_id_list] / env_init_poi_value[
            active_poi_id_list]
        square_of_sum = np.square(np.sum(uav_peo_collect_poi_ratio))
        sum_of_square = np.sum(np.square(uav_peo_collect_poi_ratio))
        if sum_of_square > self.ENV_CONF['min_value']:
            f = square_of_sum / sum_of_square / len(uav_peo_collect_poi_ratio)
        self.f_list[iter_id] = f

        # dcr
        dcr = (np.sum(peo_dc) + np.sum(uav_dc)) / np.sum(env_init_poi_value)
        self.dcr_list[iter_id] = dcr

        # dcr_uav
        self.dcr_uav_list[:, iter_id] = np.sum(uav_dc, axis=1) / np.sum(env_init_poi_value)

        # dcr_peo
        dcr_peo = np.sum(peo_dc) / np.sum(env_init_poi_value)
        self.dcr_peo_list[iter_id] = dcr_peo
        # ec
        ec = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 4]) + np.sum(uav_pos_energy_hit_mec_cec_charge[:, 5])
        self.ec_list[iter_id] = ec

        # ec_uav
        self.ec_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 4] + uav_pos_energy_hit_mec_cec_charge[:, 5]

        # mec
        mec = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 4])
        self.mec_list[iter_id] = mec

        # mec_uav
        self.mec_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 4]

        # cec
        cec = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 5])
        self.cec_list[iter_id] = cec

        # cec_uav
        self.cec_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 5]

        # hit
        hit = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 3])
        self.hit_list[iter_id] = hit

        # hit_uav
        self.hit_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 3]

        # co_peo
        co_peo = np.sum(peo_dc) / np.sum(pure_peo_dc)
        self.co_peo_list[iter_id] = co_peo

        # co_uav
        co_uav = (np.sum(uav_dc) - (np.sum(pure_peo_dc) - np.sum(peo_dc))) / (
                np.sum(env_init_poi_value) - np.sum(pure_peo_dc))
        self.co_uav_list[iter_id] = co_uav

        # co
        co = (np.sum(uav_dc) + np.sum(peo_dc) - (np.sum(pure_peo_dc) - np.sum(peo_dc))) / np.sum(env_init_poi_value)
        self.co_list[iter_id] = co

        # charge
        charge = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 6])
        self.charge_list[iter_id] = charge

        # ecr
        ecr = (np.sum(uav_pos_energy_hit_mec_cec_charge[:, 4]) + np.sum(uav_pos_energy_hit_mec_cec_charge[:, 5])) / (
                self.ENV_CONF['uav_num'] * self.ENV_CONF['uav_init_energy'] + np.sum(
            uav_pos_energy_hit_mec_cec_charge[:, 6]))
        self.ecr_list[iter_id] = ecr
        # eff
        eff = dcr * co * f / (mec / (self.ENV_CONF['uav_num'] * self.ENV_CONF['uav_init_energy'] + np.sum(
            uav_pos_energy_hit_mec_cec_charge[:, 6])))
        self.eff_list[iter_id] = eff

    def record_metrics_result(self):
        self.save_list(self.log_path + '/eff.npy', self.eff_list)
        self.save_list(self.log_path + '/f.npy', self.f_list)
        self.save_list(self.log_path + '/dcr.npy', self.dcr_list)
        self.save_list(self.log_path + '/dcr_peo.npy', self.dcr_peo_list)
        self.save_list(self.log_path + '/ec.npy', self.ec_list)
        self.save_list(self.log_path + '/mec.npy', self.mec_list)
        self.save_list(self.log_path + '/cec.npy', self.cec_list)
        self.save_list(self.log_path + '/hit.npy', self.hit_list)
        self.save_list(self.log_path + '/co.npy', self.co_list)
        self.save_list(self.log_path + '/co_uav.npy', self.co_uav_list)
        self.save_list(self.log_path + '/co_peo.npy', self.co_peo_list)
        self.save_list(self.log_path + '/dcr_uav.npy', self.dcr_uav_list)
        self.save_list(self.log_path + '/ec_uav.npy', self.ec_uav_list)
        self.save_list(self.log_path + '/mec_uav.npy', self.mec_uav_list)
        self.save_list(self.log_path + '/cec_uav.npy', self.cec_uav_list)
        self.save_list(self.log_path + '/hit_uav.npy', self.hit_uav_list)
        self.save_list(self.log_path + '/ecr.npy', self.ecr_list)
        self.save_list(self.log_path + '/charge.npy', self.charge_list)

    def save_list(self, path, result_list):
        np.save(path, result_list)

    def draw_ana(self, iter_id, reward_s):
        r = np.round(np.sum(reward_s[:, 0]), 2)
        dcr = np.round(self.dcr_list[iter_id], 2)
        ec = np.round(self.ec_list[iter_id], 2)
        f = np.round(self.f_list[iter_id], 2)
        eff = np.round(self.eff_list[iter_id], 2)
        charge = np.round(self.charge_list[iter_id], 2)
        title_str = str(iter_id) + ' r=' + str(r) + ' dcr=' + str(
            dcr) + '\n ec=' + str(ec) + ' f=' + str(f) + ' eff=' + str(eff) + ' charge=' + str(charge)

        reward_list = reward_s[:, 0, 0]
        save_path = self.log_path + '/reward_pdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        plt.plot(reward_list)
        plt.title('reward ' + title_str)
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()

        total_uav_collected_data_ratio_list = np.array(
            [np.sum(info[0]['cur_uav_dc'], axis=1) / np.sum(info[0]['env_init_poi_value']) for info in
             self.episode_info])
        save_path = self.log_path + '/dcr_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        plt.ylim(ymin=0, ymax=1)
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_collected_data_ratio_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.sum(total_uav_collected_data_ratio_list, axis=1), c='black', label='uav_all')
        plt.title('dcr ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()
        total_uav_mec_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 4] for info in
                                       self.episode_info])
        save_path = self.log_path + '/mec_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_mec_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.mean(total_uav_mec_list, axis=1), c='black', label='uav_all')
        plt.title('mec ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()
        total_uav_cec_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 5] for info in
                                       self.episode_info])
        save_path = self.log_path + '/cec_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_cec_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.mean(total_uav_cec_list, axis=1), c='black', label='uav_all')
        plt.title('cec ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()
        total_uav_e_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 2] for info in
                                     self.episode_info])
        save_path = self.log_path + '/e_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_e_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.mean(total_uav_e_list, axis=1), c='black', label='uav_all')
        plt.title('e ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()

        total_uav_hit_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 3] for info in
                                       self.episode_info])
        save_path = self.log_path + '/hit_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_hit_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.sum(total_uav_hit_list, axis=1), c='black', label='uav_all')
        plt.title('hit ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()

    def get_cmap(self, n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    def record_trace_se(self, iter_id, env):
        mpl.style.use('default')
        Fig = plt.figure(figsize=(10, 5))
        ax1 = Fig.add_subplot(121)
        ax2 = Fig.add_subplot(122)
        cm = plt.cm.get_cmap('RdYlBu_r')
        ax1.set_xlim(xmin=0, xmax=self.ENV_CONF['field_length'][0])
        ax1.set_ylim(ymin=0, ymax=self.ENV_CONF['field_length'][1])
        ax1.grid(True, linestyle='-.', color='r')
        ax2.set_xlim(xmin=0, xmax=self.ENV_CONF['field_length'][0])
        ax2.set_ylim(ymin=0, ymax=self.ENV_CONF['field_length'][1])
        ax2.grid(True, linestyle='-.', color='r')

        # draw peo trace
        peo_x_list = [[] for _ in range(self.ENV_CONF['peo_num'])]
        peo_y_list = [[] for _ in range(self.ENV_CONF['peo_num'])]
        for step_id, info in enumerate(self.episode_info):
            st = info[0]
            for peo_id in range(self.ENV_CONF['peo_num']):
                peo_x_list[peo_id].append(st['cur_peo_pos_value'][peo_id, 0])
                peo_y_list[peo_id].append(st['cur_peo_pos_value'][peo_id, 1])
        for peo_id in range(self.ENV_CONF['peo_num']):
            ax2.scatter(peo_x_list[peo_id], peo_y_list[peo_id], color='black', marker='.', s=1)

        # draw uav trace
        u_x_list = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_y_list = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_x_list_collect = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_y_list_collect = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_x_list_charge = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_y_list_charge = [[] for _ in range(self.ENV_CONF['uav_num'])]

        for step_id, info in enumerate(self.episode_info):
            st = info[0]
            for uav_id in range(self.ENV_CONF['uav_num']):
                u_x_list[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 0])
                u_y_list[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 1])
                if st['is_uav_collect'][uav_id]:
                    u_x_list_collect[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 0])
                    u_y_list_collect[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 1])
                else:
                    u_x_list_charge[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 0])
                    u_y_list_charge[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 1])

        for uav_id in range(self.ENV_CONF['uav_num']):
            color = self.uav_color[uav_id % len(self.uav_color)]
            ax2.plot(u_x_list[uav_id], u_y_list[uav_id], color=color, linewidth=1, alpha=0.5)
            ax2.scatter(u_x_list_collect[uav_id], u_y_list_collect[uav_id], color=color, marker='.', s=20, alpha=0.5)
            ax2.scatter(u_x_list_charge[uav_id], u_y_list_charge[uav_id], color=color, marker='+', s=50, )

        # draw blk
        blk_dict = env.blk_dict
        for blk in blk_dict:
            blk_att = blk_dict[blk][0]
            blk_key_list = blk_dict[blk][1]
            if blk_att == 'p':
                pgon1 = plt.Polygon(blk_key_list, color='brown', alpha=0.5)
                ax1.add_patch(pgon1)
                pgon2 = plt.Polygon(blk_key_list, color='brown', alpha=0.5)
                ax2.add_patch(pgon2)
            elif blk_att == 'r':
                circ1 = plt.Circle((blk_key_list[0][0], blk_key_list[0][1]), blk_key_list[1], color='brown',
                                   alpha=0.5)
                ax1.add_patch(circ1)
                circ2 = plt.Circle((blk_key_list[0][0], blk_key_list[0][1]), blk_key_list[1], color='brown',
                                   alpha=0.5)
                ax2.add_patch(circ2)
        # draw charge station
        for xy in env.charge_station_pos:
            circ1 = plt.Circle((xy[0], xy[1]), self.ENV_CONF['charge_sensing_range'], color='blue', fill=False)
            ax1.add_patch(circ1)
            circ2 = plt.Circle((xy[0], xy[1]), self.ENV_CONF['charge_sensing_range'], color='blue', fill=False)
            ax2.add_patch(circ2)

        # draw poi
        p_x = self.episode_info[0][0]['cur_poi_pos_value'][:, 0]
        p_y = self.episode_info[0][0]['cur_poi_pos_value'][:, 1]
        p_v_s = np.sum(self.episode_info[0][0]['cur_poi_pos_value'][:, 2:], axis=-1)
        p_v_e = np.sum(self.episode_info[-1][0]['cur_poi_pos_value'][:, 2:], axis=-1)
        p_v_s_norm = p_v_s / self.ENV_CONF['poi_value_max']
        p_v_e_norm = p_v_e / self.ENV_CONF['poi_value_max']
        ax1.scatter(p_x, p_y, c=p_v_s_norm, vmin=0, vmax=1, cmap=cm)
        ax2.scatter(p_x, p_y, c=p_v_e_norm, vmin=0, vmax=1, cmap=cm)
        for x, y in zip(p_x, p_y):
            circ1 = plt.Circle((x, y), self.ENV_CONF['uav_sensing_range'], color='red', alpha=0.1, fill=False)
            ax1.add_patch(circ1)
            circ2 = plt.Circle((x, y), self.ENV_CONF['uav_sensing_range'], color='red', alpha=0.1, fill=False)
            ax2.add_patch(circ2)
            circ1 = plt.Circle((x, y), self.ENV_CONF['peo_sensing_range'], color='grey', alpha=0.1, fill=False)
            ax1.add_patch(circ1)
            circ2 = plt.Circle((x, y), self.ENV_CONF['peo_sensing_range'], color='grey', alpha=0.1, fill=False)
            ax2.add_patch(circ2)

        eff = np.round(self.eff_list[iter_id], 2)
        f = np.round(self.f_list[iter_id], 2)
        dcr = np.round(self.dcr_list[iter_id], 2)
        dcr_uav = [self.dcr_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        dcr_uav = [np.round(np.mean(dcr_uav), 2), np.round(np.var(dcr_uav), 2)]
        dcr_peo = np.round(self.dcr_peo_list[iter_id], 2)
        ec = np.round(self.ec_list[iter_id], 2)
        ec_uav = [self.ec_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        ec_uav = [np.round(np.mean(ec_uav), 2), np.round(np.var(ec_uav), 2)]
        mec = np.round(self.mec_list[iter_id], 2)
        mec_uav = [self.mec_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        mec_uav = [np.round(np.mean(mec_uav), 2), np.round(np.var(mec_uav), 2)]
        cec = np.round(self.cec_list[iter_id], 2)
        cec_uav = [self.cec_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        cec_uav = [np.round(np.mean(cec_uav), 2), np.round(np.var(cec_uav), 2)]
        hit = np.round(self.hit_list[iter_id], 2)
        hit_uav = [self.hit_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        hit_uav = [np.round(np.mean(hit_uav), 2), np.round(np.var(hit_uav), 2)]
        co = np.round(self.co_list[iter_id], 2)
        co_uav = np.round(self.co_uav_list[iter_id], 2)
        co_peo = np.round(self.co_peo_list[iter_id], 2)

        ecr = np.round(self.ecr_list[iter_id], 2)
        charge = np.round(self.charge_list[iter_id], 2)

        title_str = 'iter: ' + str(iter_id) \
                    + ' eff: ' + str(eff) \
                    + ' f: ' + str(f) \
                    + ' dcr: ' + str(dcr) \
                    + ' dcr_uav: ' + str(dcr_uav) \
                    + ' ec: ' + str(ec) \
                    + ' mec: ' + str(mec) \
                    + ' cec: ' + str(cec) \
                    + ' hit: ' + str(hit) \
                    + ' co: ' + str(co) \
                    + '\n' \
                    + ' co_uav: ' + str(co_uav) \
                    + ' co_peo: ' + str(co_peo) \
                    + ' dcr_peo: ' + str(dcr_peo) \
                    + ' ec_uav: ' + str(ec_uav) \
                    + '\n' \
                    + ' mec_uav: ' + str(mec_uav) \
                    + ' cec_uav: ' + str(cec_uav) \
                    + ' hit_uav: ' + str(hit_uav) \
                    + ' ecr: ' + str(ecr) \
                    + ' charge: ' + str(charge)

        plt.suptitle(title_str)

        save_path = self.log_path + '/trace_se/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig.savefig(save_path + 'iter_id' + str(iter_id) + '.png')
        plt.close()
