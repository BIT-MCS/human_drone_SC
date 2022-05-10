from .conf import *


class MainLog:
    v_loss_list = []
    a_loss_list = []
    entropy_list = []
    loss_list = []
    envs_info = {}

    def __init__(self, ENV_CONF):
        self.ENV_CONF = ENV_CONF
        self.root_path = CONF['root_path']
        self.env_name = self.ENV_CONF['env_name']
        self.method_name = CONF['method_name']
        self._time = str(time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime()))
        self._log_path = os.path.join(self.root_path, self.env_name, self.method_name, self._time)
        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path)

    def get_start_time(self):
        return self._time

    def record_report(self, report_str):
        self._report_path = self._log_path + '/report.txt'
        f = open(self._report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def record_env_conf(self):
        conf = self.ENV_CONF
        self._env_conf_path = self._log_path + '/env_conf.txt'
        with open(self._env_conf_path, 'w') as f:
            lines = []
            for k in conf:
                lines.append(str(k) + '\t' + str(conf[k]) + '\n')
            f.writelines(lines)

    def record_conf(self):
        conf = CONF
        self._conf_path = self._log_path + '/conf.txt'
        with open(self._conf_path, 'w') as f:
            lines = []
            for k in conf:
                lines.append(str(k) + '\t' + str(conf[k]) + '\n')
            f.writelines(lines)

    def load_envs_info(self):
        self.envs_info['eff'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'eff.npy'), allow_pickle=True) for env_id in
            range(CONF['env_num'])]

        self.envs_info['f'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'f.npy'), allow_pickle=True) for env_id in
            range(CONF['env_num'])]

        self.envs_info['dcr'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'dcr.npy'), allow_pickle=True) for env_id in
            range(CONF['env_num'])]

        self.envs_info['dcr_peo'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'dcr_peo.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['ec'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'ec.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['mec'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'mec.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['cec'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'cec.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['hit'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'hit.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['co'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'co.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['co_uav'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'co_uav.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['co_peo'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'co_peo.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['ecr'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'ecr.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['charge'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'charge.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]
        # --------------------------------

        for uav_id in range(self.ENV_CONF['uav_num']):
            self.envs_info['dcr_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'dcr_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]
            self.envs_info['ec_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'ec_uav.npy'), allow_pickle=True)[uav_id]
                for env_id in range(CONF['env_num'])]
            self.envs_info['mec_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'mec_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]
            self.envs_info['cec_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'cec_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]
            self.envs_info['hit_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'hit_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]

        for key in self.envs_info:
            self.envs_info[key] = np.concatenate([np.expand_dims(arr, axis=1) for arr in self.envs_info[key]], axis=1)

    def record_metrics_result(self, iter_id):
        np.save(self._log_path + '/eff.npy', self.envs_info['eff'][:iter_id + 1])
        np.save(self._log_path + '/f.npy', self.envs_info['f'][:iter_id + 1])
        np.save(self._log_path + '/dcr.npy', self.envs_info['dcr'][:iter_id + 1])
        np.save(self._log_path + '/dcr_peo.npy', self.envs_info['dcr_peo'][:iter_id + 1])
        np.save(self._log_path + '/ec.npy', self.envs_info['ec'][:iter_id + 1])
        np.save(self._log_path + '/ecr.npy', self.envs_info['ecr'][:iter_id + 1])
        np.save(self._log_path + '/charge.npy', self.envs_info['charge'][:iter_id + 1])
        np.save(self._log_path + '/mec.npy', self.envs_info['mec'][:iter_id + 1])
        np.save(self._log_path + '/cec.npy', self.envs_info['cec'][:iter_id + 1])
        np.save(self._log_path + '/hit.npy', self.envs_info['hit'][:iter_id + 1])
        np.save(self._log_path + '/co.npy', self.envs_info['co'][:iter_id + 1])
        np.save(self._log_path + '/co_uav.npy', self.envs_info['co_uav'][:iter_id + 1])
        np.save(self._log_path + '/co_peo.npy', self.envs_info['co_peo'][:iter_id + 1])

        for uav_id in range(self.ENV_CONF['uav_num']):
            np.save(self._log_path + '/dcr_uav' + str(uav_id) + '.npy',
                    self.envs_info['dcr_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/ec_uav' + str(uav_id) + '.npy',
                    self.envs_info['ec_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/mec_uav' + str(uav_id) + '.npy',
                    self.envs_info['mec_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/cec_uav' + str(uav_id) + '.npy',
                    self.envs_info['cec_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/hit_uav' + str(uav_id) + '.npy',
                    self.envs_info['hit_uav' + str(uav_id)][:iter_id + 1])

    def record_loss(self, v_loss, a_loss, entropy, loss):
        self.v_loss_list.append(v_loss)
        self.a_loss_list.append(a_loss)
        self.entropy_list.append(entropy)
        self.loss_list.append(loss)

        np.save(self._log_path + '/v_loss.npy', self.v_loss_list)
        np.save(self._log_path + '/a_loss.npy', self.a_loss_list)
        np.save(self._log_path + '/entropy.npy', self.entropy_list)
        np.save(self._log_path + '/loss.npy', self.loss_list)

    def save_model(self, model, uid):
        self._model_path = self._log_path + '/model_' + str(uid) + '.pth'
        torch.save(model.state_dict(), self._model_path)
        print('model has been saved to ', self._model_path)

    def save_cur_model(self, model, uid):
        self._model_path = self._log_path + '/tmp_model_' + str(uid) + '.pth'
        torch.save(model.state_dict(), self._model_path)
