from .mainlog import *
from .ppo import *
from .storage import *
from .sharestorage import *
from .subp import *


def main(ENV_CONF, Env):
    mp.set_start_method("spawn", force=True)
    start_datetime = datetime.now()
    mainlog = MainLog(ENV_CONF)
    mainlog.record_env_conf()
    mainlog.record_conf()
    ac_list = [Policy(_).to(CONF['device']) for _ in range(ENV_CONF['uav_num'])]

    for ac in ac_list:
        ac.eval()
    agent = PPO(ac_list=ac_list)
    for uid in range(ENV_CONF['uav_num']):
        ac = ac_list[uid]
        mainlog.save_cur_model(ac, uid)
    rollouts = RolloutStorage(ENV_CONF)
    shared_rollout_list = []
    for env_id in range(CONF['env_num']):
        shared_rollout_list.append(ShareRolloutStorage(ENV_CONF))
    cur_datetime = datetime.now()
    print('process time:', cur_datetime - start_datetime)
    shared_ifdone_list = [mp.Value('b', False) for _ in range(CONF['env_num'])]
    init_poi_value_s = Env.gen_whole_init_poi_value()
    processes = []
    for env_id in range(CONF['env_num']):
        p = mp.Process(target=subp,
                       args=(env_id,
                             mainlog._log_path,
                             shared_rollout_list[env_id],
                             shared_ifdone_list[env_id],
                             init_poi_value_s,
                             ENV_CONF, Env,
                             )
                       )
        processes.append(p)
        p.start()

    eff_list = []
    max_avg_eff = 0
    max_avg_eff_iter = 0
    for iter_id in range(CONF['train_iter']):
        iter_start_datetime = datetime.now()
        ################################## gen samples ####################################
        while True:
            global_ifdone = 0
            for shared_ifdone in shared_ifdone_list:
                if shared_ifdone.value:
                    global_ifdone += 1
                else:
                    break
            if global_ifdone == CONF['env_num']:
                for env_id in range(CONF['env_num']):
                    rollouts.insert(shared_rollout_list[env_id], env_id)
                ################################## save best model ####################################
                mainlog.load_envs_info()
                mean_eff = np.mean(mainlog.envs_info['eff'][iter_id])

                eff_list.append(mean_eff)
                avg_eff = np.mean(eff_list[-50:])
                if avg_eff > max_avg_eff:
                    max_avg_eff = avg_eff
                    max_avg_eff_iter = iter_id
                    for uid in range(ENV_CONF['uav_num']):
                        ac = ac_list[uid]
                        mainlog.save_model(ac, uid)

                ################################## update params ####################################
                for ac in ac_list:
                    ac.train()
                value_loss_per_sample, action_loss_per_sample, \
                dist_entropy_per_sample, loss_per_sample = agent.update(rollouts, iter_id)
                for ac in ac_list:
                    ac.eval()
                ################################## mainlog work ####################################
                mainlog.record_metrics_result(iter_id)
                mainlog.record_loss(value_loss_per_sample, action_loss_per_sample, dist_entropy_per_sample,
                                    loss_per_sample)

                for uid in range(ENV_CONF['uav_num']):
                    ac = ac_list[uid]
                    mainlog.save_cur_model(ac, uid)
                mean_f = np.mean(mainlog.envs_info['f'][iter_id])
                mean_dcr = np.mean(mainlog.envs_info['dcr'][iter_id])
                mean_ec = np.mean(mainlog.envs_info['ec'][iter_id])
                mean_mec = np.mean(mainlog.envs_info['mec'][iter_id])
                mean_cec = np.mean(mainlog.envs_info['cec'][iter_id])
                mean_hit = np.mean(mainlog.envs_info['hit'][iter_id])
                mean_co = np.mean(mainlog.envs_info['co'][iter_id])
                mean_ecr = np.mean(mainlog.envs_info['ecr'][iter_id])
                mean_charge = np.mean(mainlog.envs_info['charge'][iter_id])
                report_str = 'iter: ' + str(iter_id) \
                             + ' max_avg_eff: ' + str(np.round(max_avg_eff, 5)) \
                             + ' max_avg_eff_iter: ' + str(max_avg_eff_iter) \
                             + ' eff: ' + str(np.round(mean_eff, 5)) \
                             + ' f: ' + str(np.round(mean_f, 5)) \
                             + ' dcr: ' + str(np.round(mean_dcr, 5)) \
                             + ' ec: ' + str(np.round(mean_ec, 5)) \
                             + '\n' \
                             + ' mec: ' + str(np.round(mean_mec, 5)) \
                             + ' cec: ' + str(np.round(mean_cec, 5)) \
                             + ' hit: ' + str(np.round(mean_hit, 5)) \
                             + ' co: ' + str(np.round(mean_co, 5)) \
                             + ' ecr: ' + str(np.round(mean_ecr, 5)) \
                             + ' charge: ' + str(np.round(mean_charge, 5))
                # ------------------------------------------------------------
                mainlog.record_report(report_str)
                print(report_str)
                print('pid', os.getpid(), CONF['device'],
                      '\'' + ENV_CONF['env_name'] + '/' + CONF[
                          'method_name'] + '\': \'' + mainlog.get_start_time() + '\',')
                cur_datetime = datetime.now()
                print('process time:', cur_datetime - start_datetime, 'iter duration:',
                      cur_datetime - iter_start_datetime,
                      '\n')
                for shared_ifdone in shared_ifdone_list:
                    shared_ifdone.value = False
                break

    for p in processes:
        p.join()
