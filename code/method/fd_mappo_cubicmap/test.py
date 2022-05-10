from .test_mainlog import *
from .sharestorage import *
from .test_subp import *
from env_method_set import *


def main(ENV_CONF, Env):
    log_time = ENV_METHOD_SET[ENV_CONF['env_name'] + '/' + CONF['method_name']]
    log_path = os.path.join(CONF['root_path'], ENV_CONF['env_name'], CONF['method_name'], log_time)
    mp.set_start_method("spawn", force=True)
    start_datetime = datetime.now()
    test_mainlog = TestMainLog(ENV_CONF, log_path)
    shared_rollout_list = []
    for env_id in range(CONF['test_num']):
        shared_rollout_list.append(ShareRolloutStorage(ENV_CONF))
    shared_ifdone_list = [mp.Value('b', False) for _ in range(CONF['test_num'])]
    init_poi_value_s = Env.gen_whole_init_poi_value()

    processes = []
    for env_id in range(CONF['test_num']):
        p = mp.Process(target=test_subp,
                       args=(env_id,
                             test_mainlog._log_root_path,
                             shared_rollout_list[env_id],
                             shared_ifdone_list[env_id],
                             init_poi_value_s,
                             ENV_CONF, Env,
                             )
                       )
        processes.append(p)
        p.start()

    for iter_id in range(1):
        iter_start_datetime = datetime.now()
        ################################## gen samples ####################################
        while True:
            global_ifdone = 0
            for shared_ifdone in shared_ifdone_list:
                if shared_ifdone.value:
                    global_ifdone += 1
                else:
                    break
            if global_ifdone == CONF['test_num']:
                ################################## test_mainlog work ####################################
                test_mainlog.load_envs_info()
                test_mainlog.record_metrics_result(iter_id)

                mean_eff = np.mean(test_mainlog.envs_info['eff'][iter_id])

                mean_f = np.mean(test_mainlog.envs_info['f'][iter_id])
                mean_dcr = np.mean(test_mainlog.envs_info['dcr'][iter_id])
                mean_ec = np.mean(test_mainlog.envs_info['ec'][iter_id])
                mean_mec = np.mean(test_mainlog.envs_info['mec'][iter_id])
                mean_cec = np.mean(test_mainlog.envs_info['cec'][iter_id])
                mean_hit = np.mean(test_mainlog.envs_info['hit'][iter_id])
                mean_co = np.mean(test_mainlog.envs_info['co'][iter_id])
                mean_co_peo = np.mean(test_mainlog.envs_info['co_peo'][iter_id])
                mean_ecr = np.mean(test_mainlog.envs_info['ecr'][iter_id])
                mean_charge = np.mean(test_mainlog.envs_info['charge'][iter_id])

                report_str = 'iter: ' + str(iter_id) \
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
                             + ' charge: ' + str(np.round(mean_charge, 5)) \
                             + ' co_peo: ' + str(np.round(mean_co_peo, 5))
                # ------------------------------------------------------------
                test_mainlog.record_report(report_str)
                print(report_str)
                print('pid', os.getpid(),
                      '\'' + log_path + '\',')
                # -------------------------------------------
                cur_datetime = datetime.now()
                print('Test time:', cur_datetime - start_datetime, 'iter duration:',
                      cur_datetime - iter_start_datetime,
                      '\n')
                for shared_ifdone in shared_ifdone_list:
                    shared_ifdone.value = False
                break

    for p in processes:
        p.join()
