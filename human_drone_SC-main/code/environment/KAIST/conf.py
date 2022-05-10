from util import *

dataset_name = 'KAIST'
ENV_CONF = {
    # field conf
    'ref_coordx': 1395,  # meter
    'ref_coordy': 997,  # meter
    'ref_lon': 127.379589,
    'ref_lat': 36.374767,
    'coordx_per_lon': 89470.20036776649,  # meter
    'coordy_per_lat': 111122.19769907878,  # meter
    'core_lon_min': 127.355114,
    'core_lon_max': 127.370748,
    'core_lat_min': 36.364889,
    'core_lat_max': 36.376781,
    'field_length': [1398.7771125501713, 1321.4651750377916],  # meter
    'poi_num': 131,

    # charge station conf
    'charge_station_num': 6,
    'charge_sensing_range': 20,  # meter
    'charge_station_dict_path': './environment/' + dataset_name + '/charge_station_dict.npy',

    # poi conf
    'poi_dict_path': './environment/' + dataset_name + '/poi_dict.npy',
    'poi_value_max': 3 * 4,
    'poi_value_min': 2 * 4,

    # peo conf
    'peo_dict_path': './environment/' + dataset_name + '/user_dict.npy',
    'peo_num': 92,
    'peo_value': 1,
    'peo_collect_speed_per_poi': 0.25,
    'peo_sensing_range': 50,  # meter
    'record_time_interval': 30,  # second
    'epoch_time_range': 30,  # second

    # block conf
    'blk_dict_path': './environment/' + dataset_name + '/block_dict.npy',

    # uav conf
    'uav_init_pos': 'center',
    'uav_num': 6,
    'uav_collect_speed_per_poi': 5,
    'uav_sensing_range': 60,  # meter
    'uav_dis_max': 100.,  # meter
    'uav_init_energy': 20.,
    'uav_move_energy_consume_ratio': 0.01,
    'uav_collect_energy_consume_ratio': 0.1,

    # other conf
    'positive_factor': 3,
    'penalty_factor': 0.1,
    'min_value': 1e-5,
    'max_step': 100,
    'env_name': dataset_name,
    'dataset_name': dataset_name,
    'data_type_num': 1,
    'charge_factor': 10.,
    'charge_min_factor': 0.3,
}
