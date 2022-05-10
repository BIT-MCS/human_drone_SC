from util import *

dataset_name = 'NCSU'
ENV_CONF = {
    # field conf
    'ref_coordx': -180,  # meter
    'ref_coordy': 935,  # meter
    'ref_lon': -78.675481,
    'ref_lat': 35.780790,
    'coordx_per_lon': 90148.59146281201,  # meter
    'coordy_per_lat': 111122.19769899677,  # meter
    'core_lon_min': -78.687777,
    'core_lon_max': -78.665273,
    'core_lat_min': 35.775175,
    'core_lat_max': 35.791285,
    'field_length': [2028.7039022791, 1790.1786049308],  # meter
    'poi_num': 104,

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
    'peo_num': 35,
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
